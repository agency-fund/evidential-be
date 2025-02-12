from contextlib import asynccontextmanager
from typing import Annotated
import uuid

from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import to_jsonable_python
from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Depends,
    Query,
    Response,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from xngin.apiserver.api_types import (
    AudienceSpec,
    BalanceCheck,
    DesignSpec,
)
from xngin.apiserver.dependencies import (
    datasource_dependency,
    gsheet_cache,
    xngin_db_session,
)
from xngin.apiserver.gsheet_cache import GSheetCache
from xngin.apiserver.routers.experiments_api import (
    CommonQueryParams,
    do_assignment,
    get_participants_config_and_schema,
)
from xngin.apiserver.settings import (
    Datasource,
)
from xngin.apiserver.models.tables import (
    Experiment,
    ArmAssignment,
    ExperimentState,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


router = APIRouter(
    lifespan=lifespan,
    prefix="",
)


class ExperimentsBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateExperimentRequest(ExperimentsBaseModel):
    design_spec: DesignSpec
    audience_spec: AudienceSpec


class AssignSummary(ExperimentsBaseModel):
    balance_check: BalanceCheck
    sample_size: int


class ExperimentConfig(ExperimentsBaseModel):
    """Representation of our stored Experiment information."""

    datasource_id: str
    state: Annotated[
        ExperimentState, Field(description="Current state of this experiment.")
    ]
    design_spec: DesignSpec
    audience_spec: AudienceSpec
    assign_summary: AssignSummary
    # created_at: datetime.datetime
    # updated_at: datetime.datetime


class CreateExperimentWithAssignmentResponse(ExperimentConfig):
    """Same as the request but with uuids filled for the experiment and arms, and summary info on the assignment."""


class GetExperimentResponse(ExperimentConfig):
    pass


class ListExperimentsResponse(ExperimentsBaseModel):
    items: list[ExperimentConfig]


@router.post(
    "/experiments/with-assignment",
    summary="Create a pending experiment and save its assignments to the database. User will still need to /experiments/<id>/commit the experiment after reviewing assignment balance summary.",
)
def create_experiment_with_assignment(
    # TODO: add authorization support here and all other endpoints
    # user: Annotated[User, Depends(user_from_token)],
    body: CreateExperimentRequest,
    chosen_n: Annotated[
        int, Query(..., description="Number of participants to assign.")
    ],
    gsheets: Annotated[GSheetCache, Depends(gsheet_cache)],
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[Session, Depends(xngin_db_session)],
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
    random_state: Annotated[
        int | None,
        Query(
            description="Specify a random seed for reproducibility.",
            include_in_schema=False,
        ),
    ] = None,
) -> CreateExperimentWithAssignmentResponse:
    """Creates an experiment and saves its assignments to the database."""
    config = datasource.config
    commons = CommonQueryParams(
        participant_type=body.audience_spec.participant_type, refresh=refresh
    )
    participants_cfg, schema = get_participants_config_and_schema(
        commons, config, gsheets
    )
    with config.dbsession() as dwh_session:
        # TODO: directly create ArmAssignments from the pd dataframe instead
        assignment_response = do_assignment(
            session=dwh_session,
            participant=participants_cfg,
            supports_reflection=config.supports_reflection(),
            body=body,
            chosen_n=chosen_n,
            id_field=schema.get_unique_id_field(),
            random_state=random_state,
        )

        # Create experiment record
        # TODO: generate the id ourselves. Enforce that it is not present in the request.
        assign_summary = AssignSummary(
            balance_check=assignment_response.balance_check,
            sample_size=assignment_response.sample_size,
        )
        experiment = Experiment(
            id=body.design_spec.experiment_id,
            datasource_id=datasource.id,
            state=ExperimentState.ASSIGNED,
            # TODO? just use @field_serializer in design_spec always for uuids and datetimes?
            design_spec=to_jsonable_python(body.design_spec),
            audience_spec=body.audience_spec.model_dump(),
            assign_summary=assign_summary.model_dump(),
        )
        xngin_session.add(experiment)

        # Create assignment records
        for assignment in assignment_response.assignments:
            # TODO: bulk insert https://docs.sqlalchemy.org/en/20/orm/queryguide/dml.html#orm-queryguide-bulk-insert {"dml_strategy": "raw"}
            db_assignment = ArmAssignment(
                experiment_id=experiment.id,
                participant_type=body.audience_spec.participant_type,
                participant_id=assignment.participant_id,
                arm_id=assignment.arm_id,
                arm_name=assignment.arm_name,
                strata=to_jsonable_python(assignment.strata),
            )
            xngin_session.add(db_assignment)

        xngin_session.commit()

    # TODO: backfill server-side-generated uuids
    return CreateExperimentWithAssignmentResponse(
        datasource_id=datasource.id,
        state=experiment.state,
        design_spec=experiment.design_spec,
        audience_spec=experiment.audience_spec,
        assign_summary=assign_summary,
    )


@router.post(
    "/experiments/{experiment_id}/commit",
    summary="Marks any ASSIGNED experiment as COMMITTED.",
)
def commit_experiment(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[Session, Depends(xngin_db_session)],
    experiment_id: uuid.UUID,
):
    experiment = xngin_session.scalars(
        select(Experiment).where(
            Experiment.id == experiment_id, Experiment.datasource_id == datasource.id
        )
    ).one_or_none()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found"
        )
    if experiment.state not in {ExperimentState.ASSIGNED}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Invalid state: {experiment.state}",
        )
    if experiment.state == ExperimentState.COMMITTED:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED)

    experiment.state = ExperimentState.COMMITTED
    xngin_session.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


# TODO: support filters on state
# TODO: support pagination, e.g. https://github.com/uriyyo/fastapi-pagination
@router.get(
    "/experiments",
    summary="Fetch experiment meta data (design & assignment specs) for the given id.",
)
def list_experiments(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[Session, Depends(xngin_db_session)],
) -> ListExperimentsResponse:
    stmt = (
        select(Experiment).where(Experiment.datasource_id == datasource.id)
        # TODO: order by start_date at least.
    )
    result = xngin_session.execute(stmt)
    experiments = result.scalars().all()
    return ListExperimentsResponse(
        items=[
            ExperimentConfig(
                datasource_id=e.datasource_id,
                state=e.state,
                design_spec=e.design_spec,
                audience_spec=e.audience_spec,
                assign_summary=e.assign_summary,
            )
            for e in experiments
        ]
    )


@router.get(
    "/experiments/{experiment_id}",
    summary="Fetch experiment meta data (design & assignment specs) for the given id.",
)
def get_experiment(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[Session, Depends(xngin_db_session)],
    experiment_id: uuid.UUID,
) -> GetExperimentResponse:
    experiment = xngin_session.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found"
        )

    return ExperimentConfig(
        datasource_id=experiment.datasource_id,
        state=experiment.state,
        design_spec=experiment.design_spec,
        audience_spec=experiment.audience_spec,
        assign_summary=experiment.assign_summary,
    )


# TODO: implement with an output=csv|json[default] query param
@router.get(
    "/experiments/{experiment_id}/assignments",
    summary="Fetch list of participant=>arm assignments for the given experiment id.",
)
def get_experiment_assignments(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[Session, Depends(xngin_db_session)],
    experiment_id: str,
) -> GetExperimentResponse:
    experiment = xngin_session.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found"
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
