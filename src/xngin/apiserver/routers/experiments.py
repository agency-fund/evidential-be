from contextlib import asynccontextmanager
from typing import Annotated
import uuid

from pydantic import BaseModel, ConfigDict, Field
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
    Assignment,
    AudienceSpec,
    BalanceCheck,
    DesignSpec,
    PowerResponse,
    Strata,
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
    power_analyses: PowerResponse | None = None


class AssignSummary(ExperimentsBaseModel):
    """Key pieces of an AssignResponse without the assignments."""

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
    power_analyses: PowerResponse | None
    assign_summary: AssignSummary


class CreateExperimentWithAssignmentResponse(ExperimentConfig):
    """Same as the request but with uuids filled for the experiment and arms, and summary info on the assignment."""


class GetExperimentResponse(ExperimentConfig):
    pass


class ListExperimentsResponse(ExperimentsBaseModel):
    items: list[ExperimentConfig]


class GetExperimentAssigmentsResponse(ExperimentsBaseModel):
    """Describes assignments for all participants and balance test results."""

    balance_check: BalanceCheck

    experiment_id: uuid.UUID
    sample_size: int
    assignments: list[Assignment]


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
            start_date=body.design_spec.start_date,
            end_date=body.design_spec.end_date,
            design_spec=body.design_spec.model_dump(mode="json"),
            audience_spec=body.audience_spec.model_dump(mode="json"),
            power_analyses=body.power_analyses.model_dump(mode="json")
            if body.power_analyses
            else None,
            assign_summary=assign_summary.model_dump(mode="json"),
        )  # .set_design_spec(body.design_spec)
        xngin_session.add(experiment)

        # Create assignment records
        for assignment in assignment_response.assignments:
            # TODO: bulk insert https://docs.sqlalchemy.org/en/20/orm/queryguide/dml.html#orm-queryguide-bulk-insert {"dml_strategy": "raw"}
            db_assignment = ArmAssignment(
                experiment_id=experiment.id,
                participant_type=body.audience_spec.participant_type,
                participant_id=assignment.participant_id,
                arm_id=assignment.arm_id,
                strata=[s.model_dump(mode="json") for s in assignment.strata],
            )
            xngin_session.add(db_assignment)

        xngin_session.commit()

    # TODO: generate uuids here instead of trusting  the client request
    return CreateExperimentWithAssignmentResponse(
        datasource_id=datasource.id,
        state=experiment.state,
        design_spec=experiment.design_spec,
        audience_spec=experiment.audience_spec,
        power_analyses=experiment.power_analyses,
        assign_summary=assign_summary,
    )


def get_experiment_or_raise(
    xngin_session: Session, experiment_id: uuid.UUID, datasource_id: str
):
    experiment = xngin_session.scalars(
        select(Experiment).where(
            Experiment.id == experiment_id, Experiment.datasource_id == datasource_id
        )
    ).one_or_none()
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found"
        )
    return experiment


@router.post(
    "/experiments/{experiment_id}/commit",
    summary="Marks any ASSIGNED experiment as COMMITTED.",
)
def commit_experiment(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[Session, Depends(xngin_db_session)],
    experiment_id: uuid.UUID,
):
    experiment = get_experiment_or_raise(xngin_session, experiment_id, datasource.id)
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


@router.get(
    "/experiments",
    summary="Fetch experiment meta data (design & assignment specs) for the given id.",
)
def list_experiments(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[Session, Depends(xngin_db_session)],
) -> ListExperimentsResponse:
    stmt = (
        select(Experiment)
        .where(Experiment.datasource_id == datasource.id)
        .order_by(Experiment.created_at.desc())
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
                power_analyses=e.power_analyses,
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
    experiment = get_experiment_or_raise(xngin_session, experiment_id, datasource.id)
    return ExperimentConfig(
        datasource_id=experiment.datasource_id,
        state=experiment.state,
        design_spec=experiment.design_spec,
        audience_spec=experiment.audience_spec,
        power_analyses=experiment.power_analyses,
        assign_summary=experiment.assign_summary,
    )


# TODO: add a query param to include strata; default to false
@router.get(
    "/experiments/{experiment_id}/assignments",
    summary="Fetch list of participant=>arm assignments for the given experiment id.",
)
def get_experiment_assignments(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[Session, Depends(xngin_db_session)],
    experiment_id: uuid.UUID,
) -> GetExperimentAssigmentsResponse:
    experiment = get_experiment_or_raise(xngin_session, experiment_id, datasource.id)

    # Get sample size and and balance check from the assign_summary
    assign_summary = experiment.assign_summary
    balance_check = BalanceCheck.model_validate(assign_summary["balance_check"])

    # Map arm IDs to names
    design_spec = DesignSpec.model_validate(experiment.design_spec)
    arm_id_to_name = {str(arm.arm_id): arm.arm_name for arm in design_spec.arms}

    # Convert ArmAssignment models to Assignment API types
    assignments = [
        Assignment(
            participant_id=arm_assignment.participant_id,
            arm_id=arm_assignment.arm_id,
            arm_name=arm_id_to_name[str(arm_assignment.arm_id)],
            strata=[Strata.model_validate(s) for s in arm_assignment.strata],
        )
        for arm_assignment in experiment.arm_assignments
    ]

    return GetExperimentAssigmentsResponse(
        balance_check=balance_check,
        experiment_id=experiment.id,
        sample_size=assign_summary["sample_size"],
        assignments=assignments,
    )
