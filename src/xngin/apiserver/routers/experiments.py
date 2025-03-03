from itertools import batched
import uuid
from contextlib import asynccontextmanager
from typing import Annotated
import csv
import io

from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Depends,
    Query,
    Response,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from xngin.apiserver.api_types import (
    Assignment,
    DesignSpec,
    Strata,
)
from xngin.apiserver.dependencies import (
    datasource_dependency,
    gsheet_cache,
    xngin_db_session,
)
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.gsheet_cache import GSheetCache
from xngin.apiserver.models.enums import ExperimentState
from xngin.apiserver.models.tables import (
    ArmAssignment,
    Experiment,
)
from xngin.apiserver.routers.experiments_api import (
    CommonQueryParams,
    do_assignment,
    get_participants_config_and_schema,
)
from xngin.apiserver.routers.experiments_api_types import (
    CreateExperimentRequest,
    AssignSummary,
    ExperimentConfig,
    CreateExperimentWithAssignmentResponse,
    GetExperimentResponse,
    ListExperimentsResponse,
    GetExperimentAssigmentsResponse,
)
from xngin.apiserver.settings import (
    Datasource,
    DatasourceConfig,
    ParticipantsConfig,
)
from xngin.schema.schema_types import ParticipantsSchema


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


router = APIRouter(
    lifespan=lifespan,
    prefix="",
)


@router.post(
    "/experiments/with-assignment",
    summary="Create a pending experiment and save its assignments to the database. User will still need to /experiments/<id>/commit the experiment after reviewing assignment balance summary.",
)
def create_experiment_with_assignment_sl(
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
    if body.design_spec.uuids_are_present():
        raise LateValidationError("Invalid DesignSpec: UUIDs must not be set.")

    config = datasource.config
    commons = CommonQueryParams(
        participant_type=body.audience_spec.participant_type, refresh=refresh
    )
    participants_cfg, schema = get_participants_config_and_schema(
        commons, config, gsheets
    )

    return create_experiment_with_assignment_impl(
        xngin_session,
        datasource,
        body,
        participants_cfg,
        config,
        schema,
        random_state,
        chosen_n,
    )


def create_experiment_with_assignment_impl(
    xngin_session: Session,
    datasource: Datasource,
    body: CreateExperimentRequest,
    participants_cfg: ParticipantsConfig,
    config: DatasourceConfig,
    schema: ParticipantsSchema,
    random_state: int | None,
    chosen_n: int,
):
    # First generate uuids for the experiment and arms, which do_assignment needs.
    body.design_spec.experiment_id = uuid.uuid4()
    for arm in body.design_spec.arms:
        arm.arm_id = uuid.uuid4()

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
            quantiles=4,  # TODO(qixotic)
            stratum_id_name=None,
        )

        # Create experiment record
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
    status_code=status.HTTP_204_NO_CONTENT,
)
def commit_experiment_sl(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[Session, Depends(xngin_db_session)],
    experiment_id: uuid.UUID,
):
    experiment = get_experiment_or_raise(xngin_session, experiment_id, datasource.id)
    return commit_experiment_impl(xngin_session, experiment)


def commit_experiment_impl(xngin_session: Session, experiment: Experiment):
    if experiment.state == ExperimentState.COMMITTED:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED)
    if experiment.state not in {ExperimentState.ASSIGNED}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Invalid state: {experiment.state}",
        )

    experiment.state = ExperimentState.COMMITTED
    xngin_session.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/experiments/{experiment_id}/abandon",
    summary="Marks any DESIGNING or ASSIGNED experiment as ABANDONED.",
    status_code=status.HTTP_204_NO_CONTENT,
)
def abandon_experiment_sl(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[Session, Depends(xngin_db_session)],
    experiment_id: uuid.UUID,
):
    experiment = get_experiment_or_raise(xngin_session, experiment_id, datasource.id)
    return abandon_experiment_impl(xngin_session, experiment)


def abandon_experiment_impl(xngin_session: Session, experiment: Experiment):
    if experiment.state == ExperimentState.ABANDONED:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED)
    if experiment.state not in {ExperimentState.DESIGNING, ExperimentState.ASSIGNED}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Invalid state: {experiment.state}",
        )

    experiment.state = ExperimentState.ABANDONED
    xngin_session.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/experiments",
    summary="Fetch experiment meta data (design & assignment specs) for the given id.",
)
def list_experiments_sl(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[Session, Depends(xngin_db_session)],
) -> ListExperimentsResponse:
    return list_experiments_impl(xngin_session, datasource)


def list_experiments_impl(xngin_session: Session, datasource: Datasource):
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
                design_spec=e.get_design_spec(),
                audience_spec=e.get_audience_spec(),
                power_analyses=e.get_power_analyses(),
                assign_summary=e.get_assign_summary(),
            )
            for e in experiments
        ]
    )


@router.get(
    "/experiments/{experiment_id}",
    summary="Fetch experiment meta data (design & assignment specs) for the given id.",
)
def get_experiment_sl(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[Session, Depends(xngin_db_session)],
    experiment_id: uuid.UUID,
) -> GetExperimentResponse:
    experiment = get_experiment_or_raise(xngin_session, experiment_id, datasource.id)
    return ExperimentConfig(
        datasource_id=experiment.datasource_id,
        state=experiment.state,
        design_spec=experiment.get_design_spec(),
        audience_spec=experiment.get_audience_spec(),
        power_analyses=experiment.get_power_analyses(),
        assign_summary=experiment.get_assign_summary(),
    )


# TODO: add a query param to include strata; default to false
@router.get(
    "/experiments/{experiment_id}/assignments",
    summary="Fetch list of participant=>arm assignments for the given experiment id.",
)
def get_experiment_assignments_sl(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[Session, Depends(xngin_db_session)],
    experiment_id: uuid.UUID,
) -> GetExperimentAssigmentsResponse:
    experiment = get_experiment_or_raise(xngin_session, experiment_id, datasource.id)

    return get_experiment_assignments_impl(experiment)


def get_experiment_assignments_impl(experiment: Experiment):
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
        balance_check=experiment.get_balance_check(),
        experiment_id=experiment.id,
        sample_size=experiment.assign_summary["sample_size"],
        assignments=assignments,
    )


def experiment_assignments_to_csv_generator(experiment: Experiment):
    """Generator function to yield CSV rows of experiment assignments as strings"""
    # Map arm IDs to names
    design_spec = experiment.get_design_spec()
    arm_id_to_name = {str(arm.arm_id): arm.arm_name for arm in design_spec.arms}

    # Get strata field names from the first assignment
    strata_field_names = []
    if len(experiment.arm_assignments) > 0:
        strata_field_names = experiment.arm_assignments[0].strata_names()

    # Create CSV header
    header = ["participant_id", "arm_id", "arm_name", *strata_field_names]

    def generate_csv(batch_size=100):
        # Use csv.writer with StringIO to format a single row at a time
        output = io.StringIO()
        writer = csv.writer(output)

        try:
            # First write out our header
            writer.writerow(header)

            # Write out each participant row in batches
            for batch in batched(experiment.arm_assignments, batch_size):
                for participant in batch:
                    row = [
                        participant.participant_id,
                        str(participant.arm_id),
                        arm_id_to_name[str(participant.arm_id)],
                        *["" if v is None else v for v in participant.strata_values()],
                    ]
                    writer.writerow(row)

                # Return the batch as string for streaming to the user
                yield output.getvalue()
                # Clear the string buffer for the next batch
                output.seek(0)
                output.truncate(0)
        finally:
            output.close()

    return generate_csv


def get_experiment_assignments_as_csv_impl(experiment: Experiment):
    csv_generator = experiment_assignments_to_csv_generator(experiment)
    filename = f"experiment_{experiment.id}_assignments.csv"
    return StreamingResponse(
        csv_generator(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/experiments/{experiment_id}/assignments/csv",
    summary="Export experiment assignments as CSV file.",
)
def get_experiment_assignments_as_csv_sl(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[Session, Depends(xngin_db_session)],
    experiment_id: uuid.UUID,
) -> StreamingResponse:
    """Exports the assignments info with header row as CSV. BalanceCheck not included.

    csv header form: participant_id,arm_id,arm_name,strata_name1,strata_name2,...
    """
    experiment = get_experiment_or_raise(xngin_session, experiment_id, datasource.id)
    return get_experiment_assignments_as_csv_impl(experiment)
