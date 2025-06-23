"""
This module defines the public API for clients to integrate with experiments.
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    status,
)
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import constants
from xngin.apiserver.dependencies import (
    datasource_dependency,
    experiment_dependency,
    gsheet_cache,
    random_seed_dependency,
    xngin_db_session,
)
from xngin.apiserver.dwh.queries import query_for_participants
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.gsheet_cache import GSheetCache
from xngin.apiserver.models import tables
from xngin.apiserver.models.storage_format_converters import ExperimentStorageConverter
from xngin.apiserver.routers.common_api_types import (
    CreateExperimentRequest,
    CreateExperimentResponse,
    GetExperimentAssignmentsResponse,
    GetExperimentResponse,
    GetParticipantAssignmentResponse,
    ListExperimentsResponse,
)
from xngin.apiserver.routers.experiments.experiments_common import (
    abandon_experiment_impl,
    commit_experiment_impl,
    create_assignment_for_participant,
    create_experiment_impl,
    get_assign_summary,
    get_existing_assignment_for_participant,
    get_experiment_assignments_as_csv_impl,
    get_experiment_assignments_impl,
    list_experiments_impl,
)
from xngin.apiserver.routers.stateless.stateless_api import (
    CommonQueryParams,
    get_participants_config_and_schema,
)
from xngin.apiserver.settings import (
    Datasource,
    infer_table,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting router: {__name__} (prefix={router.prefix})")
    yield


router = APIRouter(
    lifespan=lifespan,
    prefix=constants.API_PREFIX_V1,
)


#  TODO? Remove mutating endpoints (except assignment) from public-facing integration API
@router.post(
    "/experiments/with-assignment",
    summary="Create an experiment and save its assignments to the database.",
    description=(
        "The newly created experiment will be in the ASSIGNED state. "
        "To move them to the COMMITTED state, call the /experiments/<id>/commit API."
    ),
    include_in_schema=False,
)
async def create_experiment_with_assignment_sl(
    body: CreateExperimentRequest,
    chosen_n: Annotated[
        int, Query(..., description="Number of participants to assign.")
    ],
    gsheets: Annotated[GSheetCache, Depends(gsheet_cache)],
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
    random_state: Annotated[int | None, Depends(random_seed_dependency)],
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
) -> CreateExperimentResponse:
    """Creates an experiment and saves its assignments to the database."""
    if body.design_spec.ids_are_present():
        raise LateValidationError("Invalid DesignSpec: UUIDs must not be set.")

    ds_config = datasource.config
    commons = CommonQueryParams(
        participant_type=body.design_spec.participant_type, refresh=refresh
    )
    participants_cfg, schema = await get_participants_config_and_schema(
        commons, ds_config, gsheets
    )

    # Get participants and their schema info from the client dwh
    with ds_config.dbsession() as dwh_session:
        sa_table = infer_table(
            dwh_session.get_bind(),
            participants_cfg.table_name,
            ds_config.supports_reflection(),
        )
        participants = query_for_participants(
            dwh_session, sa_table, body.design_spec.filters, chosen_n
        )

    # Persist the experiment and assignments in the xngin database
    return await create_experiment_impl(
        request=body,
        datasource_id=datasource.id,
        participant_unique_id_field=schema.get_unique_id_field(),
        dwh_sa_table=sa_table,
        dwh_participants=participants,
        random_state=random_state,
        xngin_session=xngin_session,
        stratify_on_metrics=True,
    )


async def get_experiment_or_raise(
    xngin_session: AsyncSession, experiment_id: str, datasource_id: str
):
    result = await xngin_session.scalars(
        select(tables.Experiment).where(
            tables.Experiment.id == experiment_id,
            tables.Experiment.datasource_id == datasource_id,
        )
    )
    experiment = result.one_or_none()
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found"
        )
    return experiment


@router.post(
    "/experiments/{experiment_id}/commit",
    summary="Marks any ASSIGNED experiment as COMMITTED.",
    status_code=status.HTTP_204_NO_CONTENT,
    include_in_schema=False,
)
async def commit_experiment_sl(
    experiment: Annotated[tables.Experiment, Depends(experiment_dependency)],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
):
    return await commit_experiment_impl(xngin_session, experiment)


@router.post(
    "/experiments/{experiment_id}/abandon",
    summary="Marks any DESIGNING or ASSIGNED experiment as ABANDONED.",
    status_code=status.HTTP_204_NO_CONTENT,
    include_in_schema=False,
)
async def abandon_experiment_sl(
    experiment: Annotated[tables.Experiment, Depends(experiment_dependency)],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
):
    return await abandon_experiment_impl(xngin_session, experiment)


@router.get(
    "/experiments",
    summary="List experiments on the datasource.",
)
async def list_experiments_sl(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> ListExperimentsResponse:
    return await list_experiments_impl(xngin_session, datasource.id)


@router.get(
    "/experiments/{experiment_id}",
    summary="Get experiment metadata (design & assignment specs) for a single experiment.",
)
async def get_experiment_sl(
    experiment: Annotated[tables.Experiment, Depends(experiment_dependency)],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> GetExperimentResponse:
    converter = ExperimentStorageConverter(experiment)
    balance_check = converter.get_balance_check()
    assign_summary = await get_assign_summary(
        xngin_session, experiment.id, balance_check
    )
    return converter.get_experiment_response(assign_summary)


# TODO: add a query param to include strata; default to false
@router.get(
    "/experiments/{experiment_id}/assignments",
    summary="Fetch list of participant=>arm assignments for the given experiment id.",
)
async def get_experiment_assignments_sl(
    experiment: Annotated[tables.Experiment, Depends(experiment_dependency)],
) -> GetExperimentAssignmentsResponse:
    return get_experiment_assignments_impl(experiment)


@router.get(
    "/experiments/{experiment_id}/assignments/csv",
    summary="Export experiment assignments as CSV file.",
)
async def get_experiment_assignments_as_csv_sl(
    experiment: Annotated[tables.Experiment, Depends(experiment_dependency)],
) -> StreamingResponse:
    """Exports the assignments info with header row as CSV. BalanceCheck not included.

    csv header form: participant_id,arm_id,arm_name,strata_name1,strata_name2,...
    """
    return await get_experiment_assignments_as_csv_impl(experiment)


@router.get(
    "/experiments/{experiment_id}/assignments/{participant_id}",
    summary="Get the assignment for a specific participant, excluding strata if any.",
    description="""For 'preassigned' experiments, the participant's Assignment is returned if it
    exists.  For 'online', returns the assignment if it exists, else generates an assignment""",
)
async def get_assignment_for_participant_with_apikey(
    experiment: Annotated[tables.Experiment, Depends(experiment_dependency)],
    participant_id: str,
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
    create_if_none: Annotated[
        bool,
        Query(
            description="Create an assignment if none exists. Does nothing for preassigned experiments. Override if you just want to check if an assignment exists."
        ),
    ] = True,
    random_state: Annotated[int | None, Depends(random_seed_dependency)] = None,
) -> GetParticipantAssignmentResponse:
    assignment = await get_existing_assignment_for_participant(
        xngin_session, experiment.id, participant_id
    )
    if not assignment and create_if_none:
        assignment = await create_assignment_for_participant(
            xngin_session, experiment, participant_id, random_state
        )

    return GetParticipantAssignmentResponse(
        experiment_id=experiment.id,
        participant_id=participant_id,
        assignment=assignment,
    )
