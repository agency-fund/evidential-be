"""
This module defines the public API for clients to integrate with experiments.
(See admin_api.py for Evidential UI-facing endpoints.)
"""

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Annotated, Any, cast

import orjson
from annotated_types import Ge, Le
from fastapi import (
    APIRouter,
    Body,
    Depends,
    FastAPI,
    Query,
    Response,
)
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import constants
from xngin.apiserver.dependencies import (
    random_seed_dependency,
    xngin_db_session,
)
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.admin.admin_api import sort_contexts_by_id_or_raise
from xngin.apiserver.routers.common_api_types import (
    ArmBandit,
    AssignmentTypedDict,
    CMABContextInputRequest,
    ExperimentsType,
    GetExperimentAssignmentsResponse,
    GetExperimentResponse,
    GetParticipantAssignmentResponse,
    ListExperimentsResponse,
    OnlineAssignmentWithFiltersRequest,
    UpdateBanditArmOutcomeRequest,
)
from xngin.apiserver.routers.experiments import experiments_common_csv
from xngin.apiserver.routers.experiments.dependencies import (
    datasource_dependency,
    experiment_and_datasource_dependency,
    experiment_dependency,
    experiment_response_dependency,
    experiment_with_contexts_dependency,
)
from xngin.apiserver.routers.experiments.experiments_common import (
    create_assignment_for_participant,
    get_existing_assignment_for_participant,
    get_experiment_impl,
    get_or_create_assignment_for_participant,
    list_organization_or_datasource_experiments_impl,
    update_bandit_arm_with_outcome_impl,
)
from xngin.apiserver.routers.experiments.experiments_common_csv import (
    CsvStreamingResponse,
    get_experiment_assignments_impl,
)
from xngin.apiserver.settings import Datasource
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter

JSON_STREAM_ROWS_PER_YIELD = 1_000


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting router: {__name__} (prefix={router.prefix})")
    yield


STANDARD_INTEGRATION_RESPONSES: dict[str | int, dict[str, Any]] = {
    "400": {
        "model": dict,
        "description": "The request is invalid. Check that the request body matches the required structure and "
        "that all required fields and headers are present.",
    },
    "403": {
        "model": dict,
        "description": "You did not provide a valid API key, or your API key does not have permission for this "
        f"operation.\n\nTip: Check that the API key in the `{constants.HEADER_API_KEY}` header has "
        f"access to the data source or experiment.",
    },
    "404": {
        "model": dict,
        "description": "The resource was not found, or you do not have access to it.\n\nTip: Check that the "
        f"API key in the `{constants.HEADER_API_KEY}` header has access to "
        "the data source or experiment.",
    },
}

router = APIRouter(
    lifespan=lifespan,
    prefix=constants.API_PREFIX_V1,
    responses=STANDARD_INTEGRATION_RESPONSES,
    strict_content_type=False,  # for backwards compatibility, do not require content-type: request headers.
)


@router.get("/experiments", summary="List experiments on a data source.")
async def list_experiments(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> ListExperimentsResponse:
    return await list_organization_or_datasource_experiments_impl(
        xngin_session=xngin_session, datasource_id=datasource.id
    )


async def _stream_experiment_assignments_response(
    experiment: tables.Experiment, assignments: AsyncGenerator[AssignmentTypedDict]
) -> AsyncIterator[bytes]:
    """Efficiently streams Assignments to the client."""
    balance_check = ExperimentStorageConverter(experiment).get_balance_check()
    yield (
        b'{"balance_check":'
        + orjson.dumps(None if balance_check is None else balance_check.model_dump(mode="json"))
        + b',"experiment_id":'
        + orjson.dumps(experiment.id)
        + b',"assignments":['
    )

    sample_size = 0
    buffered = 0
    needs_comma = False
    batch: list[bytes] = []
    async for assignment in assignments:
        if needs_comma:
            batch.append(b",")
        batch.append(orjson.dumps(assignment))
        needs_comma = True
        sample_size += 1
        buffered += 1
        if buffered == JSON_STREAM_ROWS_PER_YIELD:
            yield b"".join(batch)
            batch.clear()
            buffered = 0

    yield b"".join(batch) + b'],"sample_size":' + str(sample_size).encode() + b"}"


@router.get(
    "/experiments/{experiment_id}",
    summary="Get an experiment's design.",
)
async def get_experiment(
    experiment: Annotated[tables.Experiment, Depends(experiment_response_dependency)],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> GetExperimentResponse:
    return await get_experiment_impl(xngin_session, experiment)


@router.get(
    "/experiments/{experiment_id}/assignments",
    summary="List an experiment's assignments.",
)
async def get_experiment_assignments(
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
    experiment: Annotated[tables.Experiment, Depends(experiment_dependency)],
) -> GetExperimentAssignmentsResponse:
    assignments = get_experiment_assignments_impl(xngin_session, experiment)
    return cast(
        GetExperimentAssignmentsResponse,
        cast(
            object,
            StreamingResponse(
                _stream_experiment_assignments_response(experiment, assignments),
                media_type="application/json",
            ),
        ),
    )


@router.get(
    "/experiments/{experiment_id}/assignments/csv",
    summary="Export an experiment's assignments as CSV.",
    description="""Returns a CSV stream with a header row.

    Output columns:
    - `participant_id`
    - `cluster_key`, if the experiment design defines a cluster key
    - `arm_id`
    - `arm_name`
    - `created_at`: the time the assignment was created, as a UTC ISO 8601 timestamp
    - one column for each strata field defined on the experiment
    """,
    response_class=CsvStreamingResponse,
)
async def get_experiment_assignments_as_csv(
    experiment: Annotated[tables.Experiment, Depends(experiment_and_datasource_dependency)],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> CsvStreamingResponse:
    return await experiments_common_csv.get_experiment_assignments_as_csv_impl(xngin_session, experiment)


@router.get(
    "/experiments/{experiment_id}/assignments/{participant_id}",
    summary="Get or create a participant's assignment.",
    description="""
    Gets or creates an arm assignment for a participant. The behavior depends on the experiment type:

    - Preassigned A/B experiments: returns the assignment if one exists. `create_if_none` is ignored.
    - Online A/B and Multi-armed Bandit experiments: returns the existing assignment, or creates one
      when `create_if_none` is true.
    - Contextual Multi-armed Bandit (CMAB) experiments: returns the assignment if one exists. To create
      a new assignment, use the `assign_cmab` endpoint, which accepts the required context values.
    """,
)
async def get_assignment(
    experiment: Annotated[tables.Experiment, Depends(experiment_dependency)],
    participant_id: str,
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
    response: Response,
    create_if_none: Annotated[
        bool,
        Query(
            description=(
                "Create an assignment if none exists. Set to false to check for an existing assignment "
                "without creating one. Ignored for preassigned experiments."
            )
        ),
    ] = True,
    random_state: Annotated[int | None, Depends(random_seed_dependency)] = None,
    max_age: Annotated[
        int,
        Query(
            description=(
                "Sets the Cache-Control max-age, in seconds, for Preassigned A/B and Online A/B experiments. "
                "Set to 0 to disable caching."
            )
        ),
        Ge(0),
        Le(86400),
    ] = int(timedelta(hours=1).total_seconds()),
) -> GetParticipantAssignmentResponse:
    assignment_response = await get_or_create_assignment_for_participant(
        xngin_session=xngin_session,
        experiment=experiment,
        participant_id=participant_id,
        create_if_none=create_if_none,
        properties=None,
        random_state=random_state,
    )

    # Only instruct clients to cache responses we know are immutable after creation.
    if max_age > 0 and assignment_response.assignment:
        exp_type = ExperimentsType(experiment.experiment_type)
        is_stable_frequentist = exp_type in {
            ExperimentsType.FREQ_PREASSIGNED,
            ExperimentsType.FREQ_ONLINE,
        }
        is_bandit_with_outcome = (
            exp_type in {ExperimentsType.MAB_ONLINE, ExperimentsType.MAB_ONLINE_DWH, ExperimentsType.CMAB_ONLINE}
            and assignment_response.assignment.outcome is not None
        )
        if is_stable_frequentist or is_bandit_with_outcome:
            response.headers["Cache-Control"] = f"private, max-age={max_age}"

    return assignment_response


@router.post(
    "/experiments/{experiment_id}/assignments/{participant_id}/assign_with_filters",
    summary="Get or create an Online A/B assignment with server-side filtering.",
    description="""
    Gets or creates an arm assignment for a participant on experiments using server-side filtering. If an assignment
    already exists, that assignment is returned and the properties in the request body are ignored.

    If there are no filters on the experiment, use the get_assignment endpoint.""",
)
async def get_assignment_filtered(
    experiment: Annotated[tables.Experiment, Depends(experiment_and_datasource_dependency)],
    participant_id: str,
    body: OnlineAssignmentWithFiltersRequest,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    create_if_none: Annotated[
        bool,
        Query(
            description=(
                "Create an assignment if none exists. Set to false to check for an existing assignment "
                "without creating one."
            )
        ),
    ] = True,
    random_state: Annotated[
        int | None,
        Query(description="Specify a random seed for reproducibility.", include_in_schema=False),
    ] = None,
) -> GetParticipantAssignmentResponse:
    """Get or create the frequentist online arm assignment for a participant in an experiment."""

    if experiment.experiment_type != ExperimentsType.FREQ_ONLINE.value:
        raise LateValidationError(
            f"Experiment {experiment.id} is a {experiment.experiment_type} experiment, and not a "
            f"{ExperimentsType.FREQ_ONLINE.value} experiment. Please use the corresponding GET endpoint to "
            f"create assignments."
        )

    return await get_or_create_assignment_for_participant(
        xngin_session=session,
        experiment=experiment,
        participant_id=participant_id,
        create_if_none=create_if_none,
        properties=body.properties,
        random_state=random_state,
    )


@router.post(
    "/experiments/{experiment_id}/assignments/{participant_id}/assign_cmab",
    summary="Get or create a CMAB assignment for a participant.",
    description="""
    Gets or creates an arm assignment for a participant. Used only for CMAB experiments. If an
    assignment already exists, `context_inputs` in the request body may be null. If you provide it
    anyway, it is ignored.
    """,
)
async def get_assignment_cmab(
    experiment: Annotated[tables.Experiment, Depends(experiment_with_contexts_dependency)],
    participant_id: str,
    body: CMABContextInputRequest,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    create_if_none: Annotated[
        bool,
        Query(
            description=(
                "Create an assignment if none exists. Set to false to check for an existing assignment "
                "without creating one."
            )
        ),
    ] = True,
    random_state: Annotated[
        int | None,
        Query(
            description="Specify a random seed for reproducibility.",
            include_in_schema=False,
        ),
    ] = None,
) -> GetParticipantAssignmentResponse:
    """Get or create the CMAB arm assignment for a specific participant in an experiment."""

    if experiment.experiment_type != ExperimentsType.CMAB_ONLINE.value:
        raise LateValidationError(
            f"Experiment {experiment.id} is a {experiment.experiment_type} experiment, and not a "
            f"{ExperimentsType.CMAB_ONLINE.value} experiment. Please use the corresponding GET endpoint to "
            f"create assignments."
        )

    assignment = await get_existing_assignment_for_participant(
        xngin_session=session,
        experiment_id=experiment.id,
        participant_id=participant_id,
        experiment_type=experiment.experiment_type,
    )

    if not assignment and create_if_none and experiment.stopped_assignments_at is None:
        context_inputs = body.context_inputs
        if context_inputs is None:
            raise LateValidationError("context_inputs must be provided when creating a new CMAB assignment.")
        context_defns = experiment.contexts
        sorted_context_inputs = sort_contexts_by_id_or_raise(context_defns, context_inputs)
        sorted_context_vals = [ctx.context_value for ctx in sorted_context_inputs]

        assignment = await create_assignment_for_participant(
            xngin_session=session,
            experiment=experiment,
            participant_id=participant_id,
            sorted_context_vals=sorted_context_vals,
            random_state=random_state,
        )

    return GetParticipantAssignmentResponse(
        experiment_id=experiment.id,
        participant_id=participant_id,
        assignment=assignment,
    )


@router.post(
    "/experiments/{experiment_id}/assignments/{participant_id}/outcome",
    summary="Record a bandit arm outcome for a participant.",
    description="""
    Records the outcome for a participant's assigned arm and returns the updated arm parameters.
    Used only for bandit experiments.

    Prerequisites:
    - The participant must already have an assignment. Create one with the GET assignment endpoint
      first.
    - The participant must not already have a recorded outcome. Use the GET assignment endpoint to
      check whether an outcome is already recorded.

    The endpoint returns a 422 error if a prerequisite is not met.
    """,
)
async def update_bandit_arm_with_participant_outcome(
    participant_id: str,
    body: Annotated[UpdateBanditArmOutcomeRequest, Body()],
    experiment: Annotated[tables.Experiment, Depends(experiment_and_datasource_dependency)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> ArmBandit:
    # Update the arm with the outcome
    if experiment.experiment_type == ExperimentsType.CMAB_ONLINE.value:
        await experiment.awaitable_attrs.contexts

    updated_arm = await update_bandit_arm_with_outcome_impl(
        xngin_session=session,
        experiment=experiment,
        participant_id=participant_id,
        outcome=body.outcome,
    )

    return ArmBandit(
        arm_id=updated_arm.id,
        arm_name=updated_arm.name,
        arm_description=updated_arm.description,
        alpha_init=updated_arm.alpha_init,
        beta_init=updated_arm.beta_init,
        alpha=updated_arm.alpha,
        beta=updated_arm.beta,
        mu_init=updated_arm.mu_init,
        sigma_init=updated_arm.sigma_init,
        mu=updated_arm.mu,
        covariance=updated_arm.covariance,
    )
