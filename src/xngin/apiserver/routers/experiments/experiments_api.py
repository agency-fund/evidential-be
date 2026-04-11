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
        "description": "The request is invalid. This usually indicates your request doesn't match the required "
        "structure of the request, or is missing a field or request header.",
    },
    "403": {
        "model": dict,
        "description": "Requester does not have sufficient privileges to perform this operation or is not "
        f"authenticated.\n\nTip: Check that the API key passed in the `{constants.HEADER_API_KEY}` header has "
        f"access to the requested datasource or experiment.",
    },
    "404": {
        "model": dict,
        "description": "The requested resource was not found, or you do not have access to it.\n\nTip: Check that the "
        f"API key passed in the `{constants.HEADER_API_KEY}` header has access to "
        "the requested datasource or experiment.",
    },
}

router = APIRouter(
    lifespan=lifespan,
    prefix=constants.API_PREFIX_V1,
    responses=STANDARD_INTEGRATION_RESPONSES,
    strict_content_type=False,  # for backwards compatibility, do not require content-type: request headers.
)


@router.get("/experiments", summary="List experiments on the datasource.")
async def list_experiments(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> ListExperimentsResponse:
    return await list_organization_or_datasource_experiments_impl(
        xngin_session=xngin_session, datasource_id=datasource.id
    )


async def _stream_experiment_assignments_response(
    experiment: tables.Experiment, assignments: AsyncGenerator[dict[str, object]]
) -> AsyncIterator[bytes]:
    """Efficiently streams assignments/draws to the client."""
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
    summary="Get experiment metadata (design & assignment specs) for a single experiment.",
)
async def get_experiment(
    experiment: Annotated[tables.Experiment, Depends(experiment_response_dependency)],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> GetExperimentResponse:
    return await get_experiment_impl(xngin_session, experiment)


@router.get(
    "/experiments/{experiment_id}/assignments",
    summary="Fetch list of participant=>arm assignments for the given experiment id.",
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
    summary="Export experiment assignments as CSV.",
    description="""Returns a CSV stream with a header row.

    Output columns:
    - `participant_id`
    - `arm_id`
    - `arm_name`
    - `created_at`: UTC ISO 8601 timestamp in ISO8601 format
    - one column for each configured strata field
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
    summary="Get the assignment for a specific participant, excluding strata if any.",
    description="""
    For preassigned experiments, the participant's Assignment is returned if it exists.
    For all online experiments (except contextual bandits), returns the assignment if it exists,
    else generates an assignment.
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
                "Create an assignment if none exists. Does nothing for preassigned experiments. "
                "Override if you just want to check if an assignment exists."
            )
        ),
    ] = True,
    random_state: Annotated[int | None, Depends(random_seed_dependency)] = None,
    max_age: Annotated[
        int,
        Query(
            description=(
                "Controls the Cache-Control header max-age value returned with stable assignments "
                "(freq_preassigned, freq_online). Set to 0 to disable caching."
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
            exp_type in {ExperimentsType.MAB_ONLINE, ExperimentsType.CMAB_ONLINE}
            and assignment_response.assignment.outcome is not None
        )
        if is_stable_frequentist or is_bandit_with_outcome:
            response.headers["Cache-Control"] = f"private, max-age={max_age}"

    return assignment_response


@router.post(
    "/experiments/{experiment_id}/assignments/{participant_id}/assign_with_filters",
    description="""
    Get or create a frequentist online arm assignment for a participant that requires server-side
    filtering. If an assignment already exists, the properties in the
    OnlineAssignmentWithFiltersRequest are ignored and the existing assignment is returned.""",
)
async def get_assignment_filtered(
    experiment: Annotated[tables.Experiment, Depends(experiment_and_datasource_dependency)],
    participant_id: str,
    body: OnlineAssignmentWithFiltersRequest,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    create_if_none: Annotated[
        bool,
        Query(description="Create an assignment if none exists. Override to just check for existence."),
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
    description="""
    Get or create a CMAB arm assignment for a specific participant. This endpoint is used only for CMAB assignments.
    If there is a pre-existing assignment for a given participant ID, the context inputs in the
    CMABContextInputRequest can be None, and will be disregarded if they are not None.
    """,
)
async def get_assignment_cmab(
    experiment: Annotated[tables.Experiment, Depends(experiment_with_contexts_dependency)],
    participant_id: str,
    body: CMABContextInputRequest,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    create_if_none: Annotated[
        bool,
        Query(description=("Create an assignment if none exists. Override to just check for existence.")),
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
    description="""
    Update the bandit arm with corresponding outcome for a specific participant.
    Used only for bandit experiments.

    On the first call for a given participant, this endpoint will update the assigned arm with the provided outcome,
    and return the updated arm parameters.
    For a participant without an existing assignment, this endpoint will return a 422 error,
    as there is no arm to update.
    Please use the GET assignment endpoint to first create an assignment for the participant.
    For a participant with an existing assignment and previously recorded outcome, this endpoint
    will return a 422 error.
    Please use the GET assignment endpoint to check if an assignment with an outcome already exists
    for the participant.
    """,
    summary="""
    "Update the assigned arm with the provided outcome, and return the updated arm parameters.
    Only participants with an existing assignment and no previously recorded outcome can be updated with this endpoint.
    All other cases will return a 422 error, in which case, please use the GET assignment endpoint.
    """,
)
async def update_bandit_arm_with_participant_outcome(
    participant_id: str,
    body: Annotated[UpdateBanditArmOutcomeRequest, Body()],
    experiment: Annotated[tables.Experiment, Depends(experiment_dependency)],
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
