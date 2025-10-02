"""
This module defines the public API for clients to integrate with experiments.
(See admin_api.py for Evidential UI-facing endpoints.)
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import (
    APIRouter,
    Body,
    Depends,
    FastAPI,
    Query,
)
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import constants
from xngin.apiserver.dependencies import (
    datasource_dependency,
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
    UpdateBanditArmOutcomeRequest,
)
from xngin.apiserver.routers.experiments.dependencies import (
    experiment_dependency,
    experiment_response_dependency,
    experiment_with_assignments_dependency,
    experiment_with_contexts_dependency,
)
from xngin.apiserver.routers.experiments.experiments_common import (
    create_assignment_for_participant,
    get_existing_assignment_for_participant,
    get_experiment_assignments_as_csv_impl,
    get_experiment_assignments_impl,
    get_experiment_impl,
    get_or_create_assignment_for_participant,
    list_organization_or_datasource_experiments_impl,
    update_bandit_arm_with_outcome_impl,
)
from xngin.apiserver.settings import Datasource
from xngin.apiserver.sqla import tables


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting router: {__name__} (prefix={router.prefix})")
    yield


router = APIRouter(lifespan=lifespan, prefix=constants.API_PREFIX_V1)


@router.get("/experiments", summary="List experiments on the datasource.")
async def list_experiments(
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> ListExperimentsResponse:
    return await list_organization_or_datasource_experiments_impl(
        xngin_session=xngin_session, datasource_id=datasource.id
    )


@router.get(
    "/experiments/{experiment_id}",
    summary="Get experiment metadata (design & assignment specs) for a single experiment.",
)
async def get_experiment(
    experiment: Annotated[tables.Experiment, Depends(experiment_response_dependency)],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> GetExperimentResponse:
    return await get_experiment_impl(xngin_session, experiment)


# TODO: add a query param to include strata; default to false
@router.get(
    "/experiments/{experiment_id}/assignments",
    summary="Fetch list of participant=>arm assignments for the given experiment id.",
)
async def get_experiment_assignments(
    experiment: Annotated[tables.Experiment, Depends(experiment_with_assignments_dependency)],
) -> GetExperimentAssignmentsResponse:
    return get_experiment_assignments_impl(experiment)


@router.get(
    "/experiments/{experiment_id}/assignments/csv",
    summary="Export experiment assignments as CSV file.",
)
async def get_experiment_assignments_as_csv(
    experiment: Annotated[tables.Experiment, Depends(experiment_with_assignments_dependency)],
) -> StreamingResponse:
    """Exports the assignments info with header row as CSV. BalanceCheck not included.

    csv header form: participant_id,arm_id,arm_name,strata_name1,strata_name2,...
    """
    return await get_experiment_assignments_as_csv_impl(experiment)


@router.get(
    "/experiments/{experiment_id}/assignments/{participant_id}",
    summary="Get the assignment for a specific participant, excluding strata if any.",
    description="""
    For preassigned experiments, the participant's Assignment is returned if it exists.
    For all online experiments (except contextual bandits), returns the assignment if it exists,
    else generates an assignment.
    """,
)
async def get_assignment_for_participant_with_apikey(
    experiment: Annotated[tables.Experiment, Depends(experiment_dependency)],
    participant_id: str,
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
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
) -> GetParticipantAssignmentResponse:
    return await get_or_create_assignment_for_participant(
        xngin_session=xngin_session,
        experiment=experiment,
        participant_id=participant_id,
        create_if_none=create_if_none,
        random_state=random_state,
    )


@router.post(
    "/experiments/{experiment_id}/assignments/{participant_id}/assign_cmab",
    description="""
    Get or create a CMAB arm assignment for a specific participant. This endpoint is used only for CMAB assignments.
    If there is a pre-existing assignment for a given participant ID, the context inputs in the
    CreateCMABAssignmentRequest can be None, and will be disregarded if they are not None.
    """,
)
async def get_cmab_experiment_assignment_for_participant(
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
    description="""Update the bandit arm with corresponding outcome for a specific participant.
    Used only for bandit experiments.""",
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
