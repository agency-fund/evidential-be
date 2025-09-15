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
from xngin.apiserver.routers.common_api_types import (
    ArmBandit,
    ContextType,
    CreateCMABAssignmentRequest,
    ExperimentsType,
    GetExperimentAssignmentsResponse,
    GetExperimentResponse,
    GetParticipantAssignmentResponse,
    ListExperimentsResponse,
    UpdateBanditArmOutcomeRequest,
)
from xngin.apiserver.routers.experiments.dependencies import experiment_dependency
from xngin.apiserver.routers.experiments.experiments_common import (
    create_assignment_for_participant,
    get_assign_summary,
    get_existing_assignment_for_participant,
    get_experiment_assignments_as_csv_impl,
    get_experiment_assignments_impl,
    list_organization_or_datasource_experiments_impl,
    update_bandit_arm_with_outcome_impl,
)
from xngin.apiserver.settings import (
    Datasource,
)
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting router: {__name__} (prefix={router.prefix})")
    yield


router = APIRouter(
    lifespan=lifespan,
    prefix=constants.API_PREFIX_V1,
)


@router.get(
    "/experiments",
    summary="List experiments on the datasource.",
)
async def list_experiments_sl(
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
async def get_experiment_sl(
    experiment: Annotated[tables.Experiment, Depends(experiment_dependency)],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> GetExperimentResponse:
    converter = ExperimentStorageConverter(experiment)
    balance_check = converter.get_balance_check()
    assign_summary = await get_assign_summary(xngin_session, experiment.id, balance_check)
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
    description="""For preassigned experiments, the participant's Assignment is returned if it
    exists.  For all online experiments (except contextual bandits), returns the assignment if
    it exists, else generates an assignment""",
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
    assignment = await get_existing_assignment_for_participant(
        xngin_session=xngin_session,
        experiment_id=experiment.id,
        participant_id=participant_id,
        experiment_type=experiment.experiment_type,
    )
    if not assignment and create_if_none:
        assignment = await create_assignment_for_participant(
            xngin_session=xngin_session,
            experiment=experiment,
            participant_id=participant_id,
            random_state=random_state,
        )
    return GetParticipantAssignmentResponse(
        experiment_id=experiment.id,
        participant_id=participant_id,
        assignment=assignment,
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
    experiment: Annotated[tables.Experiment, Depends(experiment_dependency)],
    participant_id: str,
    body: CreateCMABAssignmentRequest,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    create_if_none: Annotated[
        bool,
        Query(
            description=(
                "Create an assignment if none exists. Override if you just want to check if an assignment exists."
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

    # Look up the participant's assignment if it exists
    assignment = await get_existing_assignment_for_participant(
        xngin_session=session,
        experiment_id=experiment.id,
        participant_id=participant_id,
        experiment_type=experiment.experiment_type,
    )

    if not assignment and create_if_none and experiment.stopped_assignments_at is None:
        context_inputs = body.context_inputs

        if not context_inputs:
            raise LateValidationError("Context inputs are required for creating CMAB assignments.")

        context_defns = await experiment.awaitable_attrs.contexts
        context_inputs = sorted(context_inputs, key=lambda x: x.context_id)
        context_defns = sorted(context_defns, key=lambda x: x.id)

        if len(context_inputs) != len(context_defns):
            raise LateValidationError(
                f"Expected {len(context_defns)} context inputs, but got {len(context_inputs)} in "
                f"CreateCMABAssignmentRequest."
            )

        for context_input, context_def in zip(
            context_inputs,
            context_defns,
            strict=True,
        ):
            if context_input.context_id != context_def.id:
                raise LateValidationError(
                    f"Context input for id {context_input.context_id} does not match expected context id "
                    f"{context_def.id}",
                )
            if context_def.value_type == ContextType.BINARY.value and context_input.context_value not in {0.0, 1.0}:
                raise LateValidationError(
                    f"Context value for id {context_input.context_id} must be binary (0 or 1).",
                )

        context_vals = [ctx.context_value for ctx in context_inputs]

        assignment = await create_assignment_for_participant(
            xngin_session=session,
            experiment=experiment,
            participant_id=participant_id,
            context_vals=context_vals,
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
