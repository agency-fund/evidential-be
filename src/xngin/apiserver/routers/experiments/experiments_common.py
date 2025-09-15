import asyncio
import csv
import io
import random
import secrets
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import batched

from fastapi import HTTPException, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import Select, Table, func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from xngin.apiserver import constants, flags
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.dwh.queries import get_participant_metrics
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.assignment_adapters import (
    RowProtocol,
    assign_treatments_with_balance,
    bulk_insert_arm_assignments,
    make_balance_check,
)
from xngin.apiserver.routers.common_api_types import (
    Arm,
    ArmAnalysis,
    ArmSize,
    Assignment,
    AssignSummary,
    BalanceCheck,
    BaseFrequentistDesignSpec,
    CreateExperimentRequest,
    CreateExperimentResponse,
    DesignSpecMetricRequest,
    FreqExperimentAnalysisResponse,
    GetExperimentAssignmentsResponse,
    ListExperimentsResponse,
    MetricAnalysis,
    Strata,
)
from xngin.apiserver.routers.common_enums import (
    ExperimentState,
    ExperimentsType,
    LikelihoodTypes,
    PriorTypes,
    StopAssignmentReason,
)
from xngin.apiserver.settings import DatasourceConfig, ParticipantsDef
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter
from xngin.apiserver.webhooks.webhook_types import ExperimentCreatedWebhookBody
from xngin.events.experiment_created import ExperimentCreatedEvent
from xngin.stats.analysis import analyze_experiment as analyze_experiment_impl
from xngin.stats.bandit_sampling import choose_arm, update_arm
from xngin.stats.stats_errors import StatsAnalysisError
from xngin.tq.task_payload_types import WEBHOOK_OUTBOUND_TASK_TYPE, WebhookOutboundTask


class ExperimentsAssignmentError(Exception):
    """Wrapper for errors raised by our xngin.apiserver.routers.experiments_common module."""


def random_choice[T](choices: Sequence[T], seed: int | None = None) -> T:
    """Choose a random value from choices."""
    if seed:
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        # use a predictable random
        r = random.Random(seed)
        return r.choice(choices)
    # Use very strong random by default
    return secrets.choice(choices)


async def create_experiment_impl(
    request: CreateExperimentRequest,
    datasource: tables.Datasource,
    xngin_session: AsyncSession,
    chosen_n: int | None,
    stratify_on_metrics: bool,
    random_state: int | None,
    validated_webhooks: list[tables.Webhook],
) -> CreateExperimentResponse:
    # Raise error for bandit experiments
    design_spec = request.design_spec
    match design_spec.experiment_type:
        case ExperimentsType.FREQ_ONLINE | ExperimentsType.FREQ_PREASSIGNED:
            ds_config = datasource.get_config()

            participants_cfg = ds_config.find_participants(design_spec.participant_type)
            if not isinstance(participants_cfg, ParticipantsDef):
                raise LateValidationError("Invalid ParticipantsConfig: Participants must be of type schema.")

            # Get participants and their schema info from the client dwh.
            # Only fetch the columns we might need for stratified random assignment.
            participants_unique_id_field = participants_cfg.get_unique_id_field()
            metric_names = [m.field_name for m in design_spec.metrics]
            strata_names = [s.field_name for s in design_spec.strata]
            stratum_cols = strata_names + metric_names if stratify_on_metrics else strata_names

            async with DwhSession(ds_config.dwh) as dwh:
                if chosen_n is not None:
                    result = await dwh.get_participants(
                        participants_cfg.table_name,
                        select_columns={*stratum_cols, participants_unique_id_field},
                        filters=design_spec.filters,
                        n=chosen_n,
                    )
                    sa_table, participants = result.sa_table, result.participants

                elif design_spec.experiment_type == ExperimentsType.FREQ_PREASSIGNED:
                    raise LateValidationError("Preassigned experiments must have a chosen_n.")
                else:
                    sa_table = await dwh.inspect_table(participants_cfg.table_name)

            match design_spec.experiment_type:
                case ExperimentsType.FREQ_PREASSIGNED:
                    if participants is None:
                        raise LateValidationError("Preassigned experiments must have participants data")
                    return await create_preassigned_experiment_impl(
                        request=request,
                        datasource_id=datasource.id,
                        organization_id=datasource.organization_id,
                        participant_unique_id_field=participants_unique_id_field,
                        dwh_sa_table=sa_table,
                        dwh_participants=participants,
                        random_state=random_state,
                        xngin_session=xngin_session,
                        stratify_on_metrics=stratify_on_metrics,
                        validated_webhooks=validated_webhooks,
                    )

                case ExperimentsType.FREQ_ONLINE:
                    return await create_freq_online_experiment_impl(
                        request=request,
                        datasource_id=datasource.id,
                        organization_id=datasource.organization_id,
                        xngin_session=xngin_session,
                        validated_webhooks=validated_webhooks,
                    )

        case ExperimentsType.MAB_ONLINE | ExperimentsType.CMAB_ONLINE:
            return await create_bandit_online_experiment_impl(
                xngin_session=xngin_session,
                organization_id=datasource.organization_id,
                validated_webhooks=validated_webhooks,
                request=request,
                datasource_id=datasource.id,
                chosen_n=chosen_n,
            )

        case _:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid experiment type: {design_spec.experiment_type}",
            )


async def create_preassigned_experiment_impl(
    request: CreateExperimentRequest,
    datasource_id: str,
    organization_id: str,
    participant_unique_id_field: str,
    dwh_sa_table: Table,
    dwh_participants: Sequence[RowProtocol],
    random_state: int | None,
    xngin_session: AsyncSession,
    stratify_on_metrics: bool,
    validated_webhooks: list[tables.Webhook],
) -> CreateExperimentResponse:
    design_spec = request.design_spec

    if not isinstance(
        design_spec,
        BaseFrequentistDesignSpec,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bandit experiments are not supported for preassigned assignments",
        )
    metric_names = [m.field_name for m in design_spec.metrics]
    strata_names = [s.field_name for s in design_spec.strata]
    stratum_cols = strata_names + metric_names if stratify_on_metrics else strata_names

    # Check for unique participant IDs after filtering
    seen_participant_ids = set()
    for participant in dwh_participants:
        participant_id = getattr(participant, participant_unique_id_field)
        if participant_id in seen_participant_ids:
            raise LateValidationError(f"Duplicate participant ID found after filtering: '{participant_id}'.")
        seen_participant_ids.add(participant_id)

    # Do the raw assignment first so we can store the balance check with the experiment.
    assignment_result = assign_treatments_with_balance(
        sa_table=dwh_sa_table,
        data=dwh_participants,
        stratum_cols=stratum_cols,
        id_col=participant_unique_id_field,
        n_arms=len(design_spec.arms),
        quantiles=4,
        random_state=random_state,
    )
    balance_check = make_balance_check(assignment_result.balance_result, design_spec.fstat_thresh)

    experiment_converter = ExperimentStorageConverter.init_from_components(
        datasource_id=datasource_id,
        organization_id=organization_id,
        experiment_type=design_spec.experiment_type,
        design_spec=design_spec,
        state=ExperimentState.ASSIGNED,
        stopped_assignments_at=datetime.now(UTC),
        stopped_assignments_reason=StopAssignmentReason.PREASSIGNED,
        balance_check=balance_check,
        power_analyses=request.power_analyses,
    )
    experiment = experiment_converter.get_experiment()
    # Associate webhooks with the experiment
    for webhook in validated_webhooks:
        experiment.webhooks.append(webhook)
    xngin_session.add(experiment)

    # Flush to get ids
    await xngin_session.flush()

    await bulk_insert_arm_assignments(
        xngin_session=xngin_session,
        experiment_id=experiment.id,
        arms=[Arm(arm_id=arm.id, arm_name=arm.name) for arm in experiment.arms],
        participant_type=experiment.participant_type,
        participant_id_col=participant_unique_id_field,
        data=dwh_participants,
        assignment_result=assignment_result,
    )

    await xngin_session.commit()

    assign_summary = await get_assign_summary(
        xngin_session, experiment.id, balance_check, ExperimentsType.FREQ_PREASSIGNED
    )
    webhook_ids = [webhook.id for webhook in validated_webhooks]
    return experiment_converter.get_create_experiment_response(assign_summary, webhook_ids)


async def create_freq_online_experiment_impl(
    request: CreateExperimentRequest,
    datasource_id: str,
    organization_id: str,
    xngin_session: AsyncSession,
    validated_webhooks: list[tables.Webhook],
) -> CreateExperimentResponse:
    """Create an online experiment and persist it to the database."""
    design_spec = request.design_spec

    # TODO: update to support bandit experiments
    if not isinstance(design_spec, BaseFrequentistDesignSpec):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bandit experiments are not supported for online assignments",
        )

    experiment_converter = ExperimentStorageConverter.init_from_components(
        datasource_id=datasource_id,
        organization_id=organization_id,
        experiment_type=ExperimentsType.FREQ_ONLINE,
        design_spec=design_spec,
    )
    experiment = experiment_converter.get_experiment()
    # Associate webhooks with the experiment
    for webhook in validated_webhooks:
        experiment.webhooks.append(webhook)
    xngin_session.add(experiment)

    await xngin_session.commit()
    # Online experiments start with no assignments.
    empty_assign_summary = AssignSummary(
        balance_check=None,
        sample_size=0,
        arm_sizes=[ArmSize(arm=arm.model_copy(), size=0) for arm in design_spec.arms],
    )
    webhook_ids = [webhook.id for webhook in validated_webhooks]
    return experiment_converter.get_create_experiment_response(empty_assign_summary, webhook_ids)


async def create_bandit_online_experiment_impl(
    xngin_session: AsyncSession,
    organization_id: str,
    validated_webhooks: list[tables.Webhook],
    request: CreateExperimentRequest,
    datasource_id: str,
    chosen_n: int | None = None,
) -> CreateExperimentResponse:
    """Create an online experiment and persist it to the database."""
    design_spec = request.design_spec

    experiment_converter = ExperimentStorageConverter.init_from_components(
        datasource_id=datasource_id,
        organization_id=organization_id,
        experiment_type=design_spec.experiment_type,
        design_spec=design_spec,
        n_trials=chosen_n if chosen_n is not None else 0,
    )
    experiment = experiment_converter.get_experiment()

    # Associate webhooks with the experiment
    for webhook in validated_webhooks:
        experiment.webhooks.append(webhook)
    xngin_session.add(experiment)

    await xngin_session.commit()
    # Online experiments start with no assignments.
    empty_assign_summary = AssignSummary(
        balance_check=None,
        sample_size=0,
        arm_sizes=[ArmSize(arm=arm.model_copy(), size=0) for arm in design_spec.arms],
    )
    webhook_ids = [webhook.id for webhook in validated_webhooks]
    return experiment_converter.get_create_experiment_response(empty_assign_summary, webhook_ids)


async def commit_experiment_impl(xngin_session: AsyncSession, experiment: tables.Experiment):
    if experiment.state == ExperimentState.COMMITTED:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED)
    if experiment.state != ExperimentState.ASSIGNED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid state: {experiment.state}",
        )

    experiment.state = ExperimentState.COMMITTED

    experiment_id = experiment.id
    datasource = await experiment.awaitable_attrs.datasource
    webhooks = await experiment.awaitable_attrs.webhooks

    event = tables.Event(
        organization_id=datasource.organization_id,
        type=ExperimentCreatedEvent.TYPE,
    ).set_data(ExperimentCreatedEvent(datasource_id=experiment.datasource_id, experiment_id=experiment_id))
    xngin_session.add(event)
    for webhook in webhooks:
        # If the organization has a webhook for experiment.created, enqueue a task for it.
        # In the future, this may be replaced by a standalone queuing service.
        if webhook.type == ExperimentCreatedEvent.TYPE:
            webhook_task = WebhookOutboundTask(
                organization_id=datasource.organization_id,
                url=webhook.url,
                body=ExperimentCreatedWebhookBody(
                    organization_id=datasource.organization_id,
                    datasource_id=datasource.id,
                    experiment_id=experiment_id,
                    experiment_url=f"{flags.XNGIN_PUBLIC_PROTOCOL}://{flags.XNGIN_PUBLIC_HOSTNAME}/v1/experiments/{experiment_id}",
                ).model_dump(),
                headers={constants.HEADER_WEBHOOK_TOKEN: webhook.auth_token} if webhook.auth_token else {},
            )
            task = tables.Task(
                task_type=WEBHOOK_OUTBOUND_TASK_TYPE,
                payload=webhook_task.model_dump(),
            )
            xngin_session.add(task)
    await xngin_session.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def abandon_experiment_impl(xngin_session: AsyncSession, experiment: tables.Experiment):
    if experiment.state == ExperimentState.ABANDONED:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED)
    if experiment.state not in {ExperimentState.DESIGNING, ExperimentState.ASSIGNED}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid state: {experiment.state}",
        )

    experiment.state = ExperimentState.ABANDONED
    await xngin_session.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def list_organization_or_datasource_experiments_impl(
    xngin_session: AsyncSession,
    *,
    organization_id: str | None = None,
    datasource_id: str | None = None,
) -> ListExperimentsResponse:
    """
    List experiments for a given organization or datasource.
    If both are provided, datasource_id takes precedence.
    Raises ValueError if neither is provided.
    """
    stmt = select(tables.Experiment).options(
        selectinload(tables.Experiment.arms),
        selectinload(tables.Experiment.contexts),
        selectinload(tables.Experiment.webhooks),
    )

    if datasource_id:
        stmt = stmt.where(tables.Experiment.datasource_id == datasource_id)
    elif organization_id:
        stmt = stmt.join(
            tables.Datasource,
            (tables.Experiment.datasource_id == tables.Datasource.id)
            & (tables.Datasource.organization_id == organization_id),
        )
    else:
        raise ValueError(
            "Either datasource_id or organization_id must be provided",
        )

    stmt = stmt.where(
        tables.Experiment.state.in_([
            ExperimentState.DESIGNING,
            ExperimentState.COMMITTED,
            ExperimentState.ASSIGNED,
        ])
    ).order_by(tables.Experiment.created_at.desc())

    experiments = await xngin_session.scalars(stmt)
    items = []
    for e in experiments:
        converter = ExperimentStorageConverter(e)
        balance_check = converter.get_balance_check()
        assign_summary = await get_assign_summary(
            xngin_session, e.id, balance_check, experiment_type=ExperimentsType(e.experiment_type)
        )
        webhook_ids = [webhook.id for webhook in e.webhooks]
        items.append(converter.get_experiment_config(assign_summary, webhook_ids))
    return ListExperimentsResponse(items=items)


def get_experiment_assignments_impl(
    experiment: tables.Experiment,
) -> GetExperimentAssignmentsResponse:
    # Map arm IDs to names
    arm_id_to_name = {arm.id: arm.name for arm in experiment.arms}
    # Convert ArmAssignment models to Assignment API types

    assignments: list[Assignment]

    match experiment.experiment_type:
        case ExperimentsType.FREQ_ONLINE | ExperimentsType.FREQ_PREASSIGNED:
            assignments = [
                Assignment(
                    participant_id=arm_assignment.participant_id,
                    arm_id=arm_assignment.arm_id,
                    arm_name=arm_id_to_name[arm_assignment.arm_id],
                    created_at=arm_assignment.created_at,
                    strata=[Strata.model_validate(s) for s in arm_assignment.strata],
                )
                for arm_assignment in experiment.arm_assignments
            ]
        case ExperimentsType.MAB_ONLINE | ExperimentsType.CMAB_ONLINE:
            assignments = [
                Assignment(
                    participant_id=draw.participant_id,
                    arm_id=draw.arm_id,
                    arm_name=arm_id_to_name[draw.arm_id],
                    created_at=draw.created_at,
                    observed_at=draw.observed_at,
                    outcome=draw.outcome,
                    context_values=draw.context_vals,
                )
                for draw in experiment.draws
            ]
        case _:
            raise LateValidationError(f"Invalid experiment type: {experiment.experiment_type}")

    return GetExperimentAssignmentsResponse(
        balance_check=ExperimentStorageConverter(experiment).get_balance_check(),
        experiment_id=experiment.id,
        sample_size=len(assignments),
        assignments=assignments,
    )


def experiment_assignments_to_csv_generator(experiment: tables.Experiment):
    """Generator function to yield CSV rows of experiment assignments as strings"""
    # Map arm IDs to names
    arm_id_to_name = {arm.id: arm.name for arm in experiment.arms}

    # Get strata field names from the first assignment
    strata_names = []
    if len(experiment.arm_assignments) > 0:
        strata_names = experiment.arm_assignments[0].strata_names()

    # Create CSV header
    header = ["participant_id", "arm_id", "arm_name", "created_at", *strata_names]

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
                        participant.arm_id,
                        arm_id_to_name[participant.arm_id],
                        participant.created_at,
                        *participant.strata_values(),
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


async def get_experiment_assignments_as_csv_impl(
    experiment: tables.Experiment,
) -> StreamingResponse:
    csv_generator = experiment_assignments_to_csv_generator(experiment)
    filename = f"experiment_{experiment.id}_assignments.csv"
    return StreamingResponse(
        csv_generator(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def get_existing_assignment_for_participant(
    xngin_session: AsyncSession,
    experiment_id: str,
    participant_id: str,
    experiment_type: str,
) -> Assignment | None:
    """Internal helper to look up an existing assignment for a participant.  Excludes strata.

    Returns: None if no assignment exists.
    """
    stmt: (
        Select[tuple[str, str, str, datetime]]
        | Select[
            tuple[
                str,
                str,
                str,
                datetime,
                list[float] | None,
                datetime | None,
                float | None | None,
            ]
        ]
    )

    match experiment_type:
        case ExperimentsType.FREQ_ONLINE | ExperimentsType.FREQ_PREASSIGNED:
            stmt = (
                select(
                    tables.ArmAssignment.participant_id,
                    tables.Arm.id.label("arm_id"),
                    tables.Arm.name.label("arm_name"),
                    tables.ArmAssignment.created_at,
                )
                .join(
                    tables.ArmAssignment,
                    (tables.ArmAssignment.arm_id == tables.Arm.id)
                    & (tables.ArmAssignment.experiment_id == tables.Arm.experiment_id),
                )
                .filter(
                    tables.Arm.experiment_id == experiment_id,
                    tables.ArmAssignment.participant_id == participant_id,
                )
            )
        case ExperimentsType.MAB_ONLINE | ExperimentsType.CMAB_ONLINE:
            stmt = (
                select(
                    tables.Draw.participant_id,
                    tables.Draw.arm_id,
                    tables.Arm.name.label("arm_name"),
                    tables.Draw.created_at,
                    tables.Draw.context_vals,
                    tables.Draw.observed_at,
                    tables.Draw.outcome,
                )
                .join(
                    tables.Arm,
                    (tables.Draw.arm_id == tables.Arm.id) & (tables.Draw.experiment_id == tables.Arm.experiment_id),
                )
                .filter(
                    tables.Arm.experiment_id == experiment_id,
                    tables.Draw.participant_id == participant_id,
                )
            )
        case _:
            raise ExperimentsAssignmentError(f"Invalid experiment type {experiment_type}")

    res = await xngin_session.execute(stmt)
    existing_assignment = res.one_or_none()
    # If the participant already has an assignment for this experiment, return it.
    if existing_assignment:
        return Assignment(
            participant_id=existing_assignment.participant_id,
            arm_id=existing_assignment.arm_id,
            arm_name=existing_assignment.arm_name,
            created_at=existing_assignment.created_at,
            strata=[],  # Strata are not included in this query
            observed_at=existing_assignment.observed_at if hasattr(existing_assignment, "observed_at") else None,
            outcome=existing_assignment.outcome if hasattr(existing_assignment, "outcome") else None,
            context_values=existing_assignment.context_vals if hasattr(existing_assignment, "context_vals") else None,
        )
    return None


async def create_assignment_for_participant(
    xngin_session: AsyncSession,
    experiment: tables.Experiment,
    participant_id: str,
    context_vals: list[float] | None = None,
    random_state: int | None = None,
) -> Assignment | None:
    """Helper to persist a new assignment for a participant. Returned value excludes strata.

    Has side effect of updating the experiment's stopped_at and stopped_reason if we discover we should stop assigning.
    """
    if experiment.stopped_assignments_at is not None:
        # Experiment is stopped, so no new assignments can be made.
        return None

    if experiment.state != ExperimentState.COMMITTED:
        raise ExperimentsAssignmentError(f"Invalid experiment state: {experiment.state}")

    if len(experiment.arms) == 0:
        raise ExperimentsAssignmentError("Experiment has no arms")

    experiment_type = ExperimentsType(experiment.experiment_type)
    if experiment_type == ExperimentsType.FREQ_PREASSIGNED:
        # Preassigned experiments are not allowed to have new assignments added.
        return None

    # TODO: Add support for Bayesian A/B experiments.
    if experiment_type == ExperimentsType.BAYESAB_ONLINE:
        raise ValueError("Bayesian A/B experiments are not supported for assignments")

    if experiment_type == ExperimentsType.CMAB_ONLINE:
        if not context_vals:
            raise ExperimentsAssignmentError(
                "Context values are required for contextual multi-armed bandit experiments"
            )
        if len(context_vals) != len(experiment.contexts):
            raise ExperimentsAssignmentError(
                f"Expected {len(experiment.contexts)} context values, got {len(context_vals)}"
            )

    # Don't allow new assignments for experiments that have ended.
    if experiment.end_date < datetime.now(UTC):
        experiment.stopped_assignments_at = datetime.now(UTC)
        experiment.stopped_assignments_reason = StopAssignmentReason.END_DATE
        await xngin_session.commit()
        return None

    if not random_state:
        random_state = 66  # Default seed for deterministic behavior in tests.
    # For online frequentist or Bayesian A/B experiments, create a new assignment
    # with simple random assignment.
    match experiment_type:
        case ExperimentsType.FREQ_ONLINE | ExperimentsType.BAYESAB_ONLINE:
            # Sort by arm name to ensure deterministic assignment with seed for tests.
            chosen_arm = random_choice(
                sorted(experiment.arms, key=lambda a: a.name),
                seed=random_state,
            )
        case ExperimentsType.MAB_ONLINE | ExperimentsType.CMAB_ONLINE:
            chosen_arm = choose_arm(experiment=experiment, context=context_vals, random_state=random_state)

    chosen_arm_id = chosen_arm.id

    # Create and save the new assignment. We use the insert() API because it allows us to read
    # the database-generated created_at value without needing to refresh the object in the SQLAlchemy cache.
    try:
        match experiment_type:
            case ExperimentsType.FREQ_ONLINE | ExperimentsType.FREQ_PREASSIGNED:
                result = (
                    await xngin_session.execute(
                        insert(tables.ArmAssignment)
                        .values(
                            experiment_id=experiment.id,
                            participant_id=participant_id,
                            participant_type=experiment.participant_type,
                            arm_id=chosen_arm_id,
                            strata=[],
                        )
                        .returning(tables.ArmAssignment.created_at)
                    )
                ).fetchone()
            case ExperimentsType.MAB_ONLINE | ExperimentsType.CMAB_ONLINE:
                result = (
                    await xngin_session.execute(
                        insert(tables.Draw)
                        .values(
                            experiment_id=experiment.id,
                            participant_id=participant_id,
                            participant_type=experiment.participant_type,
                            arm_id=chosen_arm_id,
                            context_vals=context_vals,
                        )
                        .returning(tables.Draw.created_at)
                    )
                ).fetchone()
            case _:
                raise ExperimentsAssignmentError(f"Invalid experiment type: {experiment_type}")

        if result is None:
            raise ExperimentsAssignmentError(f"Failed to create assignment for participant '{participant_id}'")
        created_at = result[0]
        await xngin_session.commit()
    except IntegrityError as e:
        await xngin_session.rollback()
        raise ExperimentsAssignmentError(
            f"Failed to assign participant '{participant_id}' to arm '{chosen_arm_id}': {e}"
        ) from e

    return Assignment(
        participant_id=participant_id,
        arm_id=chosen_arm_id,
        arm_name=chosen_arm.name,
        created_at=created_at,
        strata=[],
        context_values=context_vals,
    )


async def update_bandit_arm_with_outcome_impl(
    xngin_session: AsyncSession,
    experiment: tables.Experiment,
    participant_id: str,
    outcome: float,
) -> tables.Arm:
    """Update the Draw table with the outcome for a bandit experiment."""
    # Not supported for frequentist experiments
    design_spec = ExperimentStorageConverter(experiment).get_design_spec()

    if isinstance(design_spec, BaseFrequentistDesignSpec):
        raise LateValidationError(
            "Cannot dynamically update arms for frequentist experiments.",
        )
    # Look up the participant's assignment if it exists
    assignment = await get_existing_assignment_for_participant(
        xngin_session, experiment.id, participant_id, experiment.experiment_type
    )
    if not assignment:
        raise ExperimentsAssignmentError(
            f"Participant {participant_id} does not have an assignment for which to record an outcome.",
        )
    if assignment.outcome is not None:
        raise ExperimentsAssignmentError(
            f"Participant {participant_id} already has an outcome recorded.",
        )

    # TODO: Add support for Bayesian A/B experiments.
    if design_spec.experiment_type == ExperimentsType.BAYESAB_ONLINE:
        raise LateValidationError(
            f"Invalid experiment type for bandit outcome update: {design_spec.experiment_type.value}"
        )

    if design_spec.reward_type == LikelihoodTypes.BERNOULLI and outcome not in {
        0,
        1,
    }:
        raise LateValidationError(f"Invalid outcome for binary reward type: {outcome}. Must be 0 or 1.")

    try:
        draw_record = await xngin_session.scalar(
            select(tables.Draw).where(
                tables.Draw.participant_id == participant_id,
                tables.Draw.experiment_id == experiment.id,
            )
        )
        if draw_record is None:
            raise ExperimentsAssignmentError(
                f"No draw record found for participant '{participant_id}' for this experiment"
            )
        if draw_record.outcome is not None:
            raise ExperimentsAssignmentError(
                f"Participant '{participant_id}' already has an outcome recorded for experiment '{experiment.id}'"
            )

        draw_record.observed_at = datetime.now(UTC)
        draw_record.outcome = outcome

        arm_to_update = next(arm for arm in experiment.arms if arm.id == draw_record.arm_id)

        # Get all prior draws for this arm, sorted by creation date
        draws = await experiment.awaitable_attrs.draws
        relevant_draws = sorted(
            (d for d in draws if d.arm_id == draw_record.arm_id),
            key=lambda d: d.created_at,
            reverse=True,
        )
        outcomes = [outcome] + [d.outcome for d in relevant_draws]
        context_vals = (
            [draw_record.context_vals] + [d.context_vals for d in relevant_draws] if draw_record.context_vals else None
        )

        # Limit to most recent 100 draws
        # TODO: Make draw limiting configurable
        outcomes = outcomes[:100]
        context_vals = context_vals[:100] if context_vals else None

        updated_parameters = update_arm(
            experiment=experiment,
            arm_to_update=arm_to_update,
            outcomes=outcomes,
            context=context_vals,
        )

        # Update the draw record and arm with the new parameters
        match experiment.prior_type:
            case PriorTypes.BETA.value:
                draw_record.current_alpha, draw_record.current_beta = updated_parameters
                arm_to_update.alpha, arm_to_update.beta = updated_parameters

            case PriorTypes.NORMAL.value:
                draw_record.current_mu, draw_record.current_covariance = updated_parameters
                arm_to_update.mu, arm_to_update.covariance = updated_parameters
            case _:
                raise ExperimentsAssignmentError(f"Unsupported prior type: {experiment.prior_type}")

        xngin_session.add(draw_record)
        xngin_session.add(arm_to_update)
        await xngin_session.commit()

    except IntegrityError as e:
        await xngin_session.rollback()
        raise ExperimentsAssignmentError(
            f"Failed to update assignment for participant '{participant_id}' with outcome {outcome}: {e}"
        ) from e

    return arm_to_update


async def get_assign_summary(
    xngin_session: AsyncSession,
    experiment_id: str,
    balance_check: BalanceCheck | None = None,
    # Default to frequentist for backward compatibility
    experiment_type: ExperimentsType = ExperimentsType.FREQ_PREASSIGNED,
) -> AssignSummary:
    """Constructs an AssignSummary from the experiment's arms and arm_assignments."""
    result = None
    match experiment_type:
        case ExperimentsType.FREQ_ONLINE | ExperimentsType.FREQ_PREASSIGNED:
            result = await xngin_session.execute(
                select(tables.ArmAssignment.arm_id, tables.Arm.name, func.count())
                .join(tables.Arm)
                .where(tables.ArmAssignment.experiment_id == experiment_id)
                .group_by(tables.ArmAssignment.arm_id, tables.Arm.name)
            )
        case ExperimentsType.MAB_ONLINE | ExperimentsType.CMAB_ONLINE:
            result = await xngin_session.execute(
                select(tables.Draw.arm_id, tables.Arm.name, func.count())
                .join(tables.Arm)
                .where(tables.Draw.experiment_id == experiment_id)
                .group_by(tables.Draw.arm_id, tables.Arm.name)
            )
            balance_check = None
        case _:
            raise LateValidationError(f"Invalid experiment type: {experiment_type}")

    arm_sizes = [
        ArmSize(
            arm=Arm(arm_id=arm_id, arm_name=name),
            size=count,
        )
        for arm_id, name, count in result
    ]
    return AssignSummary(
        balance_check=balance_check,
        arm_sizes=arm_sizes,
        sample_size=sum(arm_size.size for arm_size in arm_sizes),
    )


async def analyze_experiment_freq_impl(
    dsconfig: DatasourceConfig,
    experiment: tables.Experiment,
    baseline_arm_id: str,
    metrics: list[DesignSpecMetricRequest],
) -> FreqExperimentAnalysisResponse:
    """Analyze a frequentist experiment. Assumes arms and arm_assignments are preloaded."""

    participants_cfg = dsconfig.find_participants(experiment.participant_type)
    unique_id_field = participants_cfg.get_unique_id_field()

    assignments = experiment.arm_assignments
    participant_ids = [assignment.participant_id for assignment in assignments]
    num_participants = len(participant_ids)
    if num_participants == 0:
        raise StatsAnalysisError("No participants found for experiment.")

    async with DwhSession(dsconfig.dwh) as dwh:
        sa_table = await dwh.inspect_table(participants_cfg.table_name)

        # Mark the start of the analysis as when we begin pulling outcomes.
        created_at = datetime.now(UTC)
        participant_outcomes = await asyncio.to_thread(
            get_participant_metrics,
            dwh.session,
            sa_table,
            metrics,
            unique_id_field,
            participant_ids,
        )

    # We want to notify the user if there are participants assigned to the experiment that are not
    # in the data warehouse. E.g. in an online experiment, perhaps a new user was assigned
    # before their info was synced to the dwh.
    num_missing_participants = num_participants - len(participant_outcomes)

    analyze_results = analyze_experiment_impl(assignments, participant_outcomes, baseline_arm_id)

    metric_analyses = []
    for metric in metrics:
        metric_name = metric.field_name
        arm_analyses = []
        for arm in experiment.arms:
            arm_result = analyze_results[metric_name][arm.id]
            arm_analyses.append(
                ArmAnalysis(
                    arm_id=arm.id,
                    arm_name=arm.name,
                    arm_description=arm.description,
                    is_baseline=arm_result.is_baseline,
                    estimate=arm_result.estimate,
                    p_value=arm_result.p_value,
                    t_stat=arm_result.t_stat,
                    std_error=arm_result.std_error,
                    num_missing_values=arm_result.num_missing_values,
                )
            )
        metric_analyses.append(MetricAnalysis(metric_name=metric_name, metric=metric, arm_analyses=arm_analyses))

    return FreqExperimentAnalysisResponse(
        experiment_id=experiment.id,
        metric_analyses=metric_analyses,
        num_participants=num_participants,
        num_missing_participants=num_missing_participants,
        created_at=created_at,
    )
