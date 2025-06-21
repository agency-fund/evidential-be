import csv
import io
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import batched

from fastapi import (
    HTTPException,
    Response,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy import Table, func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from xngin.apiserver import flags
from xngin.apiserver.models import tables
from xngin.apiserver.models.enums import ExperimentState, StopAssignmentReason
from xngin.apiserver.models.storage_format_converters import ExperimentStorageConverter
from xngin.apiserver.routers.experiments_api_types import (
    AssignSummary,
    CreateExperimentRequest,
    CreateExperimentResponse,
    GetExperimentAssignmentsResponse,
    ListExperimentsResponse,
)
from xngin.apiserver.routers.stateless_api_types import (
    Arm,
    ArmSize,
    Assignment,
    BalanceCheck,
    Strata,
)
from xngin.apiserver.utils import random_choice
from xngin.apiserver.webhooks.webhook_types import ExperimentCreatedWebhookBody
from xngin.events.experiment_created import ExperimentCreatedEvent
from xngin.stats.assignment import RowProtocol, assign_treatment
from xngin.tq.task_payload_types import WEBHOOK_OUTBOUND_TASK_TYPE, WebhookOutboundTask


class ExperimentsAssignmentError(Exception):
    """Wrapper for errors raised by our xngin.apiserver.routers.experiments_common module."""


async def create_experiment_impl(
    request: CreateExperimentRequest,
    datasource_id: str,
    participant_unique_id_field: str,
    dwh_sa_table: Table,
    dwh_participants: Sequence[RowProtocol] | None,
    random_state: int | None,
    xngin_session: AsyncSession,
    stratify_on_metrics: bool,
    webhook_ids: list[str],
) -> CreateExperimentResponse:
    # Get the organization_id from the database
    db_datasource = await xngin_session.get(tables.Datasource, datasource_id)
    if not db_datasource:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Datasource {datasource_id} not found in database",
        )
    organization_id = db_datasource.organization_id

    # Validate webhook IDs exist and belong to organization
    validated_webhooks = []
    if webhook_ids:
        webhooks = await xngin_session.scalars(
            select(tables.Webhook)
            .where(tables.Webhook.id.in_(webhook_ids))
            .where(tables.Webhook.organization_id == organization_id)
        )
        validated_webhooks = list(webhooks)
        found_webhook_ids = {w.id for w in validated_webhooks}
        missing_webhook_ids = set(webhook_ids) - found_webhook_ids
        if missing_webhook_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid webhook IDs: {sorted(missing_webhook_ids)}",
            )

    # First generate ids for the experiment and arms, which do_assignment needs.
    request.design_spec.experiment_id = tables.experiment_id_factory()
    for arm in request.design_spec.arms:
        arm.arm_id = tables.arm_id_factory()

    if request.design_spec.experiment_type == "preassigned":
        if dwh_participants is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Preassigned experiments must have participants data",
            )
        return await create_preassigned_experiment_impl(
            request=request,
            datasource_id=datasource_id,
            organization_id=organization_id,
            participant_unique_id_field=participant_unique_id_field,
            dwh_sa_table=dwh_sa_table,
            dwh_participants=dwh_participants,
            random_state=random_state,
            xngin_session=xngin_session,
            stratify_on_metrics=stratify_on_metrics,
            validated_webhooks=validated_webhooks,
        )
    if request.design_spec.experiment_type == "online":
        return await create_online_experiment_impl(
            request=request,
            datasource_id=datasource_id,
            organization_id=organization_id,
            xngin_session=xngin_session,
            validated_webhooks=validated_webhooks,
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid experiment type: {request.design_spec.experiment_type}",
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
    metric_names = [m.field_name for m in design_spec.metrics]
    strata_names = [s.field_name for s in design_spec.strata]
    stratum_cols = strata_names + metric_names if stratify_on_metrics else strata_names

    experiment_id = design_spec.experiment_id
    if experiment_id is None:
        # Should not actually happen, but just in case and for the type checker:
        raise ValueError("Must have an experiment_id before assigning treatments")

    # TODO: directly create ArmAssignments from the pd dataframe instead
    assignment_response = assign_treatment(
        sa_table=dwh_sa_table,
        data=dwh_participants,
        stratum_cols=stratum_cols,
        id_col=participant_unique_id_field,
        arms=design_spec.arms,
        experiment_id=experiment_id,
        fstat_thresh=design_spec.fstat_thresh,
        quantiles=4,  # TODO(qixotic): make this configurable
        stratum_id_name=None,
        random_state=random_state,
    )

    experiment_converter = ExperimentStorageConverter.init_from_components(
        datasource_id=datasource_id,
        organization_id=organization_id,
        experiment_type="preassigned",
        design_spec=design_spec,
        state=ExperimentState.ASSIGNED,
        stopped_assignments_at=datetime.now(UTC),
        stopped_assignments_reason=StopAssignmentReason.PREASSIGNED,
        balance_check=assignment_response.balance_check,
        power_analyses=request.power_analyses,
    )
    experiment = experiment_converter.get_experiment()
    # Associate webhooks with the experiment
    for webhook in validated_webhooks:
        experiment.webhooks.append(webhook)
    xngin_session.add(experiment)

    # Create assignment records
    for assignment in assignment_response.assignments:
        # TODO: bulk insert https://docs.sqlalchemy.org/en/20/orm/queryguide/dml.html#orm-queryguide-bulk-insert {"dml_strategy": "raw"}
        db_assignment = tables.ArmAssignment(
            experiment_id=experiment.id,
            participant_type=design_spec.participant_type,
            participant_id=assignment.participant_id,
            arm_id=str(assignment.arm_id),
            strata=[s.model_dump(mode="json") for s in assignment.strata]
            if assignment.strata
            else None,
        )
        xngin_session.add(db_assignment)

    await xngin_session.commit()

    assign_summary = await get_assign_summary(
        xngin_session, experiment.id, assignment_response.balance_check
    )
    webhook_ids = [webhook.id for webhook in validated_webhooks]
    return experiment_converter.get_create_experiment_response(
        assign_summary, webhook_ids
    )


async def create_online_experiment_impl(
    request: CreateExperimentRequest,
    datasource_id: str,
    organization_id: str,
    xngin_session: AsyncSession,
    validated_webhooks: list[tables.Webhook],
) -> CreateExperimentResponse:
    design_spec = request.design_spec
    experiment_converter = ExperimentStorageConverter.init_from_components(
        datasource_id=datasource_id,
        organization_id=organization_id,
        experiment_type="online",
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
    return experiment_converter.get_create_experiment_response(
        empty_assign_summary, webhook_ids
    )


async def commit_experiment_impl(
    xngin_session: AsyncSession, experiment: tables.Experiment
):
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
    ).set_data(
        ExperimentCreatedEvent(
            datasource_id=experiment.datasource_id, experiment_id=experiment_id
        )
    )
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
                headers={"Authorization": webhook.auth_token}
                if webhook.auth_token
                else {},
            )
            task = tables.Task(
                task_type=WEBHOOK_OUTBOUND_TASK_TYPE,
                payload=webhook_task.model_dump(),
            )
            xngin_session.add(task)
    await xngin_session.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def abandon_experiment_impl(
    xngin_session: AsyncSession, experiment: tables.Experiment
):
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


async def list_organization_experiments_impl(
    xngin_session: AsyncSession, organization_id: str
) -> ListExperimentsResponse:
    stmt = (
        select(tables.Experiment)
        .options(
            selectinload(tables.Experiment.arms),
            selectinload(tables.Experiment.webhooks),
        )  # async: ExperimentStorageConverter requires .arms
        .join(
            tables.Datasource,
            (tables.Experiment.datasource_id == tables.Datasource.id)
            & (tables.Datasource.organization_id == organization_id),
        )
        .where(
            tables.Experiment.state.in_([
                ExperimentState.DESIGNING,
                ExperimentState.COMMITTED,
                ExperimentState.ASSIGNED,
            ])
        )
        .order_by(tables.Experiment.start_date.desc())
    )
    experiments = await xngin_session.scalars(stmt)
    items = []
    for e in experiments:
        converter = ExperimentStorageConverter(e)
        balance_check = converter.get_balance_check()
        assign_summary = await get_assign_summary(xngin_session, e.id, balance_check)
        webhook_ids = [webhook.id for webhook in e.webhooks]
        items.append(converter.get_experiment_config(assign_summary, webhook_ids))
    return ListExperimentsResponse(items=items)


async def list_experiments_impl(
    xngin_session: AsyncSession, datasource_id: str
) -> ListExperimentsResponse:
    stmt = (
        select(tables.Experiment)
        .options(
            selectinload(tables.Experiment.arms),
            selectinload(tables.Experiment.webhooks),
        )
        .where(tables.Experiment.datasource_id == datasource_id)
        .where(
            tables.Experiment.state.in_([
                ExperimentState.DESIGNING,
                ExperimentState.COMMITTED,
                ExperimentState.ASSIGNED,
            ])
        )
        .order_by(tables.Experiment.created_at.desc())
    )
    experiments = await xngin_session.scalars(stmt)
    items = []
    for e in experiments:
        converter = ExperimentStorageConverter(e)
        balance_check = converter.get_balance_check()
        assign_summary = await get_assign_summary(xngin_session, e.id, balance_check)
        webhook_ids = [webhook.id for webhook in e.webhooks]
        items.append(converter.get_experiment_config(assign_summary, webhook_ids))
    return ListExperimentsResponse(items=items)


def get_experiment_assignments_impl(
    experiment: tables.Experiment,
) -> GetExperimentAssignmentsResponse:
    # Map arm IDs to names
    arm_id_to_name = {arm.id: arm.name for arm in experiment.arms}
    # Convert ArmAssignment models to Assignment API types
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
    xngin_session: AsyncSession, experiment_id: str, participant_id: str
) -> Assignment | None:
    """Internal helper to look up an existing assignment for a participant.  Excludes strata.

    Returns: None if no assignment exists.
    """
    stmt = (
        select(
            tables.ArmAssignment.participant_id,
            tables.ArmTable.id.label("arm_id"),
            tables.ArmTable.name.label("arm_name"),
            tables.ArmAssignment.created_at,
        )
        .join(
            tables.ArmAssignment,
            (tables.ArmAssignment.arm_id == tables.ArmTable.id)
            & (tables.ArmAssignment.experiment_id == tables.ArmTable.experiment_id),
        )
        .filter(
            tables.ArmTable.experiment_id == experiment_id,
            tables.ArmAssignment.participant_id == participant_id,
        )
    )
    res = await xngin_session.execute(stmt)
    existing_assignment = res.one_or_none()
    # If the participant already has an assignment for this experiment, return it.
    if existing_assignment:
        return Assignment(
            participant_id=existing_assignment.participant_id,
            arm_id=existing_assignment.arm_id,
            arm_name=existing_assignment.arm_name,
            created_at=existing_assignment.created_at,
            strata=[],
        )
    return None


async def create_assignment_for_participant(
    xngin_session: AsyncSession,
    experiment: tables.Experiment,
    participant_id: str,
    random_state: int | None,
) -> Assignment | None:
    """Helper to persist a new assignment for a participant. Returned value excludes strata.

    Has side effect of updating the experiment's stopped_at and stopped_reason if we discover we should stop assigning.
    """
    if experiment.stopped_assignments_at is not None:
        # Experiment is stopped, so no new assignments can be made.
        return None

    if experiment.state != ExperimentState.COMMITTED:
        raise ExperimentsAssignmentError(
            f"Invalid experiment state: {experiment.state}"
        )

    if len(experiment.arms) == 0:
        raise ExperimentsAssignmentError("Experiment has no arms")

    experiment_type = experiment.experiment_type
    if experiment_type == "preassigned":
        # Preassigned experiments are not allowed to have new assignmentsadded.
        return None
    if experiment_type != "online":
        raise ExperimentsAssignmentError(f"Invalid experiment type: {experiment_type}")

    # Don't allow new assignments for experiments that have ended.
    if experiment.end_date < datetime.now(UTC):
        experiment.stopped_assignments_at = datetime.now(UTC)
        experiment.stopped_assignments_reason = StopAssignmentReason.END_DATE
        await xngin_session.commit()
        return None

    # For online experiments, create a new assignment with simple random assignment.
    if random_state:
        # Sort by arm name to ensure deterministic assignment with seed for tests.
        chosen_arm = random_choice(
            sorted(experiment.arms, key=lambda a: a.name),
            seed=random_state,
        )
    else:
        chosen_arm = random_choice(experiment.arms)

    chosen_arm_id = chosen_arm.id

    # Create and save the new assignment. We use the insert() API because it allows us to read
    # the database-generated created_at value without needing to refresh the object in the SQLAlchemy cache.
    try:
        created_at = (
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
        ).fetchone()[0]
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
    )


async def get_assign_summary(
    xngin_session: AsyncSession,
    experiment_id: str,
    balance_check: BalanceCheck | None = None,
) -> AssignSummary:
    """Constructs an AssignSummary from the experiment's arms and arm_assignments."""
    result = await xngin_session.execute(
        select(tables.ArmAssignment.arm_id, tables.ArmTable.name, func.count())
        .join(tables.ArmTable)
        .where(tables.ArmAssignment.experiment_id == experiment_id)
        .group_by(tables.ArmAssignment.arm_id, tables.ArmTable.name)
    )
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
