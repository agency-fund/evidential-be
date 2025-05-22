import csv
import io
from collections.abc import Sequence
from itertools import batched
from fastapi import (
    HTTPException,
    Response,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy import Table, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from xngin.apiserver import flags
from xngin.apiserver.routers.stateless_api_types import (
    Arm,
    ArmSize,
    Assignment,
    PreassignedExperimentSpec,
    Strata,
)
from xngin.apiserver.models.enums import ExperimentState
from xngin.apiserver.models import tables
from xngin.apiserver.routers.experiments_api_types import (
    AssignSummary,
    CreateExperimentRequest,
    CreateExperimentResponse,
    ExperimentConfig,
    GetExperimentAssignmentsResponse,
    ListExperimentsResponse,
)
from xngin.apiserver.utils import random_choice
from xngin.apiserver.webhooks.webhook_types import ExperimentCreatedWebhookBody
from xngin.events.experiment_created import ExperimentCreatedEvent
from xngin.stats.assignment import RowProtocol, assign_treatment
from xngin.tq.task_payload_types import WEBHOOK_OUTBOUND_TASK_TYPE, WebhookOutboundTask
from xngin.tq.task_queue import Task


class ExperimentsAssignmentError(Exception):
    """Wrapper for errors raised by our xngin.apiserver.routers.experiments_common module."""


def create_experiment_impl(
    request: CreateExperimentRequest,
    datasource_id: str,
    participant_unique_id_field: str,
    dwh_sa_table: Table,
    dwh_participants: Sequence[RowProtocol] | None,
    random_state: int | None,
    xngin_session: Session,
    stratify_on_metrics: bool,
) -> CreateExperimentResponse:
    # Get the organization_id from the database
    db_datasource = xngin_session.get(tables.Datasource, datasource_id)
    if not db_datasource:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Datasource {datasource_id} not found in database",
        )
    organization_id = db_datasource.organization_id

    # First generate uuids for the experiment and arms, which do_assignment needs.
    request.design_spec.experiment_id = tables.experiment_id_factory()
    for arm in request.design_spec.arms:
        arm.arm_id = tables.arm_id_factory()

    if request.design_spec.experiment_type == "preassigned":
        if dwh_participants is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Preassigned experiments must have participants data",
            )
        return create_preassigned_experiment_impl(
            request=request,
            datasource_id=datasource_id,
            organization_id=organization_id,
            participant_unique_id_field=participant_unique_id_field,
            dwh_sa_table=dwh_sa_table,
            dwh_participants=dwh_participants,
            random_state=random_state,
            xngin_session=xngin_session,
            stratify_on_metrics=stratify_on_metrics,
        )
    if request.design_spec.experiment_type == "online":
        return create_online_experiment_impl(
            request=request,
            datasource_id=datasource_id,
            organization_id=organization_id,
            xngin_session=xngin_session,
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid experiment type: {request.design_spec.experiment_type}",
    )


def create_preassigned_experiment_impl(
    request: CreateExperimentRequest,
    datasource_id: str,
    organization_id: str,
    participant_unique_id_field: str,
    dwh_sa_table: Table,
    dwh_participants: Sequence[RowProtocol],
    random_state: int | None,
    xngin_session: Session,
    stratify_on_metrics: bool,
) -> CreateExperimentResponse:
    metric_names = [m.field_name for m in request.design_spec.metrics]
    strata_names = [s.field_name for s in request.design_spec.strata]
    stratum_cols = strata_names + metric_names if stratify_on_metrics else strata_names
    # TODO: directly create ArmAssignments from the pd dataframe instead
    assignment_response = assign_treatment(
        sa_table=dwh_sa_table,
        data=dwh_participants,
        stratum_cols=stratum_cols,
        id_col=participant_unique_id_field,
        arms=request.design_spec.arms,
        experiment_id=request.design_spec.experiment_id,
        fstat_thresh=request.design_spec.fstat_thresh,
        quantiles=4,  # TODO(qixotic): make this configurable
        stratum_id_name=None,
        random_state=random_state,
    )

    # Create experiment record
    balance_check = (
        assignment_response.balance_check.model_dump()
        if assignment_response.balance_check
        else None
    )
    experiment = tables.Experiment(
        id=request.design_spec.experiment_id,
        datasource_id=datasource_id,
        experiment_type="preassigned",
        participant_type=request.design_spec.participant_type,
        name=request.design_spec.experiment_name,
        description=request.design_spec.description,
        state=ExperimentState.ASSIGNED,
        start_date=request.design_spec.start_date,
        end_date=request.design_spec.end_date,
        design_spec=request.design_spec.model_dump(mode="json"),
        power_analyses=request.power_analyses.model_dump(mode="json")
        if request.power_analyses
        else None,
        balance_check=balance_check,
    )  # .set_design_spec(body.design_spec)
    xngin_session.add(experiment)

    # Create arm records
    for arm in request.design_spec.arms:
        db_arm = tables.ArmTable(
            id=arm.arm_id,
            name=arm.arm_name,
            description=arm.arm_description,
            experiment_id=experiment.id,
            organization_id=organization_id,
        )
        xngin_session.add(db_arm)

    # Create assignment records
    for assignment in assignment_response.assignments:
        # TODO: bulk insert https://docs.sqlalchemy.org/en/20/orm/queryguide/dml.html#orm-queryguide-bulk-insert {"dml_strategy": "raw"}
        db_assignment = tables.ArmAssignment(
            experiment_id=experiment.id,
            participant_type=request.design_spec.participant_type,
            participant_id=assignment.participant_id,
            arm_id=str(assignment.arm_id),
            strata=[s.model_dump(mode="json") for s in assignment.strata]
            if assignment.strata
            else None,
        )
        xngin_session.add(db_assignment)

    xngin_session.commit()

    return CreateExperimentResponse(
        datasource_id=datasource_id,
        state=experiment.state,
        design_spec=experiment.get_design_spec(),
        power_analyses=experiment.get_power_analyses(),
        assign_summary=get_assign_summary(xngin_session, experiment),
    )


def create_online_experiment_impl(
    request: CreateExperimentRequest,
    datasource_id: str,
    organization_id: str,
    xngin_session: Session,
) -> CreateExperimentResponse:
    experiment = tables.Experiment(
        id=request.design_spec.experiment_id,
        datasource_id=datasource_id,
        experiment_type="online",
        participant_type=request.design_spec.participant_type,
        name=request.design_spec.experiment_name,
        description=request.design_spec.description,
        # No assignments nor power check (for now), but we still want to allow a review.
        state=ExperimentState.ASSIGNED,
        start_date=request.design_spec.start_date,
        end_date=request.design_spec.end_date,
        design_spec=request.design_spec.model_dump(mode="json"),
        power_analyses=None,
    )
    xngin_session.add(experiment)
    # Create arm records
    for arm in request.design_spec.arms:
        db_arm = tables.ArmTable(
            id=arm.arm_id,
            name=arm.arm_name,
            description=arm.arm_description,
            experiment_id=experiment.id,
            organization_id=organization_id,
        )
        xngin_session.add(db_arm)
    xngin_session.commit()
    # Return the committed experiment config with no assignments.
    # Online experiments start with no assignments.
    empty_assign_summary = AssignSummary(
        balance_check=None,
        sample_size=0,
        arm_sizes=[
            ArmSize(arm=arm.model_copy(), size=0) for arm in request.design_spec.arms
        ],
    )
    return CreateExperimentResponse(
        datasource_id=datasource_id,
        state=experiment.state,
        design_spec=experiment.get_design_spec(),
        power_analyses=None,
        assign_summary=empty_assign_summary,
    )


def commit_experiment_impl(xngin_session: Session, experiment: tables.Experiment):
    if experiment.state == ExperimentState.COMMITTED:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED)
    if experiment.state != ExperimentState.ASSIGNED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid state: {experiment.state}",
        )

    experiment.state = ExperimentState.COMMITTED

    experiment_id = experiment.id
    event = tables.Event(
        organization=experiment.datasource.organization,
        type=ExperimentCreatedEvent.TYPE,
    ).set_data(
        ExperimentCreatedEvent(
            datasource_id=experiment.datasource_id, experiment_id=experiment_id
        )
    )
    xngin_session.add(event)

    for webhook in experiment.datasource.organization.webhooks:
        # If the organization has a webhook for experiment.created, enqueue a task for it.
        # In the future, this may be replaced by a standalone queuing service.
        if webhook.type == ExperimentCreatedEvent.TYPE:
            webhook_task = WebhookOutboundTask(
                organization_id=experiment.datasource.organization_id,
                url=webhook.url,
                body=ExperimentCreatedWebhookBody(
                    organization_id=experiment.datasource.organization_id,
                    datasource_id=experiment.datasource.id,
                    experiment_id=experiment_id,
                    experiment_url=f"{flags.XNGIN_PUBLIC_PROTOCOL}://{flags.XNGIN_PUBLIC_HOSTNAME}/v1/experiments/{experiment_id}",
                ).model_dump(),
                headers={"Authorization": webhook.auth_token}
                if webhook.auth_token
                else {},
            )
            task = Task(
                task_type=WEBHOOK_OUTBOUND_TASK_TYPE,
                payload=webhook_task.model_dump(),
            )
            xngin_session.add(task)
    xngin_session.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


def abandon_experiment_impl(xngin_session: Session, experiment: tables.Experiment):
    if experiment.state == ExperimentState.ABANDONED:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED)
    if experiment.state not in {ExperimentState.DESIGNING, ExperimentState.ASSIGNED}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid state: {experiment.state}",
        )

    experiment.state = ExperimentState.ABANDONED
    xngin_session.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


def list_experiments_impl(
    xngin_session: Session, datasource_id: str
) -> ListExperimentsResponse:
    stmt = (
        select(tables.Experiment)
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
    result = xngin_session.execute(stmt)
    experiments = result.scalars().all()
    return ListExperimentsResponse(
        items=[
            ExperimentConfig(
                datasource_id=e.datasource_id,
                state=e.state,
                design_spec=e.get_design_spec(),
                power_analyses=e.get_power_analyses(),
                assign_summary=get_assign_summary(xngin_session, e),
            )
            for e in experiments
        ]
    )


def get_experiment_assignments_impl(
    experiment: tables.Experiment,
) -> GetExperimentAssignmentsResponse:
    # Map arm IDs to names
    design_spec = PreassignedExperimentSpec.model_validate(experiment.design_spec)
    arm_id_to_name = {arm.arm_id: arm.arm_name for arm in design_spec.arms}
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
        balance_check=experiment.get_balance_check(),
        experiment_id=experiment.id,
        sample_size=len(assignments),
        assignments=assignments,
    )


def experiment_assignments_to_csv_generator(experiment: tables.Experiment):
    """Generator function to yield CSV rows of experiment assignments as strings"""
    # Map arm IDs to names
    design_spec = experiment.get_design_spec()
    arm_id_to_name = {arm.arm_id: arm.arm_name for arm in design_spec.arms}

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


def get_experiment_assignments_as_csv_impl(
    experiment: tables.Experiment,
) -> StreamingResponse:
    csv_generator = experiment_assignments_to_csv_generator(experiment)
    filename = f"experiment_{experiment.id}_assignments.csv"
    return StreamingResponse(
        csv_generator(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def get_existing_assignment_for_participant(
    xngin_session: Session, experiment_id: str, participant_id: str
) -> Assignment | None:
    """Internal helper to look up an existing assignment for a participant.  Excludes strata.

    Returns: None if no assignment exists.
    """
    existing_assignment = xngin_session.execute(
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
    ).one_or_none()
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


def create_assignment_for_participant(
    xngin_session: Session,
    experiment: tables.Experiment,
    participant_id: str,
    random_state: int | None,
) -> Assignment | None:
    """Internal helper to make and persist an assignment for a participant depending on the
    experiment type. Excludes strata."""

    if experiment.state != ExperimentState.COMMITTED:
        raise ExperimentsAssignmentError(
            f"Invalid experiment state: {experiment.state}"
        )
    available_arms = xngin_session.execute(
        select(tables.ArmTable.id, tables.ArmTable.name).where(
            tables.ArmTable.experiment_id == experiment.id
        )
    ).all()
    if len(available_arms) == 0:
        raise ExperimentsAssignmentError("Experiment has no arms")

    experiment_type = experiment.experiment_type
    if experiment_type == "preassigned":
        # Preassigned experiments are not allowed to have new ones added.
        return None
    if experiment_type != "online":
        raise ExperimentsAssignmentError(f"Invalid experiment type: {experiment_type}")

    # For online experiments, create a new assignment with simple random assignment.
    if random_state:
        # Sort by arm name to ensure deterministic assignment with seed for tests.
        chosen_arm = random_choice(
            sorted(available_arms, key=lambda a: a.name),
            seed=random_state,
        )
    else:
        chosen_arm = random_choice(available_arms)

    # Create and save the new assignment
    new_assignment = tables.ArmAssignment(
        experiment_id=experiment.id,
        participant_id=participant_id,
        participant_type=experiment.participant_type,
        arm_id=chosen_arm.id,
        strata=[],  # Online assignments don't have strata
    )
    try:
        xngin_session.add(new_assignment)
        xngin_session.commit()
    except IntegrityError as e:
        xngin_session.rollback()
        raise ExperimentsAssignmentError(
            f"Failed to assign participant '{participant_id}' to arm '{chosen_arm.id}': {e}"
        ) from e

    return Assignment(
        participant_id=participant_id,
        arm_id=chosen_arm.id,
        arm_name=chosen_arm.name,
        created_at=new_assignment.created_at,
        strata=[],
    )


def get_assign_summary(
    xngin_session: Session, experiment: tables.Experiment
) -> AssignSummary:
    """Constructs an AssignSummary from the experiment's arms and arm_assignments."""
    rows = xngin_session.execute(
        select(tables.ArmAssignment.arm_id, tables.ArmTable.name, func.count())
        .join(tables.ArmTable)
        .where(tables.ArmAssignment.experiment_id == experiment.id)
        .group_by(tables.ArmAssignment.arm_id, tables.ArmTable.name)
    ).all()
    arm_sizes = [
        ArmSize(
            arm=Arm(arm_id=arm_id, arm_name=name),
            size=count,
        )
        for arm_id, name, count in rows
    ]
    return AssignSummary(
        balance_check=experiment.get_balance_check(),
        arm_sizes=arm_sizes,
        sample_size=sum(arm_size.size for arm_size in arm_sizes),
    )
