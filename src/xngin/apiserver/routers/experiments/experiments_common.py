import asyncio
import enum
import io
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast

import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from pandas import DataFrame
from sqlalchemy import Select, Table, func, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from xngin.apiserver import constants, flags
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.dwh.participant_metrics_queries import get_participant_metrics
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
    BanditExperimentAnalysisResponse,
    BaseFrequentistDesignSpec,
    CreateExperimentRequest,
    CreateExperimentResponse,
    DesignSpecMetricRequest,
    FreqExperimentAnalysisResponse,
    GetExperimentAssignmentsResponse,
    GetExperimentResponse,
    GetParticipantAssignmentResponse,
    ListExperimentsResponse,
    MetricAnalysis,
    ParticipantProperty,
    Strata,
)
from xngin.apiserver.routers.common_enums import (
    DataType,
    ExperimentState,
    ExperimentsType,
    LikelihoodTypes,
    StopAssignmentReason,
    UpdateTypeBeta,
    UpdateTypeNormal,
)
from xngin.apiserver.routers.experiments.property_filters import passes_filters, validate_filter_value
from xngin.apiserver.settings import DatasourceConfig
from xngin.apiserver.sql.queries import select_as_csv
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter
from xngin.apiserver.webhooks.webhook_types import ExperimentCreatedWebhookBody
from xngin.events.experiment_created import ExperimentCreatedEvent
from xngin.stats.analysis import analyze_experiment as analyze_freq_experiment
from xngin.stats.bandit_analysis import analyze_experiment as analyze_bandit_experiment
from xngin.stats.bandit_sampling import choose_arm as choose_bandit_arm
from xngin.stats.bandit_sampling import update_arm as update_bandit_arm
from xngin.stats.stats_errors import StatsAnalysisError
from xngin.tq.task_payload_types import WEBHOOK_OUTBOUND_TASK_TYPE, WebhookOutboundTask

CSV_PARSE_CHUNK_SIZE_BYTES = 5 << 20


class ExperimentsAssignmentError(Exception):
    """Wrapper for errors raised by our xngin.apiserver.routers.experiments_common module."""


class MismatchedExperimentTypeError(Exception):
    """Error raised when an experiment type is mismatched with the expected type, caused by developer error."""


class CommitExperimentResult(enum.StrEnum):
    COMMITTED = "committed"
    INVALID_STATE = "invalid_state"


class AbandonExperimentResult(enum.StrEnum):
    ABANDONED = "abandoned"
    INVALID_STATE = "invalid_state"


async def fetch_fields_or_raise(
    datasource: tables.Datasource, design_spec: BaseFrequentistDesignSpec, table_name: str, primary_key: str
) -> dict[str, DataType]:
    """Inspect an explicit table/primary_key experiment request and return field metadata.

    Returns: Field name => datatype map.
    Raises: LateValidationError if any fields used in the request are not found or are invalid.
    """
    async with DwhSession(datasource.get_config().dwh) as dwh:
        inspected = await dwh.inspect_table_with_descriptors(table_name, primary_key)

    referenced_fields = {
        *[metric.field_name for metric in design_spec.metrics],
        *[filter_.field_name for filter_ in design_spec.filters],
        *[stratum.field_name for stratum in design_spec.strata],
        primary_key,
    }

    field_type_map = {
        field_name: descriptor.data_type
        for field_name, descriptor in inspected.db_schema.items()
        if field_name in referenced_fields
    }
    if field_type_map is None:
        raise LateValidationError("Experiment design must use valid datasource fields.")

    missing_fields = referenced_fields - field_type_map.keys()
    if missing_fields:
        raise LateValidationError(
            "Design fields are not present in the inspected table: "
            + ", ".join(repr(field) for field in sorted(missing_fields))
        )

    for filter_ in design_spec.filters:
        field_type = field_type_map[filter_.field_name]
        for value in filter_.value:
            validate_filter_value(filter_.field_name, value, field_type)

    return field_type_map


async def fetch_fields_or_raise(
    datasource: tables.Datasource, design_spec: BaseFrequentistDesignSpec, table_name: str, primary_key: str
) -> dict[str, DataType]:
    """Inspect an explicit table/primary_key experiment request and return field metadata.

    Returns: Field name => datatype map.
    Raises: LateValidationError if any fields used in the request are not found, invalid for a
    certain use, or if filter values are invalid for the field type.
    """
    async with DwhSession(datasource.get_config().dwh) as dwh:
        sa_table = await dwh.inspect_table(table_name)
        return await fetch_fields_from_table_or_raise(sa_table, design_spec, primary_key)


async def fetch_fields_from_table_or_raise(
    table: Table,
    design_spec: BaseFrequentistDesignSpec,
    primary_key: str,
) -> dict[str, DataType]:
    """Helper to fetch_fields_or_raise that operates on a pre-inspected SQLAlchemy table."""
    schema_supported_fields_map: dict[str, DataType] = {}
    for column in table.columns.values():
        data_type = DataType.match(column.type)
        if data_type.is_supported():
            schema_supported_fields_map[column.name] = data_type

    referenced_fields = {
        *[metric.field_name for metric in design_spec.metrics],
        *[filter_.field_name for filter_ in design_spec.filters],
        *[stratum.field_name for stratum in design_spec.strata],
        primary_key,
    }

    referenced_fields_and_types = {
        field_name: schema_supported_fields_map[field_name]
        for field_name in referenced_fields
        if field_name in schema_supported_fields_map
    }

    missing_fields = referenced_fields - referenced_fields_and_types.keys()
    if missing_fields:
        raise LateValidationError(
            "The .design_spec field refers to columns that do not exist in the table: "
            f"{', '.join(sorted(missing_fields))}"
        )

    bad_metric_types = [
        m.field_name
        for m in design_spec.metrics
        if not referenced_fields_and_types[m.field_name].is_supported_as_metric()
    ]
    if bad_metric_types:
        raise LateValidationError(
            f"Invalid metric field(s): ({', '.join(bad_metric_types)}). "
            "Only boolean or numeric data types are supported as metrics."
        )

    for filter_ in design_spec.filters:
        field_type = referenced_fields_and_types[filter_.field_name]
        for value in filter_.value:
            validate_filter_value(filter_.field_name, value, field_type)

    return referenced_fields_and_types


async def create_experiment_impl(
    request: CreateExperimentRequest,
    datasource: tables.Datasource,
    xngin_session: AsyncSession,
    desired_n: int | None,
    stratify_on_metrics: bool,
    random_state: int | None,
    validated_webhooks: list[tables.Webhook],
) -> CreateExperimentResponse:
    match request.design_spec.experiment_type:
        case ExperimentsType.FREQ_PREASSIGNED:
            table_name = request.table_name
            primary_key = request.primary_key

            if table_name is None or primary_key is None:
                # Should not actually ever happen as both ParticipantsSchema and
                # CreateExperimentRequest validate that there is a unique ID | primary key.
                raise LateValidationError("Preassigned experiments must have a table name and unique ID field.")
            if desired_n is None:
                raise LateValidationError("Preassigned experiments must have a desired_n.")

            field_type_map = await fetch_fields_or_raise(datasource, request.design_spec, table_name, primary_key)

            # Get participants and their schema info from the client dwh.
            # Only fetch the columns we might need for stratified random assignment.
            metric_names = [m.field_name for m in request.design_spec.metrics]
            strata_names = [s.field_name for s in request.design_spec.strata]
            stratum_cols = strata_names + metric_names if stratify_on_metrics else strata_names

            ds_config = datasource.get_config()
            async with DwhSession(ds_config.dwh) as dwh:
                result = await dwh.get_participants(
                    table_name,
                    select_columns={*stratum_cols, primary_key},
                    filters=request.design_spec.filters,
                    n=desired_n,
                )
                sa_table, participants = result.sa_table, result.participants

            if participants is None:
                raise LateValidationError("Preassigned experiments must have eligible participants data")

            return await create_preassigned_experiment_impl(
                request=request,
                datasource_id=datasource.id,
                organization_id=datasource.organization_id,
                participant_unique_id_field=primary_key,
                dwh_sa_table=sa_table,
                dwh_participants=participants,
                random_state=random_state,
                xngin_session=xngin_session,
                stratify_on_metrics=stratify_on_metrics,
                validated_webhooks=validated_webhooks,
                table_name=table_name,
                field_type_map=field_type_map,
                unique_id_name=primary_key,
            )

        case ExperimentsType.FREQ_ONLINE:
            table_name = request.table_name
            primary_key = request.primary_key

            if table_name is None or primary_key is None:
                raise LateValidationError("Frequentist online experiments must have a table name and unique ID field.")

            field_type_map = await fetch_fields_or_raise(datasource, request.design_spec, table_name, primary_key)

            return await create_freq_online_experiment_impl(
                request=request,
                datasource_id=datasource.id,
                organization_id=datasource.organization_id,
                xngin_session=xngin_session,
                validated_webhooks=validated_webhooks,
                table_name=table_name,
                field_type_map=field_type_map,
                unique_id_name=primary_key,
            )

        case ExperimentsType.MAB_ONLINE | ExperimentsType.CMAB_ONLINE:
            return await create_bandit_online_experiment_impl(
                xngin_session=xngin_session,
                organization_id=datasource.organization_id,
                validated_webhooks=validated_webhooks,
                request=request,
                datasource_id=datasource.id,
                desired_n=desired_n,
            )

        case _:
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
    table_name: str | None,
    field_type_map: dict[str, DataType],
    unique_id_name: str | None,
) -> CreateExperimentResponse:
    """Create a frequentist preassigned experiment and persist it to the database."""

    design_spec = request.design_spec

    if design_spec.experiment_type != ExperimentsType.FREQ_PREASSIGNED:
        raise MismatchedExperimentTypeError(f"can't create preassigned exp of type: {design_spec.experiment_type}")

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

    arm_weights = design_spec.get_validated_arm_weights()

    # Do the raw assignment first so we can store the balance check with the experiment.
    assignment_result = assign_treatments_with_balance(
        sa_table=dwh_sa_table,
        data=dwh_participants,
        stratum_cols=stratum_cols,
        id_col=participant_unique_id_field,
        n_arms=len(design_spec.arms),
        quantiles=4,
        random_state=random_state,
        arm_weights=arm_weights,
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
        table_name=table_name,
        field_type_map=field_type_map,
        unique_id_name=unique_id_name,
    )
    experiment = experiment_converter.get_experiment()
    # Associate webhooks with the experiment
    for webhook in validated_webhooks:
        experiment.webhooks.append(webhook)
    xngin_session.add(experiment)

    await xngin_session.flush()  # Flush to get ids

    await bulk_insert_arm_assignments(
        xngin_session=xngin_session,
        experiment_id=experiment.id,
        arm_ids=[arm.id for arm in experiment.arms],
        participant_type=experiment.participant_type,
        participant_id_col=participant_unique_id_field,
        data=dwh_participants,
        assignment_result=assignment_result,
    )

    await xngin_session.flush()

    assign_summary = await get_assign_summary(
        xngin_session, (await experiment.awaitable_attrs.id), balance_check, ExperimentsType.FREQ_PREASSIGNED
    )
    webhook_ids = [webhook.id for webhook in validated_webhooks]
    return await experiment_converter.get_create_experiment_response(assign_summary, webhook_ids)


async def create_freq_online_experiment_impl(
    request: CreateExperimentRequest,
    datasource_id: str,
    organization_id: str,
    xngin_session: AsyncSession,
    validated_webhooks: list[tables.Webhook],
    table_name: str | None,
    field_type_map: dict[str, DataType],
    unique_id_name: str | None,
) -> CreateExperimentResponse:
    """Create a frequentist online experiment and persist it to the database."""
    design_spec = request.design_spec

    if design_spec.experiment_type != ExperimentsType.FREQ_ONLINE:
        raise MismatchedExperimentTypeError(f"Can't create freq online exp of type: {design_spec.experiment_type}")

    experiment_converter = ExperimentStorageConverter.init_from_components(
        datasource_id=datasource_id,
        organization_id=organization_id,
        experiment_type=ExperimentsType.FREQ_ONLINE,
        design_spec=design_spec,
        table_name=table_name,
        field_type_map=field_type_map,
        unique_id_name=unique_id_name,
    )
    experiment = experiment_converter.get_experiment()
    # Associate webhooks with the experiment
    for webhook in validated_webhooks:
        experiment.webhooks.append(webhook)
    xngin_session.add(experiment)

    await xngin_session.flush()

    # Online experiments start with no assignments.
    empty_assign_summary = AssignSummary(
        balance_check=None,
        sample_size=0,
        arm_sizes=[ArmSize(arm=arm.model_copy(), size=0) for arm in design_spec.arms],
    )
    webhook_ids = [webhook.id for webhook in validated_webhooks]
    return await experiment_converter.get_create_experiment_response(empty_assign_summary, webhook_ids)


async def create_bandit_online_experiment_impl(
    xngin_session: AsyncSession,
    organization_id: str,
    validated_webhooks: list[tables.Webhook],
    request: CreateExperimentRequest,
    datasource_id: str,
    desired_n: int | None = None,
) -> CreateExperimentResponse:
    """Create a bandit experiment and persist it to the database."""
    design_spec = request.design_spec

    if design_spec.experiment_type not in {ExperimentsType.MAB_ONLINE, ExperimentsType.CMAB_ONLINE}:
        raise MismatchedExperimentTypeError(f"can't create bandit exp of type: {design_spec.experiment_type}")

    experiment_converter = ExperimentStorageConverter.init_from_components(
        datasource_id=datasource_id,
        organization_id=organization_id,
        experiment_type=design_spec.experiment_type,
        design_spec=design_spec,
        n_trials=desired_n if desired_n is not None else 0,
    )
    experiment = experiment_converter.get_experiment()
    # Associate webhooks with the experiment
    for webhook in validated_webhooks:
        experiment.webhooks.append(webhook)
    xngin_session.add(experiment)

    await xngin_session.flush()

    # Online experiments start with no assignments.
    empty_assign_summary = AssignSummary(
        balance_check=None,
        sample_size=0,
        arm_sizes=[ArmSize(arm=arm.model_copy(), size=0) for arm in design_spec.arms],
    )
    webhook_ids = [webhook.id for webhook in validated_webhooks]
    return await experiment_converter.get_create_experiment_response(empty_assign_summary, webhook_ids)


async def commit_experiment_impl(xngin_session: AsyncSession, experiment: tables.Experiment) -> CommitExperimentResult:
    if experiment.state == ExperimentState.COMMITTED:
        return CommitExperimentResult.COMMITTED
    if experiment.state != ExperimentState.ASSIGNED:
        return CommitExperimentResult.INVALID_STATE

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

    return CommitExperimentResult.COMMITTED


async def abandon_experiment_impl(experiment: tables.Experiment):
    if experiment.state == ExperimentState.ABANDONED:
        return AbandonExperimentResult.ABANDONED
    if experiment.state not in {ExperimentState.DESIGNING, ExperimentState.ASSIGNED}:
        return AbandonExperimentResult.INVALID_STATE

    experiment.state = ExperimentState.ABANDONED

    return AbandonExperimentResult.ABANDONED


async def get_experiment_impl(
    xngin_session: AsyncSession,
    experiment: tables.Experiment,
) -> GetExperimentResponse:
    converter = ExperimentStorageConverter(experiment)
    assign_summary = await get_assign_summary(
        xngin_session,
        experiment.id,
        converter.get_balance_check(),
        experiment_type=ExperimentsType(experiment.experiment_type),
    )
    webhook_ids = [webhook.id for webhook in experiment.webhooks]
    return await converter.get_experiment_response(assign_summary, webhook_ids)


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
        selectinload(tables.Experiment.experiment_fields).selectinload(tables.ExperimentField.experiment_filters),
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
        items.append(await converter.get_experiment_config(assign_summary, webhook_ids))
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
                float | None,
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


def _participant_passes_filters(experiment: tables.Experiment, participant_props: list[ParticipantProperty]) -> bool:
    if not participant_props or experiment.experiment_type != ExperimentsType.FREQ_ONLINE.value:
        return True

    props_map = {p.field_name: p.value for p in participant_props}
    experiment_converter = ExperimentStorageConverter(experiment)
    field_map = experiment_converter.get_field_type_map()
    return passes_filters(props_map, field_map, experiment_converter.get_design_spec_filters())


async def get_or_create_assignment_for_participant(
    xngin_session: AsyncSession,
    experiment: tables.Experiment,
    participant_id: str,
    create_if_none: bool,
    properties: list[ParticipantProperty] | None,
    random_state: int | None = None,
) -> GetParticipantAssignmentResponse:
    """Get or create the arm assignment for a specific participant in a non-CMAB experiment.

    If properties are provided for a FREQ_ONLINE experiment, they are used to filter the participant
    to determine eligibility.  Assignment is None if it does not pass the filters.

    Set create_if_none=False to only get an assignment if it already exists; do not create a new one.
    """

    assignment = await get_existing_assignment_for_participant(
        xngin_session=xngin_session,
        experiment_id=experiment.id,
        participant_id=participant_id,
        experiment_type=experiment.experiment_type,
    )

    if not assignment and create_if_none and experiment.stopped_assignments_at is None:
        if experiment.experiment_type == ExperimentsType.CMAB_ONLINE.value:
            raise LateValidationError(
                f"New arm assignments for {ExperimentsType.CMAB_ONLINE.value} cannot be created at this endpoint, "
                f"please use the corresponding POST endpoint instead."
            )

        if not properties or _participant_passes_filters(experiment, properties):
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


def choose_online_arm(
    experiment: tables.Experiment,
    random_state: int | None = None,
) -> tables.Arm:
    """Choose an arm for online experiments using simple or weighted random assignment depending on its design."""
    # Sort by arm name to ensure deterministic assignment with seed for tests.
    sorted_arms = sorted(experiment.arms, key=lambda a: a.name)
    arm_weights_as_probabilities = None

    arm_id_to_weight = {arm.id: arm.arm_weight for arm in experiment.arms if arm.arm_weight is not None}
    if len(arm_id_to_weight) == len(experiment.arms):
        # Convert to probabilities for weighted random selection.
        arm_weights_sorted = [arm_id_to_weight[arm.id] for arm in sorted_arms]
        sum_weights = sum(arm_weights_sorted)
        arm_weights_as_probabilities = [w / sum_weights for w in arm_weights_sorted]

    rng = np.random.default_rng(random_state)
    index = rng.choice(len(sorted_arms), p=arm_weights_as_probabilities)
    return sorted_arms[index]


async def create_assignment_for_participant(
    xngin_session: AsyncSession,
    experiment: tables.Experiment,
    participant_id: str,
    sorted_context_vals: list[float] | None = None,
    random_state: int | None = None,
) -> Assignment | None:
    """Helper to persist a new assignment for a participant. Returned value excludes strata.

    sorted_context_vals are assumed to be sorted by the experiment's corresponding context ids in ascending order.

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
        if not sorted_context_vals:
            raise ExperimentsAssignmentError(
                "Context values are required for contextual multi-armed bandit experiments"
            )
        if len(sorted_context_vals) != len(experiment.contexts):
            raise ExperimentsAssignmentError(
                f"Expected {len(experiment.contexts)} context values, got {len(sorted_context_vals)}"
            )

    # Don't allow new assignments for experiments that have ended.
    if experiment.end_date < datetime.now(UTC):
        experiment.stopped_assignments_at = datetime.now(UTC)
        experiment.stopped_assignments_reason = StopAssignmentReason.END_DATE
        await xngin_session.commit()
        return None

    # For online frequentist or Bayesian A/B experiments, create a new assignment
    # with simple random assignment or weighted random assignment if arm_weights are specified.
    match experiment_type:
        case ExperimentsType.FREQ_ONLINE | ExperimentsType.BAYESAB_ONLINE:
            chosen_arm = choose_online_arm(experiment=experiment, random_state=random_state)
        case ExperimentsType.MAB_ONLINE | ExperimentsType.CMAB_ONLINE:
            chosen_arm = choose_bandit_arm(
                experiment=experiment,
                sorted_context_vals=sorted_context_vals,
                random_state=random_state,
            )

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
                            context_vals=sorted_context_vals,
                        )
                        .returning(tables.Draw.created_at)
                    )
                ).fetchone()
            case _:
                raise ExperimentsAssignmentError(f"Invalid experiment type: {experiment_type}")

        if result is None:
            raise ExperimentsAssignmentError(f"Failed to create assignment for participant '{participant_id}'")
        created_at = result[0]
        stmt = (
            pg_insert(tables.ArmStats)
            .values(arm_id=chosen_arm_id, population=1)
            .on_conflict_do_update(
                index_elements=[tables.ArmStats.arm_id],
                set_={"population": tables.ArmStats.population + 1},
            )
        )
        await xngin_session.execute(stmt)
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
        context_values=sorted_context_vals,
    )


async def update_bandit_arm_with_outcome_impl(
    xngin_session: AsyncSession,
    experiment: tables.Experiment,
    participant_id: str,
    outcome: float,
) -> tables.Arm:
    """Update the Draw table with the outcome for a bandit experiment."""
    # Not supported for frequentist experiments
    design_spec = await ExperimentStorageConverter(experiment).get_design_spec()

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
        stmt = (
            select(tables.Draw)
            .where(
                tables.Draw.experiment_id == experiment.id,
                tables.Draw.arm_id == draw_record.arm_id,
                tables.Draw.outcome.is_not(None),
            )
            .order_by(tables.Draw.created_at.desc())
            .limit(100)  # TODO: Make draw limiting configurable
        )

        relevant_draws = await xngin_session.scalars(stmt)

        outcomes = [outcome] + [cast(float, d.outcome) for d in relevant_draws]
        context_vals = (
            None
            if draw_record.context_vals is None
            else ([draw_record.context_vals] + [cast(list[float], d.context_vals) for d in relevant_draws])
        )

        updated_parameters = update_bandit_arm(
            experiment=experiment,
            arm_to_update=arm_to_update,
            outcomes=outcomes,
            context=context_vals,
        )

        # Update the draw record and arm with the new parameters
        match updated_parameters:
            case UpdateTypeBeta():
                draw_record.current_alpha = updated_parameters.alpha
                draw_record.current_beta = updated_parameters.beta
                arm_to_update.alpha = updated_parameters.alpha
                arm_to_update.beta = updated_parameters.beta
            case UpdateTypeNormal():
                draw_record.current_mu = updated_parameters.mu
                draw_record.current_covariance = updated_parameters.covariance
                arm_to_update.mu = updated_parameters.mu
                arm_to_update.covariance = updated_parameters.covariance
            case _:
                raise ExperimentsAssignmentError(
                    f"Unsupported prior update type: {type(updated_parameters)} for prior type {experiment.prior_type}"
                )

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
    balance_check: BalanceCheck | None,
    experiment_type: ExperimentsType,
) -> AssignSummary:
    """Constructs an AssignSummary from the experiment's arms and arm_assignments."""
    result = await xngin_session.execute(
        select(
            tables.Arm.id,
            tables.Arm.name,
            func.coalesce(tables.ArmStats.population, 0),
        )
        .outerjoin(tables.ArmStats, tables.Arm.id == tables.ArmStats.arm_id)
        .where(tables.Arm.experiment_id == experiment_id)
        .order_by(tables.Arm.position)
    )
    arm_sizes = [
        ArmSize(
            arm=Arm(arm_id=arm_id, arm_name=name),
            size=count,
        )
        for arm_id, name, count in result
    ]

    if experiment_type in {ExperimentsType.MAB_ONLINE, ExperimentsType.CMAB_ONLINE}:
        balance_check = None
    return AssignSummary(
        balance_check=balance_check,
        arm_sizes=arm_sizes,
        sample_size=sum(arm_size.size for arm_size in arm_sizes),
    )


async def analyze_experiment_freq_impl(
    xngin_session: AsyncSession,
    dsconfig: DatasourceConfig,
    experiment: tables.Experiment,
    baseline_arm_id: str,
    metrics: list[DesignSpecMetricRequest],
) -> FreqExperimentAnalysisResponse:
    """Analyze a frequentist experiment. Assumes arms and arm_assignments are preloaded."""

    unique_id_field = experiment.unique_id_field
    if experiment.datasource_table is None or unique_id_field is None:
        raise StatsAnalysisError("Experiment must have a datasource table and unique ID field to analyze.")

    participant_ids, assignments_df = await read_assignments_efficiently(xngin_session, experiment.id)
    if assignments_df.empty:
        raise StatsAnalysisError("No participants found for experiment.")
    async with DwhSession(dsconfig.dwh) as dwh:
        sa_table = await dwh.inspect_table(experiment.datasource_table)

        # Mark the start of the analysis as when we begin pulling outcomes.
        created_at = datetime.now(UTC)
        participant_outcomes = await asyncio.to_thread(
            get_participant_metrics,
            dwh.session,
            sa_table,
            metrics,
            unique_id_field.field_name,
            participant_ids,
        )

    if len(participant_outcomes) == 0:
        raise StatsAnalysisError(
            "No assigned participants found in the datasource. Check that "
            f"ids used in assignment are usable with your unique identifier ({unique_id_field.field_name}), and "
            "that metric data exists for them."
        )

    # We want to notify the user if there are participants assigned to the experiment that are not
    # in the data warehouse. E.g. in an online experiment, perhaps a new user was assigned
    # before their info was synced to the dwh.
    num_participants = len(participant_ids)
    num_missing_participants = num_participants - len(participant_outcomes)

    analyze_results = analyze_freq_experiment(
        assignments_df,
        participant_outcomes,
        baseline_arm_id,
        alpha=experiment.alpha,
    )

    metric_analyses = []
    for metric in metrics:
        metric_name = metric.field_name
        arm_analyses = []
        for arm in experiment.arms:
            if arm.id in analyze_results[metric_name]:
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
                        ci_lower=arm_result.ci_lower,
                        ci_upper=arm_result.ci_upper,
                        mean_ci_lower=arm_result.mean_ci_lower,
                        mean_ci_upper=arm_result.mean_ci_upper,
                        num_missing_values=arm_result.num_missing_values,
                    )
                )
            else:
                # If arm.id is missing due to no participants or partcipants with outcomes yet, append a default
                arm_analyses.append(
                    ArmAnalysis(
                        arm_id=arm.id,
                        arm_name=arm.name,
                        arm_description=arm.description,
                        is_baseline=arm.id == baseline_arm_id,
                        estimate=0,
                        p_value=float("nan"),
                        t_stat=float("nan"),
                        std_error=float("nan"),
                        ci_lower=float("nan"),
                        ci_upper=float("nan"),
                        mean_ci_lower=float("nan"),
                        mean_ci_upper=float("nan"),
                        num_missing_values=-1,  # -1 indicates arm analysis not available
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


async def read_assignments_efficiently(xngin_session: AsyncSession, experiment_id: str) -> tuple[list[str], DataFrame]:
    """Reads assignments directly from Postgres via a COPY statement.

    Reads CSV output in row-bounded chunks and concatenates the parsed frames.
    """
    select_query = t"SELECT arm_id, participant_id FROM arm_assignments WHERE experiment_id = {experiment_id}"  # type: ignore
    dfs = [
        pd.read_csv(io.BytesIO(chunk), names=["arm_id", "participant_id"], dtype=str)
        async for chunk in select_as_csv(
            xngin_session, select_query, buffer_size_bytes=CSV_PARSE_CHUNK_SIZE_BYTES, newline_framed=True
        )
    ]
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["arm_id", "participant_id"]).astype(str)
    return df["participant_id"].to_list(), df


def analyze_experiment_bandit_impl(
    experiment: tables.Experiment,
    context_vals: list[float] | None = None,
) -> BanditExperimentAnalysisResponse:
    """Analyze a bandit experiment. Assumes arms and draws are preloaded."""

    draws = experiment.draws
    outcomes = [draw.outcome for draw in draws if draw.outcome is not None]

    outcome_std_dev = np.std(outcomes).astype(float) if len(outcomes) > 1 else 0.0
    arm_analyses = analyze_bandit_experiment(
        experiment=experiment,
        outcome_std_dev=outcome_std_dev,
        context_vals=context_vals,
    )

    return BanditExperimentAnalysisResponse(
        experiment_id=experiment.id,
        arm_analyses=arm_analyses,
        n_outcomes=len(outcomes),
        created_at=datetime.now(UTC),
        contexts=context_vals,
    )
