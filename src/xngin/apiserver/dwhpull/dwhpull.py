"""Reads bandit outcomes from an organization's data warehouse.

MAB_ONLINE_DWH experiments have no outcomes pushed to the API. For those, this module finds draws
that still have no outcome, reads the experiment's target column for those participants, and applies
each value through the same per-outcome update the push API uses.
"""

import asyncio
from dataclasses import dataclass

import sentry_sdk
from loguru import logger
from sqlalchemy import func, select, text
from sqlalchemy.orm import selectinload

from xngin.apiserver import database
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.dwh.participant_metrics_queries import get_participant_metrics
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import DesignSpecMetricRequest
from xngin.apiserver.routers.common_enums import ExperimentState, ExperimentsType
from xngin.apiserver.routers.experiments.experiments_common import (
    ExperimentsAssignmentError,
    MismatchedExperimentTypeError,
    update_bandit_arm_with_outcome_impl,
)
from xngin.apiserver.sqla import tables

# The amount of time one experiment's pull may run before it is abandoned. The pull job can specify a
# different timeout via command line flags.
PULL_TIMEOUT_SECS = 90


@dataclass(slots=True)
class PullReport:
    """Counts from one experiment's pull."""

    ingested: int = 0
    invalid: int = 0
    pending: int = 0

    def summary(self) -> str:
        return f"ingested={self.ingested} invalid={self.invalid} pending={self.pending}"


async def select_experiments_to_pull() -> list[str]:
    """Return the ids of the experiments whose outcomes we read from a data warehouse.

    Covers committed MAB-DWH experiments that are running, start tomorrow, or ended yesterday. The
    day either side gives outcomes that land late a chance to arrive.

    This method establishes its own database connection.
    """
    buffer = text("interval '1 day'")
    async with database.async_session() as session:
        return list(
            (
                await session.execute(
                    select(tables.Experiment.id)
                    .where(
                        tables.Experiment.experiment_type == ExperimentsType.MAB_ONLINE_DWH.value,
                        tables.Experiment.state == ExperimentState.COMMITTED.value,
                        func.now().between(
                            func.date_trunc("minute", tables.Experiment.start_date - buffer),
                            func.date_trunc("minute", tables.Experiment.end_date + buffer),
                        ),
                    )
                    .order_by(tables.Experiment.id)
                )
            ).scalars()
        )


async def pull_one_experiment(experiment_id: str) -> PullReport:
    """Read newly landed outcomes for one MAB_ONLINE_DWH experiment and apply them.

    Opens its own database sessions so that each outcome commits independently of any transaction
    held by the caller.

    An interrupted run keeps the outcomes already committed; the next run picks up the remainder
    because this selects only draws that still have no outcome.
    """
    experiment_query = (
        select(tables.Experiment)
        .where(tables.Experiment.id == experiment_id)
        .options(
            selectinload(tables.Experiment.arms),
            selectinload(tables.Experiment.contexts),
            selectinload(tables.Experiment.datasource),
            selectinload(tables.Experiment.experiment_fields),
        )
    )

    async with database.async_session() as session:
        experiment = (await session.execute(experiment_query)).scalar_one()
        if ExperimentsType(experiment.experiment_type) != ExperimentsType.MAB_ONLINE_DWH:
            raise MismatchedExperimentTypeError(f"Cannot pull outcomes for a {experiment.experiment_type} experiment.")
        unique_id_field = experiment.unique_id_field()
        target_field = next((ef for ef in experiment.experiment_fields if ef.is_target), None)
        if experiment.datasource_table is None or unique_id_field is None or target_field is None:
            raise LateValidationError(
                "MAB-DWH experiment is missing its datasource table, unique id field, or target field."
            )
        dsconfig = experiment.datasource.get_config()
        table_name = experiment.datasource_table
        unique_id_field_name = unique_id_field.field_name
        target_field_name = target_field.field_name

        pending_ids = list(
            (
                await session.execute(
                    select(tables.Draw.participant_id)
                    .where(
                        tables.Draw.experiment_id == experiment_id,
                        tables.Draw.outcome.is_(None),
                    )
                    .order_by(tables.Draw.created_at)
                )
            ).scalars()
        )
    if not pending_ids:
        return PullReport()

    async with DwhSession(dsconfig.dwh) as dwh:
        sa_table = await dwh.inspect_table(table_name)
        # model_construct: get_participant_metrics only reads field_name; the power-analysis fields
        # the validator demands (metric_pct_change/metric_target) don't apply here.
        target_metric = DesignSpecMetricRequest.model_construct(field_name=target_field_name)
        participant_outcomes = await asyncio.to_thread(
            get_participant_metrics,
            dwh.session,
            sa_table,
            [target_metric],
            unique_id_field_name,
            pending_ids,
        )

    values_by_participant: dict[str, float | None] = {
        po.participant_id: next(
            (mv.metric_value for mv in po.metric_values if mv.metric_name == target_field_name),
            None,
        )
        for po in participant_outcomes
    }

    async with database.async_session() as session:
        report = PullReport()
        experiment = (await session.execute(experiment_query)).scalar_one()
        for participant_id in pending_ids:
            value = values_by_participant.get(participant_id)
            if value is None:
                # Not in the DWH yet, or the target column is still NULL: re-read on a later run.
                report.pending += 1
                continue
            try:
                await update_bandit_arm_with_outcome_impl(
                    xngin_session=session,
                    experiment=experiment,
                    participant_id=participant_id,
                    outcome=value,
                )
                # The update leaves the transaction open for its caller. Commit per outcome so an
                # interrupted run keeps what it already applied.
                await session.commit()
                report.ingested += 1
            except (ExperimentsAssignmentError, LateValidationError) as exc:
                # Value failed the reward/target validation (or a concurrent writer already recorded
                # an outcome). Rejected draws keep outcome=NULL, so a corrected DWH value is picked
                # up on a later run.
                await session.rollback()
                logger.info(f"{experiment_id}: skipping outcome for participant '{participant_id}': {exc}")
                report.invalid += 1
                # rollback() expires every object in the session; reload so later iterations don't
                # trigger sync lazy loads on expired attributes.
                experiment = (await session.execute(experiment_query)).scalar_one()
    return report


async def pull_all_experiments(pull_timeout: int) -> None:
    """Pull outcomes for every experiment that needs them, one experiment at a time.

    A failure on one experiment is reported and does not stop the others.
    """
    experiment_ids = await select_experiments_to_pull()
    logger.info(f"Pulling outcomes for {len(experiment_ids)} experiments.")
    sentry_sdk.metrics.count("dwh_pull.experiments", len(experiment_ids))

    failures = 0
    for experiment_id in experiment_ids:
        with logger.contextualize(experiment_id=experiment_id):
            try:
                async with asyncio.timeout(pull_timeout):
                    report = await pull_one_experiment(experiment_id)
            except Exception as exc:
                failures += 1
                logger.opt(exception=exc).error(f"{experiment_id}: pull failed")
                sentry_sdk.capture_exception(exc)
                sentry_sdk.metrics.count("dwh_pull.failed", 1, attributes={"experiment_id": experiment_id})
                continue
            logger.info(f"{experiment_id}: {report.summary()}")
            attributes = {"experiment_id": experiment_id}
            sentry_sdk.metrics.count("dwh_pull.ingested", report.ingested, attributes=attributes)
            sentry_sdk.metrics.count("dwh_pull.invalid", report.invalid, attributes=attributes)
            sentry_sdk.metrics.count("dwh_pull.pending", report.pending, attributes=attributes)

    logger.info(f"Pulled outcomes for {len(experiment_ids) - failures} experiments, {failures} failed.")
