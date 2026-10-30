"""Reads bandit outcomes from an organization's data warehouse.

MAB_ONLINE_DWH experiments have no outcomes pushed to the API. For those, this module finds draws
that still have no outcome, reads the experiment's target column for those participants, and applies
each value through the same per-outcome update the push API uses.

DwhPullJob runs one experiment's pull inside a worker attempt. Its outcomes land with the attempt's commit
or not at all: a pull that fails partway rolls back, and the job's backoff schedules a retry.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import ClassVar

import google.api_core.exceptions
import sentry_sdk
import sqlalchemy.exc
from loguru import logger
from sqlalchemy import ColumnElement, case, literal, select
from sqlalchemy.orm import Session

from xngin.apiserver.dwh.dwh_session import CannotFindTableError, DwhSession
from xngin.apiserver.exceptions_common import (
    DwhConnectionError,
    DwhDatabaseDoesNotExistError,
    DwhTimeoutError,
    LateValidationError,
)
from xngin.apiserver.routers.common_enums import ExperimentsType
from xngin.apiserver.sqla import tables
from xngin.apiserver.worker.jobs.bandit_outcomes import Outcome, OutcomeReport, apply_outcomes, is_pending_draw
from xngin.apiserver.worker.jobs.base import (
    Backoff,
    JobFailed,
    JobResult,
    JobSucceeded,
    next_run_in_window,
)

# How long pulls continue after the end date while any draw still has no outcome. The API stops assigning at the end
# date, but outcomes for draws made near it can land in the warehouse days later, after a slow ETL or a warehouse
# outage. The grace also lets the worker catch up after an outage of its own. Pulls stop once every draw has an
# outcome, so the grace only costs warehouse queries for experiments still missing outcomes, and only for this long,
# even if the warehouse never fills them.
LATE_OUTCOME_GRACE = timedelta(days=8)


class WarehouseReadError(Exception):
    """Reading the organization's data warehouse failed. The cause is the warehouse's own error."""


# What a warehouse read raises when the organization's warehouse fails, rather than our code: errors DwhSession
# classifies, and the database driver's own errors during the read. Anything else is a bug and should alert.
# See EVE-267.
WAREHOUSE_FAILURES = (
    CannotFindTableError,
    DwhConnectionError,
    DwhDatabaseDoesNotExistError,
    DwhTimeoutError,
    sqlalchemy.exc.DBAPIError,
    google.api_core.exceptions.GoogleAPICallError,
)


def pull_one_experiment(session: Session, experiment: tables.Experiment, *, dwh_timeout: float) -> OutcomeReport:
    """Read newly landed outcomes for one MAB_ONLINE_DWH experiment and apply them.

    Runs in the caller's transaction, which DwhPullJob.execute receives from the worker, in three phases:
    1. Read what the warehouse query needs, and lock the draws we intend to fill,
    2. Read the warehouse,
    3. Compute + apply the values.

    Either every valid outcome lands or none does: if anything raises, the job's savepoint rolls back, so a
    failure never leaves an experiment half pulled. Values the warehouse has not filled in yet, and values that
    fail validation, stay NULL and are re-read on a later run.

    The locks taken in phase 1 are held until the attempt's transaction ends, or until the job's savepoint rolls
    back if the pull fails. This means dwh_timeout is a lower limit on that lock duration, so we should be careful.
    """
    # 1. Read what the warehouse query needs: the datasource config, the target and unique-id
    # columns, and the draws that still have no outcome.
    unique_id_field = experiment.unique_id_field()
    target_field = next((ef for ef in experiment.experiment_fields if ef.is_target), None)
    if experiment.datasource_table is None or unique_id_field is None or target_field is None:
        # Unreachable through the API, which requires all three when creating a MAB-DWH experiment.
        raise LateValidationError(
            "MAB-DWH experiment is missing its datasource table, unique id field, or target field."
        )
    dsconfig = experiment.datasource.get_config()
    table_name = experiment.datasource_table
    unique_id_field_name = unique_id_field.field_name
    target_field_name = target_field.field_name

    # FOR NO KEY UPDATE holds these draws for the rest of the transaction while still letting
    # foreign keys reference them. Taking the lock here, before the warehouse read, makes other writes
    # to these draws wait for the pull instead of changing them partway through it, and a data reset that
    # deletes them waits too. Draws another transaction already holds are skipped and left for a later pull:
    # deleting an experiment locks its draws before the experiment row, which the worker's claim holds, so
    # waiting for them here would deadlock.
    pending_ids = list(
        session.scalars(
            select(tables.Draw.participant_id)
            .where(is_pending_draw(experiment.id))
            .order_by(tables.Draw.created_at)
            .with_for_update(of=tables.Draw, key_share=True, skip_locked=True)
        )
    )
    if not pending_ids:
        return OutcomeReport()

    # 2. Read the external DWH, under a deadline so a wedged warehouse cannot hold the locks
    # taken above indefinitely.
    try:
        with DwhSession.open(dsconfig.dwh, timeout=dwh_timeout) as dwh:
            participant_outcomes = dwh.get_participant_metrics(
                table_name,
                metric_names=[target_field_name],
                unique_id_field=unique_id_field_name,
                participant_ids=pending_ids,
            )
    except WAREHOUSE_FAILURES as exc:
        raise WarehouseReadError(f"{type(exc).__name__}: {exc}") from exc

    # 3. Compute the value per participant, then apply them.
    values_by_participant: dict[str, float | None] = {
        po.participant_id: next(
            (mv.metric_value for mv in po.metric_values if mv.metric_name == target_field_name),
            None,
        )
        for po in participant_outcomes
    }

    outcomes = (Outcome(participant_id, values_by_participant.get(participant_id)) for participant_id in pending_ids)
    return apply_outcomes(session, experiment, outcomes, skip_invalid=True)


@dataclass(slots=True)
class DwhPullJob:
    """Pull each committed MAB_ONLINE_DWH experiment's outcomes every interval while any draw has no outcome.

    Pulls start at the experiment's start date, not a day earlier as snapshots do, because there are rarely draws to
    pull before then, and run until LATE_OUTCOME_GRACE past the end date. A pull with no pending draws would have
    nothing to do, so the job isn't scheduled then; a newly inserted draw schedules it on the next look.
    """

    name: ClassVar[str] = "dwh_pull"
    interval: timedelta
    dwh_timeout: float
    backoff: Backoff

    def next_run_at(self, now: ColumnElement[datetime]) -> ColumnElement[datetime | None]:
        experiment = tables.Experiment
        # An index probe on ix_draws_pending. LIMIT 1 keeps Postgres from planning EXISTS as one scan of every
        # experiment's pending draws.
        has_pending_draws = (
            select(literal(True).label("pending"))
            .where(is_pending_draw(experiment.id))
            .limit(1)
            .scalar_subquery()
            .is_not(None)
        )
        # CASE evaluates its conditions in order, unlike AND, so only DWH experiments read their draws.
        return case(
            (experiment.experiment_type != ExperimentsType.MAB_ONLINE_DWH, None),
            (
                has_pending_draws,
                next_run_in_window(
                    now,
                    self.interval,
                    opens=experiment.start_date,
                    closes=experiment.end_date + LATE_OUTCOME_GRACE,
                    name="pull_window",
                ),
            ),
        )

    def execute(self, session: Session, experiment: tables.Experiment) -> JobResult:
        try:
            report = pull_one_experiment(session, experiment, dwh_timeout=self.dwh_timeout)
        except (LateValidationError, WarehouseReadError) as exc:
            # The experiment's configuration or the organization's warehouse failed, not our code, so this logs
            # without alerting. Both fail before any outcome is written, so the failure has nothing to roll back.
            logger.opt(exception=exc).warning("DWH pull failed.")
            return JobFailed(error=f"{type(exc).__name__}: {exc}")
        logger.info(f"Pulled outcomes: ingested={report.ingested} invalid={report.invalid} pending={report.pending}")
        for status, count in (("applied", report.ingested), ("invalid", report.invalid), ("pending", report.pending)):
            sentry_sdk.metrics.count("worker.dwh_pull.outcomes", count, attributes={"status": status})
        return JobSucceeded()
