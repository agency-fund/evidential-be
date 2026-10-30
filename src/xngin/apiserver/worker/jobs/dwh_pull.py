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

import sentry_sdk
from loguru import logger
from sqlalchemy import ColumnElement, and_, exists, false, func, select
from sqlalchemy.orm import Session

from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_enums import ExperimentState, ExperimentsType
from xngin.apiserver.sqla import tables
from xngin.apiserver.worker.jobs.bandit_outcomes import Outcome, OutcomeReport, apply_outcomes
from xngin.apiserver.worker.jobs.base import (
    Backoff,
    JobResult,
    JobSucceeded,
    next_run_in_window,
)

# How long pulls continue after the end date whatever the draws' state. This guarantees a pull after the last draw,
# including any still committing at the end date, before compute_next_run_at decides from the pending draws whether
# to keep pulling. Nothing else reschedules the job, so without it a draw made just before the end could go unpulled.
PULL_AFTER_END = timedelta(days=1)

# How long pulls continue after PULL_AFTER_END while any draw still has no outcome. Outcomes for draws made near the
# end date can land in the warehouse days later, after a slow ETL or a warehouse outage, and PULL_AFTER_END alone
# would stop pulling before they arrive. The grace also lets the worker catch up after an outage of its own. Pulls end
# earlier once every draw has an outcome, so the grace only costs warehouse queries for experiments still missing
# outcomes, and only for this long, even if the warehouse never fills them.
LATE_OUTCOME_GRACE = timedelta(days=7)


def _is_pending_draw(experiment_id: str) -> ColumnElement[bool]:
    return and_(tables.Draw.experiment_id == experiment_id, tables.Draw.outcome.is_(None))


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
    # deletes them waits too.
    pending_ids = list(
        session.scalars(
            select(tables.Draw.participant_id)
            .where(_is_pending_draw(experiment.id))
            .order_by(tables.Draw.created_at)
            .with_for_update(of=tables.Draw, key_share=True)
        )
    )
    if not pending_ids:
        return OutcomeReport()

    # 2. Read the external DWH, under a deadline so a wedged warehouse cannot hold the locks
    # taken above indefinitely.
    with DwhSession.open(dsconfig.dwh, timeout=dwh_timeout) as dwh:
        participant_outcomes = dwh.get_participant_metrics(
            table_name,
            metric_names=[target_field_name],
            unique_id_field=unique_id_field_name,
            participant_ids=pending_ids,
        )

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
    """Pull each committed MAB_ONLINE_DWH experiment's outcomes every interval.

    Pulls start at the experiment's start date, not a day earlier as snapshots do, because there are rarely draws to
    pull before then. They run until PULL_AFTER_END past the end date, and then continue for up to LATE_OUTCOME_GRACE
    while any draw still has no outcome.
    """

    name: ClassVar[str] = "dwh_pull"
    interval: timedelta
    dwh_timeout: float
    backoff: Backoff

    def needs_refresh(self) -> ColumnElement[bool]:
        return false()

    def compute_next_run_at(
        self, session: Session, experiment: tables.Experiment, status: tables.JobStatus, now: datetime
    ) -> datetime | None:
        if (
            experiment.experiment_type != ExperimentsType.MAB_ONLINE_DWH
            or experiment.state != ExperimentState.COMMITTED
        ):
            return None
        last_success_at = status.last_success_at
        if last_success_at is None:
            # The job has never succeeded, but the latest outcome an earlier pull recorded shows when it last ran, so
            # a new deployment doesn't pull every experiment at once. This fallback can be removed once the worker is
            # fully deployed.
            last_success_at = session.scalar(
                select(func.max(tables.Draw.observed_at)).where(tables.Draw.experiment_id == experiment.id)
            )
        pulls_close = experiment.end_date + PULL_AFTER_END
        run_at = next_run_in_window(
            last_success_at, now, self.interval, opens=experiment.start_date, closes=pulls_close
        )
        # The API stops assigning at the end date, so after PULL_AFTER_END pending draws can only resolve, and
        # no other change needs to reschedule the job: needs_refresh can stay false.
        if run_at is None and session.scalar(select(exists().where(_is_pending_draw(experiment.id)))):
            run_at = next_run_in_window(
                last_success_at,
                now,
                self.interval,
                opens=experiment.start_date,
                closes=pulls_close + LATE_OUTCOME_GRACE,
            )
        return run_at

    def execute(self, session: Session, experiment: tables.Experiment) -> JobResult:
        report = pull_one_experiment(session, experiment, dwh_timeout=self.dwh_timeout)
        logger.info(f"Pulled outcomes: ingested={report.ingested} invalid={report.invalid} pending={report.pending}")
        for status, count in (("applied", report.ingested), ("invalid", report.invalid), ("pending", report.pending)):
            sentry_sdk.metrics.count("worker.dwh_pull.outcomes", count, attributes={"status": status})
        return JobSucceeded()
