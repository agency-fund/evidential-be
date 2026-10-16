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
from sqlalchemy import ColumnElement, false, func, select
from sqlalchemy.orm import Session, selectinload

from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.dwh.participant_metrics_queries import get_participant_metrics
from xngin.apiserver.routers.common_api_types import DesignSpecMetricRequest
from xngin.apiserver.routers.common_enums import ExperimentsType
from xngin.apiserver.sqla import tables
from xngin.apiserver.worker.jobs.bandit_outcomes import Outcome, OutcomeReport, apply_outcomes
from xngin.apiserver.worker.jobs.base import (
    Backoff,
    JobResult,
    JobSucceeded,
    periodic_next_run_at,
)


def pull_one_experiment(session: Session, experiment_id: str, *, dwh_timeout: float) -> OutcomeReport:
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

    # 1. Read what the warehouse query needs: the datasource config, the target and unique-id
    # columns, and the draws that still have no outcome.
    experiment = session.execute(experiment_query).scalar_one()
    unique_id_field = experiment.unique_id_field()
    target_field = next((ef for ef in experiment.experiment_fields if ef.is_target), None)
    if experiment.datasource_table is None or unique_id_field is None or target_field is None:
        # Unreachable through the API, which requires all three when creating a MAB-DWH experiment.
        raise AssertionError("MAB-DWH experiment is missing its datasource table, unique id field, or target field.")
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
            .where(
                tables.Draw.experiment_id == experiment_id,
                tables.Draw.outcome.is_(None),
            )
            .order_by(tables.Draw.created_at)
            .with_for_update(of=tables.Draw, key_share=True)
        )
    )
    if not pending_ids:
        return OutcomeReport()

    # 2. Read the external DWH, under a deadline so a wedged warehouse cannot hold the locks
    # taken above indefinitely.
    with DwhSession.open(dsconfig.dwh, timeout=dwh_timeout) as dwh:
        sa_table = dwh.inspect_table(table_name)
        # model_construct: get_participant_metrics only reads field_name; the power-analysis
        # fields the validator demands (metric_pct_change/metric_target) don't apply here.
        target_metric = DesignSpecMetricRequest.model_construct(field_name=target_field_name)
        participant_outcomes = dwh.run(
            get_participant_metrics,
            sa_table,
            [target_metric],
            unique_id_field_name,
            pending_ids,
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
    name: ClassVar[str] = "dwh_pull"
    interval: timedelta
    dwh_timeout: float
    backoff: Backoff

    def needs_refresh(self) -> ColumnElement[bool]:
        return false()

    def compute_next_run_at(
        self, session: Session, experiment: tables.Experiment, status: tables.JobStatus, now: datetime
    ) -> datetime | None:
        if experiment.experiment_type != ExperimentsType.MAB_ONLINE_DWH:
            return None
        last_success_at = status.last_success_at
        if last_success_at is None:
            # The job has never succeeded, but the latest outcome an earlier pull recorded shows when it last ran, so
            # a new deployment doesn't pull every experiment at once. This fallback can be removed once the worker is
            # fully deployed.
            last_success_at = session.scalar(
                select(func.max(tables.Draw.observed_at)).where(tables.Draw.experiment_id == experiment.id)
            )
        return periodic_next_run_at(experiment, last_success_at, now, self.interval)

    def execute(self, session: Session, experiment_id: str) -> JobResult:
        report = pull_one_experiment(session, experiment_id, dwh_timeout=self.dwh_timeout)
        logger.info(f"Pulled outcomes: ingested={report.ingested} invalid={report.invalid} pending={report.pending}")
        for status, count in (("applied", report.ingested), ("invalid", report.invalid), ("pending", report.pending)):
            sentry_sdk.metrics.count("worker.dwh_pull.outcomes", count, attributes={"status": status})
        return JobSucceeded()
