"""Scheduled bandit arm updates from newly recorded outcomes.

Outcomes arrive from three places (the outcome API, AutofailJob, and DwhPullJob), and each one only records the
outcome on its draw. BanditUpdateJob is what turns those outcomes into arm parameters: every update_frequency_hours
it applies the outcomes recorded since the last update to each arm's current parameters, and marks them applied.
The marks commit with the new parameters, so a failed or retried run never counts an outcome twice, and the next
successful run catches up on everything recorded in the meantime.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

import sentry_sdk
from loguru import logger
from sqlalchemy import ColumnElement, and_, case, func
from sqlalchemy.orm import Session

from xngin.apiserver.routers.common_enums import ExperimentsType
from xngin.apiserver.routers.experiments.experiments_common import update_bandit_arms_impl
from xngin.apiserver.sqla import tables
from xngin.apiserver.worker.jobs.base import Backoff, JobResult, JobSucceeded, next_run_in_window
from xngin.apiserver.worker.jobs.dwh_pull import LATE_OUTCOME_GRACE


@dataclass(slots=True)
class BanditUpdateJob:
    """Update each committed bandit experiment's arms every update_frequency_hours.

    Runs from the start date until LATE_OUTCOME_GRACE past the end date, the same window DwhPullJob uses, so outcomes
    that land after the end date (late warehouse loads, late autofails) still reach the final arm parameters.

    The job runs on its interval whether or not new outcomes arrived. Checking for new outcomes would read the
    experiment's draws on every scheduler poll, which costs more than an update once per interval, and an update
    with nothing new leaves the arms unchanged.
    """

    name: ClassVar[str] = "bandit_update"
    backoff: Backoff

    def next_run_at(self, now: ColumnElement[datetime]) -> ColumnElement[datetime | None]:
        experiment = tables.Experiment
        # The interval is per experiment, so build it in SQL from the experiment's column.
        interval = func.make_interval(0, 0, 0, 0, experiment.update_frequency_hours)
        return case(
            (
                and_(
                    experiment.experiment_type.in_([t for t in ExperimentsType if t.is_bandit()]),
                    experiment.update_frequency_hours.is_not(None),
                ),
                next_run_in_window(
                    now,
                    interval,
                    opens=experiment.start_date,
                    closes=experiment.end_date + LATE_OUTCOME_GRACE,
                ),
            ),
            else_=None,
        )

    def execute(self, session: Session, experiment: tables.Experiment) -> JobResult:
        """Apply the experiment's new outcomes to its arms in the caller's transaction."""
        num_outcomes = update_bandit_arms_impl(session, experiment)
        logger.info(f"Updated {len(experiment.arms)} arms from {num_outcomes} outcomes.")
        sentry_sdk.metrics.count("worker.bandit_update.outcomes", num_outcomes)
        return JobSucceeded()
