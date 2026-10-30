"""Periodic, best-effort experiment snapshots."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import ClassVar

from sqlalchemy import ColumnElement, false, func, select
from sqlalchemy.orm import Session

from xngin.apiserver.routers.common_enums import ExperimentState
from xngin.apiserver.sqla import tables
from xngin.apiserver.worker.jobs.base import (
    Backoff,
    JobFailed,
    JobResult,
    JobSucceeded,
    next_run_in_window,
)
from xngin.apiserver.worker.jobs.snapshot_analysis import capture_snapshot
from xngin.stats.stats_errors import StatsNoParticipantsError

# Snapshots run from this long before an experiment starts until this long after it ends.
SNAPSHOT_MARGIN = timedelta(days=1)


@dataclass(slots=True)
class SnapshotJob:
    """Record each analysis attempt as a snapshot, whether the analysis succeeds or fails."""

    name: ClassVar[str] = "snapshot"
    interval: timedelta
    dwh_timeout: float
    backoff: Backoff

    def needs_refresh(self) -> ColumnElement[bool]:
        return false()

    def compute_next_run_at(
        self, session: Session, experiment: tables.Experiment, status: tables.JobStatus, now: datetime
    ) -> datetime | None:
        if experiment.state != ExperimentState.COMMITTED:
            return None
        last_success_at = status.last_success_at
        if last_success_at is None:
            # The job has never succeeded, but the experiment may have successful snapshots from before the worker
            # tracked this job, so a new deployment doesn't snapshot every experiment at once. This fallback can be
            # removed once the worker is fully deployed.
            last_success_at = session.scalar(
                select(func.max(tables.Snapshot.updated_at)).where(
                    tables.Snapshot.experiment_id == experiment.id, tables.Snapshot.status == "success"
                )
            )
        return next_run_in_window(
            last_success_at,
            now,
            self.interval,
            opens=experiment.start_date - SNAPSHOT_MARGIN,
            closes=experiment.end_date + SNAPSHOT_MARGIN,
        )

    def execute(self, session: Session, experiment: tables.Experiment) -> JobResult:
        capture = capture_snapshot(session, experiment, dwh_timeout=self.dwh_timeout)
        # A new experiment has no participants yet. The failed snapshot still tells users why there is no data,
        # but the job did what it could, so it succeeds and waits a full interval rather than backing off.
        if capture.snapshot.status == "failed" and not isinstance(capture.error, StatsNoParticipantsError):
            return JobFailed(error=capture.snapshot.message or "The snapshot failed without a message.")
        return JobSucceeded()
