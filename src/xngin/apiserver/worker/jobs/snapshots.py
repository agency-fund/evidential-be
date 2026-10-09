"""Periodic, best-effort experiment snapshots."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import ClassVar

from sqlalchemy import ColumnElement, false, func, select
from sqlalchemy.orm import Session

from xngin.apiserver.snapshots.snapshotter import capture_snapshot
from xngin.apiserver.sqla import tables
from xngin.apiserver.worker.jobs.base import (
    Backoff,
    JobFailed,
    JobResult,
    JobSucceeded,
    periodic_next_run_at,
)


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
        return periodic_next_run_at(experiment, last_success_at, now, self.interval)

    def execute(self, session: Session, experiment_id: str) -> JobResult:
        snapshot = capture_snapshot(session, experiment_id, dwh_timeout=self.dwh_timeout)
        if snapshot.status == "failed":
            return JobFailed(error=snapshot.message or "The snapshot failed without a message.")
        return JobSucceeded()
