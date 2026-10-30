"""Job contracts and scheduling policies independent of worker orchestration."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import ClassVar, Protocol

from sqlalchemy import ColumnElement
from sqlalchemy.orm import Session

from xngin.apiserver.sqla import tables


@dataclass(slots=True, frozen=True)
class Backoff:
    """Delays after the first, second, third, and later consecutive failures."""

    delays: tuple[timedelta, ...]

    @classmethod
    def constant(cls, delay: timedelta) -> Backoff:
        """Wait the same delay after every failure, however many in a row."""
        return cls(delays=(delay,))

    def delay_after(self, failures: int) -> timedelta:
        return self.delays[min(max(failures - 1, 0), len(self.delays) - 1)]


@dataclass(slots=True, frozen=True)
class JobSucceeded:
    """The run succeeded. The worker records it as the last success and clears the failure count and error.

    The next run is scheduled from compute_next_run_at.
    """


@dataclass(slots=True, frozen=True)
class JobFailed:
    """The run failed without raising, so its writes are kept, unlike when Job.execute raises.

    The worker records the error, increments the failure count, and holds off the next run until the job's
    backoff delay for that many consecutive failures has passed.
    """

    error: str

    def __post_init__(self) -> None:
        if not self.error:
            raise ValueError("JobFailed needs an error message.")


@dataclass(slots=True, frozen=True)
class JobDeferred:
    """The run found nothing it could do yet, for a reason that should pass soon.

    Use it when the job was due but could make no progress, such as autofail finding every overdue draw locked by
    the API. The worker records neither a success nor a failure, and holds off the next run until retry_after has
    passed, whatever compute_next_run_at returns. The delay schedules no run on its own: if compute_next_run_at
    returns None, there is no next run.
    """

    retry_after: timedelta


# What Job.execute reports about one run. The worker records it in the job's status row.
type JobResult = JobSucceeded | JobFailed | JobDeferred


class Job(Protocol):
    # Names persist in job_statuses; renaming a job requires migrating those rows.
    name: ClassVar[str]
    backoff: Backoff

    def needs_refresh(self) -> ColumnElement[bool]:
        """Return a condition on Experiment and this job's JobStatus row that holds when its stored run time is stale.

        The refresh already recomputes run times for experiments whose updated_at changed. This condition covers
        the other rows compute_next_run_at reads, which change without touching the experiment row: for autofail,
        a newly inserted draw. Return false() when compute_next_run_at depends only on the experiment row and the
        job's own status.

        When the condition holds, compute_next_run_at must return a time; otherwise the stored run time stays as
        it was and the refresh revisits the experiment on every pass.
        """
        ...

    def compute_next_run_at(
        self, session: Session, experiment: tables.Experiment, status: tables.JobStatus, now: datetime
    ) -> datetime | None:
        """Return the time at which the job should run next, or None to leave it unscheduled until the next refresh
        pass.

        The worker stores the return value in JobStatus.next_run_at.

        Return None when there is no work for the job to do. This does not stop the job for good: the refresh calls
        this method again when the experiment changes or when needs_refresh holds.

        Return a datetime when you know that it would be useful to run the job again immediately (a time now or in
        the past), or when you want to ensure it runs no earlier than some time in the future. For example, a periodic
        job that runs every three hours would return now + 3 hours.

        Backoff is handled by the worker.
        """
        ...

    def execute(self, session: Session, experiment: tables.Experiment) -> JobResult:
        """Do one unit of the job's work for the experiment, in the caller's transaction.

        The experiment is the row the caller claimed. Its relationships load lazily when the job first reads them.

        To fail, raise any exception, or return JobFailed. A raised exception rolls back the job's writes, is
        logged and sent to Sentry, and its type and message are recorded as the error. A returned JobFailed keeps
        the job's writes, for jobs whose failure produces a record of its own, such as a failed snapshot. Either
        way the failure counts toward the job's backoff.

        Do not commit or roll back the caller's transaction.
        """
        ...


def next_run_in_window(
    last_success_at: datetime | None, now: datetime, interval: timedelta, *, opens: datetime, closes: datetime
) -> datetime | None:
    """Return when a job that runs every interval between opens and closes should next run.

    A job without a last_success_at is due immediately, but no earlier than opens. Returns None once no run fits
    before closes.
    """
    if closes < now:
        return None
    run_at = last_success_at + interval if last_success_at else now
    run_at = max(run_at, opens)
    return run_at if run_at <= closes else None
