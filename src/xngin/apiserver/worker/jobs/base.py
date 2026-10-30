"""Job contracts and scheduling policies independent of worker orchestration."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import ClassVar, Protocol

from sqlalchemy import ColumnElement, DateTime, SQLColumnExpression, and_, case, column, func, select, values
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

    The next run is scheduled from Job.next_run_at.
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
    passed, whatever Job.next_run_at returns. The delay schedules no run on its own: if Job.next_run_at
    returns NULL, there is no next run.
    """

    retry_after: timedelta


# What Job.execute reports about one run. The worker records it in the job's status row.
type JobResult = JobSucceeded | JobFailed | JobDeferred


class Job(Protocol):
    # Names persist in job_statuses; renaming a job requires migrating those rows.
    name: ClassVar[str]
    backoff: Backoff

    def next_run_at(self, now: ColumnElement[datetime]) -> ColumnElement[datetime | None]:
        """Return a SQL expression for when the job should next run for an experiment, or NULL when it has no work.

        The expression may refer to tables.Experiment and to tables.JobStatus, this job's status row for the
        experiment. The worker evaluates it only for committed experiments, so it need not check the state. The
        status row is outer-joined, so its columns are NULL until the job first runs. It may read other tables, such
        as draws, through correlated subqueries. now is the time the expression is evaluated at.

        Nothing about the result is stored. The worker evaluates the expression on every look for work, which happens
        at least once per polling interval and again after each attempt finishes, and once more under the
        experiment's claim before running the job. A time at or before now means the job is due. The worker delays the
        run to the status row's retry_not_before when that is later, so the expression need not consider backoff.

        NULL does not stop the job for good: it only means the job doesn't run while the expression stays NULL. As
        soon as the rows it reads change so that it gives a time, such as when an experiment is committed or a draw
        is inserted, the next look schedules the job.
        """
        ...

    def execute(self, session: Session, experiment: tables.Experiment) -> JobResult:
        """Do one unit of the job's work for the experiment, in the caller's transaction.

        The experiment is the row the caller claimed.

        To fail, raise any exception, or return JobFailed. A raised exception rolls back the job's writes, is
        logged and sent to Sentry, and its type and message are recorded as the error. A returned JobFailed keeps
        the job's writes, for jobs whose failure produces a record of its own, such as a failed snapshot. Either
        way the failure counts toward the job's backoff.

        Do not commit or roll back the caller's transaction.
        """
        ...


def next_run_in_window(
    now: ColumnElement[datetime],
    interval: timedelta,
    *,
    opens: SQLColumnExpression[datetime],
    closes: SQLColumnExpression[datetime],
) -> ColumnElement[datetime | None]:
    """Return SQL for when a job that runs every interval within a window should next run, or NULL if it shouldn't.

    The job's last success and last attempt are read from its JobStatus row. A job that has never succeeded is due
    when the window opens, or now if it is already open, and never once it has closed. After a success, the next
    run is due interval later, but no earlier than opens and no later than closes, so the window always ends with a
    run. A success at or after closes ends the runs.

    Once the window has closed, a run that came due inside it still gets one attempt, however late, if none has
    started since it came due. There are no retries after the close.
    """
    last_success_at = tables.JobStatus.last_success_at
    last_run_at = tables.JobStatus.last_run_at
    # Name the window's bounds and the next run after the last success once per row, so the CASE below reads them
    # instead of repeating their arithmetic. The next run is kept inside the window: one that would come before the
    # window opens waits for it to open, and one that would overshoot the close runs at the close, so the window
    # still gets a final run.
    window = values(
        column("opens", DateTime(timezone=True)),
        column("closes", DateTime(timezone=True)),
        column("after_success", DateTime(timezone=True)),
        name="run_window",
    ).data([(opens, closes, func.least(func.greatest(last_success_at + interval, opens), closes))])
    return (
        select(
            case(
                # The window closed before the job ever succeeded.
                (and_(last_success_at.is_(None), now > window.c.closes), None),
                # The job has never succeeded, so it's due now, or when the window opens if that's later.
                (last_success_at.is_(None), func.greatest(now, window.c.opens)),
                # The job already ran at the close, which is the window's final run.
                (last_success_at >= window.c.closes, None),
                # The window has closed, and an attempt started after the run came due without succeeding, since a
                # success would have moved last_success_at past it. Either the final run at the close was attempted
                # after the close and failed, or a run due inside the window failed and its backoff retry would come
                # after the close. Failed runs aren't retried once the window has closed.
                (and_(now > window.c.closes, last_run_at >= window.c.after_success), None),
                else_=window.c.after_success,
            ).label("run_at")
        )
        .select_from(window)
        .scalar_subquery()
    )
