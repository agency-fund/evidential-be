"""The scheduler thread: decide when to look for work, find the due experiments, and hand them to attempts."""

import math
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from threading import Event
from typing import assert_never

import sentry_sdk
from loguru import logger
from sqlalchemy import DateTime, func, literal, select

from xngin.apiserver import database
from xngin.apiserver.worker.attempts import AttemptResult, FinishedAttempt, run_times
from xngin.apiserver.worker.executor import AttemptExecutor, Clock
from xngin.apiserver.worker.jobs.base import Job


def run_worker(jobs: Sequence[Job], stop: Event, pacer: Pacer, executor: AttemptExecutor) -> None:
    """Loop, looking for work and running it, until the stop event is set or Pacer tells us to exit.

    The pacer decides when to look for work, and which experiments to skip for now (last attempt
    aborted, or failed to claim). The executor runs each attempt on its own thread with its own database session.
    """
    while not stop.is_set():
        pacer.record_finished(executor.collect_finished())
        if pacer.should_exit():
            stop.set()
            break
        if pacer.should_look() and not executor.is_full():
            skip = (*pacer.skipped(), *executor.in_progress())
            schedule = _look(jobs, skip)
            pacer.looked(schedule)
            executor.start(jobs, stop, schedule.due)
        executor.wait(stop, pacer.wake_at(can_start=not executor.is_full()))


def _look(jobs: Sequence[Job], skip: Collection[str]) -> Schedule:
    """Look for work: read which experiments, other than those in skip, are due, and when the next run is.

    If we fail to look for work, assume it is transient, and return that we have no work to do. The next polling
    interval might succeed!
    """
    try:
        schedule = read_schedule(jobs, skip)
    except Exception as exc:
        logger.opt(exception=exc).error("Reading the worker's schedule failed; retrying.")
        sentry_sdk.capture_exception(exc)
        sentry_sdk.metrics.count("worker.looks.failed", 1)
        return Schedule()
    sentry_sdk.metrics.count("worker.looks.succeeded", 1)
    return schedule


@dataclass(slots=True, frozen=True)
class Schedule:
    """What a look found: the experiments due now and how far off the next run is.

    A look that failed finds nothing and knows of no next run.
    """

    # The experiments with a job due, in the order to attempt them.
    due: tuple[str, ...] = ()
    # Seconds until the earliest future run time, or None if none is known.
    next_run_in: float | None = None


class Pacer:
    """Tracks when the worker should next look for work, and which experiments it should skip for now.

    The Pacer lets the worker look for work once per polling interval, sooner when a known run time comes
    due, and again as soon as an attempt finishes. It skips an experiment for one polling interval after an attempt
    on it aborts or cannot claim it, so that one broken or contended experiment cannot crowd out the rest. Pacer also
    counts executed jobs, so the worker can exit after a set number.
    """

    def __init__(self, poll_interval: float, clock: Clock, exit_after_jobs: int | None = None) -> None:
        self.poll_interval = poll_interval
        self.clock = clock
        self.exit_after_jobs = exit_after_jobs
        self.next_look = 0.0
        # Experiments to skip, mapped to when they may be attempted again.
        self.skips: dict[str, float] = {}
        self.jobs_executed = 0

    def record_finished(self, finished: Iterable[FinishedAttempt]) -> None:
        """Count executed jobs, and skip experiments whose attempt aborted or could not claim them.

        If any attempt finished, the next look is due now, so outstanding work proceeds without waiting for the
        polling interval.
        """
        now = self.clock.monotonic()
        self.skips = {key: until for key, until in self.skips.items() if until > now}
        for attempt in finished:
            match attempt.result:
                case AttemptResult.JOB_RAN:
                    self.jobs_executed += 1
                case AttemptResult.NOTHING_READY:
                    pass
                case AttemptResult.ABORTED | AttemptResult.NOT_CLAIMED:
                    self.skips[attempt.experiment_id] = now + self.poll_interval
                case _:
                    assert_never(attempt.result)
            self.next_look = now

    def should_exit(self) -> bool:
        return self.exit_after_jobs is not None and self.jobs_executed >= self.exit_after_jobs

    def should_look(self) -> bool:
        return self.clock.monotonic() >= self.next_look

    def skipped(self) -> tuple[str, ...]:
        """Return the experiments the next look should skip."""
        return tuple(self.skips)

    def looked(self, schedule: Schedule) -> None:
        """Look again after one polling interval, or sooner when the schedule knows of an earlier run time.

        The polling interval caps every wait so that a wrong estimate, such as one caused by the database server's
        clock jumping, corrects itself within an interval.
        """
        next_run_in = math.inf if schedule.next_run_in is None else schedule.next_run_in
        self.next_look = self.clock.monotonic() + min(self.poll_interval, next_run_in)

    def wake_at(self, *, can_start: bool) -> float:
        """Return when the worker should wake if no attempt finishes first.

        When the worker can start no more attempts, only a finishing attempt can lead to new work. Waking for a look
        it could not act on would loop without doing anything.
        """
        return self.next_look if can_start else math.inf


def read_schedule(jobs: Sequence[Job], excluded: Collection[str] = (), *, now: datetime | None = None) -> Schedule:
    """Return the experiments, other than excluded ones, with a job due now, and how far off the next run is.

    Due experiments are ordered by their most recently run job, least recently run first. Every run moves its
    experiment to the back, so experiments take turns.

    The time until the next run is measured by the database, which keeps database time inside the database; the
    scheduler adds it to its own monotonic clock. statement_timestamp() is fixed before the query reads any rows, so
    the result can only overstate the wait: the scheduler may wake late by the query's latency, but never before
    the run time.
    """
    at = func.statement_timestamp() if now is None else literal(now, DateTime(timezone=True))
    times = run_times(jobs, at)
    query = (
        select(
            times.c.experiment_id,
            func.bool_or(times.c.run_at <= at).label("is_due"),
            func.extract("epoch", func.min(times.c.run_at).filter(times.c.run_at > at) - at).label("next_run_in"),
        )
        .group_by(times.c.experiment_id)
        # Planning nudge: filtering in WHERE instead would shrink the sort, but Postgres pushes WHERE conditions into
        # each branch of the union in run_times, where run_at expands to the job's whole expression, evaluating it
        # twice more.
        .having(func.bool_or(times.c.run_at.is_not(None)))
        .order_by(func.max(times.c.last_run_at).asc().nulls_first(), times.c.experiment_id)
    )
    with database.get_session() as session:
        rows = session.execute(query).all()
    due = tuple(row.experiment_id for row in rows if row.is_due and row.experiment_id not in excluded)
    future = [float(row.next_run_in) for row in rows if row.next_run_in is not None]
    return Schedule(due, min(future, default=None))
