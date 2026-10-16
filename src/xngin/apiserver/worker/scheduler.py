"""The scheduler thread: decide when to look for work, find the due experiments, and hand them to attempts."""

import math
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from threading import Event
from typing import assert_never

import sentry_sdk
from loguru import logger
from sqlalchemy import and_, exists, func, or_, select
from sqlalchemy.orm import Session

from xngin.apiserver import database
from xngin.apiserver.sqla import tables
from xngin.apiserver.worker.attempts import (
    AttemptExecutor,
    AttemptResult,
    Clock,
    FinishedAttempt,
    claim_experiment,
    database_now,
    refresh_run_times,
)
from xngin.apiserver.worker.jobs.base import Job

# A look refreshes at most this many stale experiments before reading the schedule, so a pass over many of them,
# such as the first after a deployment, delays the due jobs by a bounded amount. The rest are refreshed by later looks.
_STALE_EXPERIMENTS_PER_LOOK = 500


def run_worker(jobs: Sequence[Job], stop: Event, clock: Clock, pacer: Pacer, executor: AttemptExecutor) -> None:
    """Loop looking for due work and running it, until stop is set or the pacer says the worker is done.

    Each pass of the loop reports the attempts that finished since the last pass to the pacer. If the pacer's next
    look has arrived and a slot is free, it looks: it refreshes the schedule, reads which experiments are due, and
    gives one to each free slot. Then it sleeps until the next look, until an attempt finishes, or until stop is
    set. The pacer makes every timing decision; this loop just respects them.

    The executor bounds how many attempts run at once, each on its own thread with its own database session, so no
    session crosses threads. When stop is set or the exit limit is reached, no new attempts start and this returns;
    leaving the executor's context then waits for the running attempts to commit or roll back.
    """
    while not stop.is_set():
        pacer.record_finished(executor.collect_finished(), clock.monotonic())
        if pacer.should_exit():
            stop.set()
            break
        if pacer.look_due(clock.monotonic()) and not executor.is_full():
            skip = (*pacer.cooling_down(), *executor.in_progress())
            schedule = _look(jobs, stop, skip)
            pacer.looked(schedule, clock.monotonic())
            executor.start(jobs, stop, schedule.due)
        executor.wait(stop, pacer.wake_at(can_start=not executor.is_full()))


def _look(jobs: Sequence[Job], stop: Event, skip: Collection[str]) -> Schedule:
    """Look for work: bring stored run times up to date and read the schedule, skipping over the experiments in skip.

    If we have fail to look for work, assume it is transient, and returns that we have no work to do. The next polling
    interval might succeed!
    """
    try:
        failed = refresh_stale_experiments(jobs, stop, skip)
        due = due_experiment_ids(jobs, (*skip, *failed))
        next_run_in = _seconds_until_next_run(jobs)
    except Exception as exc:
        logger.opt(exception=exc).error("Refreshing the worker's schedule failed; retrying.")
        sentry_sdk.capture_exception(exc)
        sentry_sdk.metrics.count("worker.looks.failed", 1)
        return Schedule()
    sentry_sdk.metrics.count("worker.looks.succeeded", 1)
    return Schedule(due, failed, next_run_in)


@dataclass(slots=True, frozen=True)
class Schedule:
    """What a look found: the experiments due now and how far off the next run is.

    A look that failed finds nothing and knows of no next run.
    """

    # The experiments with a job due, in the order to attempt them.
    due: tuple[str, ...] = ()
    # The experiments whose refresh failed.
    failed: tuple[str, ...] = ()
    # Seconds until the earliest future run time, or None if none is known.
    next_run_in: float | None = None


@dataclass(slots=True)
class Pacer:
    """Decide when the scheduler looks for work, which experiments it passes over, and when it wakes.

    The timing policy is a single deadline, next_look, that each event moves: a finished attempt brings it to now,
    and each look sets it a polling interval away, or sooner when a known run time comes due. Pacer holds no
    threads, sessions, or clocks: every method takes the current Clock.monotonic() value, and it is the only place
    that does arithmetic with those values.
    """

    poll_interval: float
    exit_after_jobs: int | None = None
    next_look: float = 0.0
    # Experiments whose attempt aborted or could not claim them, or whose refresh failed, mapped to when they may
    # be attempted again.
    cooldowns: dict[str, float] = field(default_factory=dict)
    jobs_executed: int = 0

    def record_finished(self, finished: Iterable[FinishedAttempt], now: float) -> None:
        """Account for finished attempts: count executed jobs, cool down failing experiments, and look again at once.

        Any finished attempt frees a slot, and the look that filled it may have found more due experiments than
        slots, so the worker looks again at once, like a consumer draining a queue. A job can also make its own
        experiment due again: an autofail batch that leaves more overdue draws is due as soon as it commits, so a
        backlog drains batch after batch instead of one per polling interval. Once a look finds nothing, the
        worker waits for the next scheduled look.

        Looking again cannot spin, because every result leaves its experiment out of the next look: a run or a
        refresh moves its run time, and an abort or a failed claim cools it down. Cooling down an aborted
        experiment also keeps a bug in a job's compute_next_run_at, or an experiment held by another process,
        from starving the rest.
        """
        self.cooldowns = {key: until for key, until in self.cooldowns.items() if until > now}
        for attempt in finished:
            match attempt.result:
                case AttemptResult.JOB_RAN:
                    self.jobs_executed += 1
                case AttemptResult.NOTHING_READY:
                    pass
                case AttemptResult.ABORTED | AttemptResult.NOT_CLAIMED:
                    self.cool_down((attempt.experiment_id,), now)
                case _:
                    assert_never(attempt.result)
            self.next_look = now

    def should_exit(self) -> bool:
        return self.exit_after_jobs is not None and self.jobs_executed >= self.exit_after_jobs

    def look_due(self, now: float) -> bool:
        return now >= self.next_look

    def cooling_down(self) -> tuple[str, ...]:
        return tuple(self.cooldowns)

    def cool_down(self, experiment_ids: Iterable[str], now: float) -> None:
        """Pass over these experiments, in both refreshes and attempts, for one polling interval."""
        for experiment_id in experiment_ids:
            self.cooldowns[experiment_id] = now + self.poll_interval

    def looked(self, schedule: Schedule, now: float) -> None:
        """Cool down the experiments whose refresh failed, and look again after one polling interval, or sooner if
        the schedule has the next run time due earlier.

        The polling interval bounds every wait, so a wrong expectation, such as from a step in the database server's
        clock, corrects itself within an interval.
        """
        self.cool_down(schedule.failed, now)
        next_run_in = math.inf if schedule.next_run_in is None else schedule.next_run_in
        self.next_look = now + min(self.poll_interval, next_run_in)

    def wake_at(self, *, can_start: bool) -> float:
        """Return when to wake if no attempt finishes first.

        With every slot occupied, only a completion can lead to new work; waking for a past-due look would loop
        without being able to start anything.
        """
        return self.next_look if can_start else math.inf


def refresh_stale_experiments(
    jobs: Sequence[Job],
    stop: Event | None = None,
    excluded: Collection[str] = (),
    *,
    limit: int = _STALE_EXPERIMENTS_PER_LOOK,
) -> tuple[str, ...]:
    """Create missing status rows and refresh run times for experiments whose run times may be stale.

    Returns the experiments whose refresh failed. Each one would fail again on the next pass, so the scheduler
    excludes them for a while.

    Each stale experiment gets its own short transaction so a pass does not accumulate locks, and busy experiments
    are passed over until a later pass. We cap the number of experiments we refresh at 500 per run, mitigating the risk
    of stampedes after a deployment.
    """
    failed: list[str] = []
    with database.get_session() as session:
        ids = _stale_experiment_ids(session, jobs, excluded, limit=limit)
    for experiment_id in ids:
        if stop is not None and stop.is_set():
            break
        try:
            with database.get_session() as session, session.begin():
                if (experiment := claim_experiment(session, experiment_id)) is not None:
                    refresh_run_times(session, jobs, experiment, database_now(session))
        except Exception as exc:
            # The refresh supervises independent experiments: one bad configuration or transient lock failure
            # must not prevent the rest of the pass from being scheduled.
            logger.bind(experiment_id=experiment_id).opt(exception=exc).error("Experiment refresh failed.")
            sentry_sdk.capture_exception(exc)
            failed.append(experiment_id)
    return tuple(failed)


def _stale_experiment_ids(
    session: Session, jobs: Sequence[Job], excluded: Collection[str] = (), *, limit: int | None = None
) -> tuple[str, ...]:
    """Return up to limit of the experiments, other than excluded ones, whose stored run times may be out of date.

    That is any experiment missing a status row for some job, edited since its run times were computed (its
    updated_at no longer matches the status rows' experiment_updated_at), or matching some job's needs_refresh,
    which covers rows other than the experiment that the job schedules from, such as its draws. After a refresh,
    an experiment matches again only when one of these changes again.
    """
    status = tables.JobStatus
    experiment = tables.Experiment
    for_experiment = status.experiment_id == experiment.id
    missing = [~exists().where(for_experiment, status.job_name == job.name) for job in jobs]
    edited = exists().where(
        for_experiment,
        status.job_name.in_([job.name for job in jobs]),
        status.experiment_updated_at.is_distinct_from(experiment.updated_at),
    )
    woken = [exists().where(for_experiment, and_(status.job_name == job.name, job.needs_refresh())) for job in jobs]
    return tuple(
        session.scalars(
            select(experiment.id)
            .where(experiment.id.not_in(excluded), or_(*missing, edited, *woken))
            .order_by(experiment.id)
            .limit(limit)
        )
    )


def due_experiment_ids(
    jobs: Sequence[Job], excluded: Collection[str] = (), *, now: datetime | None = None
) -> tuple[str, ...]:
    """Return the experiments, other than excluded ones, with a job due to run, in the order to attempt them.

    Experiments have one JobStatus row per job. We order the rows such that each experiment is ordered by its most
    recently run job, prioritizing the least recently run job first. Every run moves its experiment to the back,
    so experiments take turns.
    """
    status = tables.JobStatus
    due = select(status.experiment_id).where(
        status.job_name.in_([job.name for job in jobs]),
        status.next_run_at <= (func.statement_timestamp() if now is None else now),
    )
    last_job_ran_at = func.max(status.last_run_at)
    with database.get_session() as session:
        return tuple(
            session.scalars(
                select(status.experiment_id)
                .where(status.experiment_id.in_(due), status.experiment_id.not_in(excluded))
                .group_by(status.experiment_id)
                .order_by(last_job_ran_at.asc().nulls_first(), status.experiment_id)
            )
        )


def _seconds_until_next_run(jobs: Sequence[Job]) -> float | None:
    """Return how many seconds until the earliest future run time, as measured by the database, or None if none.

    Returning a duration keeps database time inside the database; the scheduler adds it to its own monotonic clock.
    """
    # statement_timestamp() is fixed before the query reads any rows, so the result can only overstate the wait: the
    # scheduler may wake late by the query's latency, but never before the run time.
    next_run_at = func.min(tables.JobStatus.next_run_at)
    with database.get_session() as session:
        seconds = session.scalar(
            select(func.extract("epoch", next_run_at - func.statement_timestamp())).where(
                tables.JobStatus.job_name.in_([job.name for job in jobs]),
                tables.JobStatus.next_run_at > func.statement_timestamp(),
            )
        )
        return None if seconds is None else float(seconds)
