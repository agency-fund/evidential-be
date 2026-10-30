"""Attempts: one experiment, one transaction, and at most one job, each on one of the executor's threads."""

import time
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from itertools import starmap
from typing import assert_never

import sentry_sdk
from loguru import logger
from sqlalchemy import ColumnElement, DateTime, Subquery, and_, case, func, literal, select, union_all
from sqlalchemy.orm import Session

from xngin.apiserver import database
from xngin.apiserver.routers.common_enums import ExperimentState
from xngin.apiserver.sqla import tables
from xngin.apiserver.worker.jobs.base import Job, JobDeferred, JobFailed, JobResult, JobSucceeded


@dataclass(slots=True, frozen=True)
class FinishedAttempt:
    # The experiment the attempt was given.
    experiment_id: str
    # The status of the completed attempt. The scheduler skips experiments whose attempt aborted or could not claim
    # them so that they don't get re-run immediately.
    result: AttemptResult


class AttemptResult(StrEnum):
    # A job ran, and its success or its failure and backoff were committed to its status row.
    JOB_RAN = "job_ran"
    # The experiment was claimed, but no job did any work: either no job was due anymore, because another attempt
    # ran it or its data changed since the look, or the due job deferred itself.
    NOTHING_READY = "nothing_ready"
    # The attempt failed outside the job's own error handling and rolled back, recording nothing: an operational
    # failure, such as a lost connection (even during the job) or a failed commit, or a bug in the worker or in a
    # job's next_run_at. A failing Job.execute is not an abort; it ends as JOB_RAN with its error recorded.
    ABORTED = "aborted"
    # The experiment could not be claimed: another process holds it, a delete is in progress, or it no longer
    # exists.
    NOT_CLAIMED = "not_claimed"


def attempt_experiment(jobs: Sequence[Job], experiment_id: str) -> FinishedAttempt:
    """Claim the experiment and run at most one of its due jobs, in one transaction.

    This is run on one of the AttemptExecutor's worker threads.

    Interrupted attempts roll back and return ABORTED.
    """
    try:
        with logger.contextualize(experiment_id=experiment_id), database.get_session() as session, session.begin():
            now = database_now(session)
            experiment = claim_experiment(session, experiment_id)
            if experiment is None:
                return FinishedAttempt(experiment_id, AttemptResult.NOT_CLAIMED)
            # The look that handed out this experiment read its run times before the claim, so check them again.
            run_time = _select_best_next_job(experiment_run_times(session, jobs, experiment_id, now), now)
            if run_time is None:
                return FinishedAttempt(experiment_id, AttemptResult.NOTHING_READY)
            job = next(job for job in jobs if job.name == run_time.job_name)
            status = session.get(tables.JobStatus, (experiment_id, job.name))
            # A job's first run for an experiment has no status row yet, so we create it here. That can come long after
            # the experiment is created: autofail first runs when the experiment's first draw expires.
            if status is None:
                status = tables.JobStatus(experiment_id=experiment_id, job_name=job.name, consecutive_failures=0)
                session.add(status)
            sentry_sdk.metrics.count("worker.jobs.started", 1, attributes={"job": job.name})
            started = time.monotonic()
            result = _execute_job(session, job, experiment)
            finished_at = database_now(session)
            _record_run(job, status, result, started_at=now, finished_at=finished_at)
    except Exception as exc:
        logger.opt(exception=exc).error("Worker attempt aborted.")
        sentry_sdk.capture_exception(exc)
        return FinishedAttempt(experiment_id, AttemptResult.ABORTED)
    # The run has committed, so a failure to report its metrics must not turn it into an abort.
    try:
        _report_job_metrics(job, result, time.monotonic() - started, now, run_time.run_at)
    except Exception as exc:
        logger.opt(exception=exc).warning("Reporting job metrics failed.")
    if isinstance(result, JobDeferred):
        return FinishedAttempt(experiment_id, AttemptResult.NOTHING_READY)
    return FinishedAttempt(experiment_id, AttemptResult.JOB_RAN)


def claim_experiment(session: Session, experiment_id: str) -> tables.Experiment | None:
    """Try to claim the experiment for this transaction, without waiting, and return it if the session got the locks.

    Claiming takes two locks with unrelated purposes:

    - A KEY SHARE row-level lock on the experiment, skipped if already locked FOR UPDATE. This ensures the experiment
      doesn't get deleted while we're working on it, but doesn't block the API server from updating experiment
      fields. When there is a concurrent delete of the experiment, either the delete will wait, or this will fail to
      claim the lock.
    - A transaction-scoped advisory lock, on the experiment_id, which ensures only one worker can claim the
      experiment at a time.

    Returning the experiment means this transaction holds both locks until it ends. The experiment exists and
    cannot be deleted, and no other attempt can claim it.

    Returning None means the caller should end the transaction without touching the experiment's rows. There are
    three possible reasons: another connection holds the claim, a delete of the experiment is in progress, or the
    experiment no longer exists.
    """
    # Loading the whole row here lets the caller pass it to every job without reloading it.
    experiment = session.scalar(
        select(tables.Experiment)
        .where(tables.Experiment.id == experiment_id)
        .with_for_update(read=True, key_share=True, skip_locked=True)
    )
    if experiment is None or not session.scalar(select(func.pg_try_advisory_xact_lock(_claim_lock_key(experiment_id)))):
        return None
    return experiment


def _claim_lock_key(experiment_id: str) -> ColumnElement[int]:
    """The 64-bit PostgreSQL advisory lock key that claims an experiment for one transaction at a time."""
    return func.hashtextextended(literal("experiment_id:") + experiment_id, 0)


def run_times(jobs: Sequence[Job], now: ColumnElement[datetime]) -> Subquery:
    """Return a subquery of when each job should next run for each committed experiment.

    Each job supplies its own schedule as a SQL expression, Job.next_run_at, evaluated at now. This combines them into
    one query with a row per committed experiment and job, so that a single read finds due work across every job. The
    backoff is applied here, once, rather than by each job: run_at is the job's next_run_at, delayed to the status
    row's retry_not_before when that is later, or NULL when the job has no next run.

    Columns returned: experiment_id, job_name, run_at, and last_run_at, which callers use to give the least recently
    run work priority.
    """
    status = tables.JobStatus
    # Each job's expression is computed once per experiment by its branch of the union. Postgres doesn't share
    # repeated subexpressions, so the delay is applied above the union rather than around each expression.
    jobs_run_at = union_all(
        *(
            select(
                tables.Experiment.id.label("experiment_id"),
                literal(job.name).label("job_name"),
                job.next_run_at(now).label("next_run_at"),
                status.retry_not_before,
                status.last_run_at,
            )
            .select_from(tables.Experiment)
            .outerjoin(status, and_(status.experiment_id == tables.Experiment.id, status.job_name == job.name))
            .where(tables.Experiment.state == ExperimentState.COMMITTED)
            for job in jobs
        )
    ).subquery()
    return select(
        jobs_run_at.c.experiment_id,
        jobs_run_at.c.job_name,
        # greatest() ignores NULLs, so it would turn a missing run time into the retry time.
        case(
            (jobs_run_at.c.next_run_at.is_(None), None),
            else_=func.greatest(jobs_run_at.c.next_run_at, jobs_run_at.c.retry_not_before),
        ).label("run_at"),
        jobs_run_at.c.last_run_at,
    ).subquery("run_times")


def experiment_run_times(session: Session, jobs: Sequence[Job], experiment_id: str, now: datetime) -> list[RunTime]:
    """Return each job's run time for one experiment, evaluated at now, or nothing if it isn't committed."""
    times = run_times(jobs, literal(now, DateTime(timezone=True)))
    query = select(times.c.job_name, times.c.run_at, times.c.last_run_at).where(times.c.experiment_id == experiment_id)
    return list(starmap(RunTime, session.execute(query)))


def database_now(session: Session) -> datetime:
    # now() is frozen at transaction start, including time spent reading a warehouse.
    value: datetime = session.execute(select(func.clock_timestamp())).scalar_one()
    return value


@dataclass(slots=True, frozen=True)
class RunTime:
    job_name: str
    run_at: datetime | None
    last_run_at: datetime | None


def _select_best_next_job(candidates: Iterable[RunTime], now: datetime) -> RunTime | None:
    """Return the run time of the job to run next, or None if no job's run time has arrived.

    Prefer a job that has never run, then the one that ran least recently, so one job's backlog, such as
    autofail's, cannot starve the others. Ties go to the job name.
    """
    due = [run_time for run_time in candidates if run_time.run_at is not None and run_time.run_at <= now]

    def priority(run_time: RunTime) -> tuple[bool, datetime, str]:
        return (run_time.last_run_at is not None, run_time.last_run_at or now, run_time.job_name)

    return min(due, key=priority, default=None)


def _execute_job(session: Session, job: Job, experiment: tables.Experiment) -> JobResult:
    try:
        with session.begin_nested():
            return job.execute(session, experiment)
    except Exception as exc:
        # An invalidated connection cannot record a failure or reconnect.
        if not session.is_active or session.connection().invalidated:
            raise
        logger.opt(exception=exc).error("Worker job failed.")
        sentry_sdk.capture_exception(exc)
        return JobFailed(error="".join(traceback.format_exception_only(exc)).strip())


def _record_run(
    job: Job, status: tables.JobStatus, result: JobResult, *, started_at: datetime, finished_at: datetime
) -> None:
    """Record a run's result in the job's status row."""
    status.last_run_at = started_at
    match result:
        case JobFailed(error=error):
            status.consecutive_failures += 1
            # Fit job_statuses.last_error, whether the error came from an exception or a job's own JobFailed.
            status.last_error = error[:2048]
            status.retry_not_before = finished_at + job.backoff.delay_after(status.consecutive_failures)
        case JobSucceeded():
            status.last_success_at = finished_at
            status.consecutive_failures = 0
            status.last_error = None
            status.retry_not_before = None
        case JobDeferred(retry_after=retry_after):
            status.retry_not_before = finished_at + retry_after
        case _:
            assert_never(result)


def _report_job_metrics(
    job: Job, result: JobResult, duration: float, started_at: datetime, scheduled_at: datetime | None
) -> None:
    attributes = {"job": job.name}
    match result:
        case JobSucceeded():
            outcome = "succeeded"
        case JobFailed():
            outcome = "failed"
        case JobDeferred():
            outcome = "deferred"
        case _:
            assert_never(result)
    sentry_sdk.metrics.count(f"worker.jobs.{outcome}", 1, attributes=attributes)
    sentry_sdk.metrics.distribution("worker.jobs.duration", duration, unit="second", attributes=attributes)
    if scheduled_at is not None:
        sentry_sdk.metrics.distribution(
            "worker.jobs.lag",
            max(0.0, (started_at - scheduled_at).total_seconds()),
            unit="second",
            attributes=attributes,
        )
