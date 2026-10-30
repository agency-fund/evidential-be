"""Attempts: one experiment, one transaction, and at most one job, each on one of the executor's threads."""

import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import assert_never

import sentry_sdk
from loguru import logger
from sqlalchemy import ColumnElement, func, literal, select
from sqlalchemy.orm import Session

from xngin.apiserver import database
from xngin.apiserver.sqla import tables
from xngin.apiserver.worker.jobs.base import Job, JobDeferred, JobFailed, JobResult, JobSucceeded


@dataclass(slots=True, frozen=True)
class FinishedAttempt:
    # The experiment the attempt was given. The scheduler cools down experiments whose attempt aborted or could not
    # claim them, so the next look moves on to other due experiments.
    experiment_id: str
    result: AttemptResult


class AttemptResult(StrEnum):
    # A job ran, and its success or its failure and backoff were committed to its status row.
    JOB_RAN = "job_ran"
    # The experiment was claimed and its run times refreshed, but no job did any work: either no job was ready,
    # because the stored run time was stale, or the ready job deferred itself.
    NOTHING_READY = "nothing_ready"
    # The attempt failed outside the job's own error handling and rolled back, recording nothing: an operational
    # failure, such as a lost connection (even during the job) or a failed commit, or a bug in the worker or in a
    # job's compute_next_run_at. A failing Job.execute is not an abort; it ends as JOB_RAN with its error recorded.
    ABORTED = "aborted"
    # The experiment could not be claimed: another process holds it, a delete is in progress, or it no longer
    # exists.
    NOT_CLAIMED = "not_claimed"


def attempt_experiment(jobs: Sequence[Job], experiment_id: str) -> FinishedAttempt:
    """Claim the experiment, refresh its run times, and run at most one of its ready jobs, in one transaction.

    This is run on one of the AttemptExecutor's worker threads.

    Interrupted attempts roll back and return ABORTED.
    """
    try:
        with logger.contextualize(experiment_id=experiment_id), database.get_session() as session, session.begin():
            now = database_now(session)
            experiment = claim_experiment(session, experiment_id)
            if experiment is None:
                return FinishedAttempt(experiment_id, AttemptResult.NOT_CLAIMED)
            statuses = refresh_run_times(session, jobs, experiment, now)
            job = _select_best_next_job(jobs, statuses, now)
            if job is None:
                return FinishedAttempt(experiment_id, AttemptResult.NOTHING_READY)
            status = statuses[job.name]
            scheduled_at = status.next_run_at
            sentry_sdk.metrics.count("worker.jobs.started", 1, attributes={"job": job.name})
            started = time.monotonic()
            result = _execute_job(session, job, experiment)
            finished_at = database_now(session)
            _record_run(session, job, experiment, status, result, started_at=now, finished_at=finished_at)
    except Exception as exc:
        logger.opt(exception=exc).error("Worker attempt aborted.")
        sentry_sdk.capture_exception(exc)
        return FinishedAttempt(experiment_id, AttemptResult.ABORTED)
    # The run has committed, so a failure to report its metrics must not turn it into an abort.
    try:
        _report_job_metrics(job, result, time.monotonic() - started, now, scheduled_at)
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
    cannot be deleted, and no other attempt or refresh transaction can claim it.

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


def refresh_run_times(
    session: Session, jobs: Sequence[Job], experiment: tables.Experiment, now: datetime
) -> dict[str, tables.JobStatus]:
    """Recompute every job's run time for the experiment, creating missing status rows, while holding both locks.

    Each job's compute_next_run_at reads the experiment row, the job's own status row, and for some jobs the
    experiment's draws or snapshots, all as committed when the claim was taken.
    """
    statuses = {
        row.job_name: row
        for row in session.scalars(select(tables.JobStatus).where(tables.JobStatus.experiment_id == experiment.id))
    }
    for job in jobs:
        if job.name not in statuses:
            statuses[job.name] = tables.JobStatus(experiment_id=experiment.id, job_name=job.name)
            session.add(statuses[job.name])
            session.flush()
        statuses[job.name].next_run_at = _next_run_at_after_backoff(session, job, experiment, statuses[job.name], now)
        # The experiment row was read under the claim, so this is the exact version these run times reflect. An
        # edit that commits later changes updated_at, and the next refresh then sees the mismatch.
        statuses[job.name].experiment_updated_at = experiment.updated_at
    return statuses


def _next_run_at_after_backoff(
    session: Session, job: Job, experiment: tables.Experiment, status: tables.JobStatus, now: datetime
) -> datetime | None:
    run_at = job.compute_next_run_at(session, experiment, status, now)
    if run_at is not None and status.retry_not_before is not None:
        run_at = max(run_at, status.retry_not_before)
    return run_at


def database_now(session: Session) -> datetime:
    # now() is frozen at transaction start, including time spent reading a warehouse.
    value: datetime = session.execute(select(func.clock_timestamp())).scalar_one()
    return value


def _select_best_next_job(jobs: Sequence[Job], statuses: Mapping[str, tables.JobStatus], now: datetime) -> Job | None:
    """Return the ready job to run next, or None if no job's run time has arrived.

    Prefer a job that has never run, then the one that ran least recently, so one job's backlog, such as
    autofail's, cannot starve the others. Ties go to the job name.
    """
    ready = [job for job in jobs if (run_at := statuses[job.name].next_run_at) is not None and run_at <= now]

    def priority(job: Job) -> tuple[bool, datetime, str]:
        last_run_at = statuses[job.name].last_run_at
        return (last_run_at is not None, last_run_at or now, job.name)

    return min(ready, key=priority, default=None)


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
    session: Session,
    job: Job,
    experiment: tables.Experiment,
    status: tables.JobStatus,
    result: JobResult,
    *,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    """Record a run's result in the job's status row and schedule its next run."""
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
    status.next_run_at = _next_run_at_after_backoff(session, job, experiment, status, finished_at)


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
