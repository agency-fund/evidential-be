"""Run the experiment maintenance service."""

import os
import signal
import sys
from datetime import timedelta
from threading import Event
from typing import Annotated

import sentry_sdk
import typer
from loguru import logger

from xngin.apiserver import customlogging, database
from xngin.apiserver.worker.executor import AttemptExecutor, SystemClock
from xngin.apiserver.worker.jobs.autofail import AutofailJob
from xngin.apiserver.worker.jobs.base import Backoff
from xngin.apiserver.worker.jobs.dwh_pull import DwhPullJob
from xngin.apiserver.worker.jobs.snapshots import SnapshotJob
from xngin.apiserver.worker.scheduler import Pacer, run_worker
from xngin.ops import sentry
from xngin.xsecrets import secretservice

app = typer.Typer(help="Runs experiment snapshots, autofail updates, and data warehouse pulls.")

# A warehouse that is briefly unavailable is retried within minutes. One that stays broken, such as with revoked
# credentials or a dropped table, settles at hourly pulls, since each failing pull holds the experiment's draw locks
# and a worker slot for up to the pull timeout, and leaves a thread stuck on the warehouse call.
DWH_PULL_BACKOFF = Backoff(
    delays=(
        timedelta(minutes=1),
        timedelta(minutes=2),
        timedelta(minutes=5),
        timedelta(minutes=15),
        timedelta(minutes=30),
        timedelta(hours=1),
    )
)
# Autofail failures are quick database errors or bugs that hold nothing external, so they keep retrying every 2
# minutes.
AUTOFAIL_BACKOFF = Backoff(delays=(timedelta(minutes=1), timedelta(minutes=2)))


def seconds(**kwargs: float) -> int:
    """Express a flag's default duration in the seconds the flag accepts, e.g. seconds(hours=3)."""
    return int(timedelta(**kwargs).total_seconds())


@app.command()
def run(
    poll_interval: Annotated[
        float,
        typer.Option(
            envvar="XNGIN_WORKER_POLL_INTERVAL", min=1, help="Maximum seconds between checks for new or changed work."
        ),
    ] = seconds(minutes=5),
    parallelism: Annotated[
        int,
        typer.Option(
            "--parallelism", "-j", envvar="XNGIN_WORKER_PARALLELISM", min=1, help="Maximum concurrent maintenance jobs."
        ),
    ] = 4,
    snapshot_interval: Annotated[
        float,
        typer.Option(
            envvar="XNGIN_WORKER_SNAPSHOT_INTERVAL",
            min=1,
            help="Seconds between successful snapshots. Failed snapshots retry hourly.",
        ),
    ] = seconds(hours=3),
    snapshot_dwh_timeout: Annotated[
        float,
        typer.Option(
            envvar="XNGIN_WORKER_SNAPSHOT_DWH_TIMEOUT", min=1, help="Maximum seconds for a snapshot's warehouse read."
        ),
    ] = 90,
    dwh_pull_interval: Annotated[
        float,
        typer.Option(
            envvar="XNGIN_WORKER_DWH_PULL_INTERVAL", min=1, help="Seconds between successful data warehouse pulls."
        ),
    ] = seconds(hours=4),
    dwh_pull_timeout: Annotated[
        float,
        typer.Option(
            envvar="XNGIN_WORKER_DWH_PULL_TIMEOUT", min=1, help="Maximum seconds for a pull's warehouse read."
        ),
    ] = 90,
    autofail_batch_size: Annotated[
        int,
        typer.Option(
            envvar="XNGIN_WORKER_AUTOFAIL_BATCH_SIZE", min=1, help="Maximum autofail outcomes per transaction."
        ),
    ] = 500,
    exit_after_jobs: Annotated[
        int,
        typer.Option(
            envvar="XNGIN_WORKER_EXIT_AFTER_JOBS",
            min=1,
            help="Exit after this many jobs finish, so a fresh process replaces this one.",
        ),
    ] = 1000,
) -> None:
    """Run until SIGTERM, SIGINT, or --exit-after-jobs finished jobs, then drain active jobs and exit."""
    customlogging.setup()
    sentry.setup()
    secretservice.setup()

    jobs = (
        # Snapshots retain their original behavior of running periodically, retrying failures hourly, and recording
        # successes and failures in a dedicated table, preserving existing UI functionality.
        SnapshotJob(
            interval=timedelta(seconds=snapshot_interval),
            dwh_timeout=snapshot_dwh_timeout,
            backoff=Backoff.constant(timedelta(hours=1)),
        ),
        DwhPullJob(
            interval=timedelta(seconds=dwh_pull_interval), dwh_timeout=dwh_pull_timeout, backoff=DWH_PULL_BACKOFF
        ),
        # AutofailJob computes its next run time from the experiment's pending draws.
        AutofailJob(batch_size=autofail_batch_size, backoff=AUTOFAIL_BACKOFF),
    )
    # Graceful shutdown: SIGINT and SIGTERM only set this event, so the worker stops starting new jobs
    # and drains active ones instead of dying mid-job.
    stop = Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    exit_code = 1
    try:
        clock = SystemClock()
        pacer = Pacer(poll_interval=poll_interval, clock=clock, exit_after_jobs=exit_after_jobs)
        # Each worker will get its own connection, plus one for scheduling. Leaving the executor's context drains
        # the running attempts, before the database is torn down.
        with database.setup(pool_size=parallelism + 1), AttemptExecutor(parallelism, clock) as executor:
            run_worker(jobs, stop, pacer, executor)
        exit_code = 0
    except BaseException as exc:
        # exit_now discards the exception, so report it here.
        logger.opt(exception=exc).error("Worker stopped unexpectedly.")
        sentry_sdk.capture_exception(exc)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        exit_now(exit_code)


def exit_now(exit_code: int) -> None:
    """Exit at once, killing threads left behind by abandoned warehouse calls.

    A warehouse call that outlives its timeout keeps running on a helper thread, possibly forever. A normal
    interpreter exit joins every executor thread and would wait for them, so flush what we can and exit without
    running exit handlers, even when an exception is propagating. Railway's ALWAYS restart policy
    (railway.worker.json) then starts a fresh process.
    """
    logger.info(f"Worker exiting with status {exit_code}.")
    sentry_sdk.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
