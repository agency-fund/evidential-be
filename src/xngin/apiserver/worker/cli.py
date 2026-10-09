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
from xngin.apiserver.worker.attempts import AttemptExecutor, SystemClock
from xngin.apiserver.worker.jobs.base import Backoff
from xngin.apiserver.worker.jobs.snapshots import SnapshotJob
from xngin.apiserver.worker.scheduler import Pacer, run_worker
from xngin.ops import sentry
from xngin.xsecrets import secretservice

app = typer.Typer(help="Runs experiment snapshots.")


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
    )
    # Graceful shutdown: SIGINT and SIGTERM only set this event, so the worker stops starting new jobs
    # and drains active ones instead of dying mid-job.
    stop = Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        clock = SystemClock()
        # Each worker will get its own connection, plus one for scheduling. Leaving the executor's context drains
        # the running attempts, before the database is torn down.
        with database.setup(pool_size=parallelism + 1), AttemptExecutor(parallelism, clock) as executor:
            run_worker(jobs, stop, clock, Pacer(poll_interval=poll_interval, exit_after_jobs=exit_after_jobs), executor)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    exit_now()


def exit_now() -> None:
    """Exit at once, killing threads left behind by abandoned warehouse calls.

    A warehouse call that outlives its timeout keeps running on a helper thread, possibly forever. A normal
    interpreter exit joins every executor thread and would wait for them, so flush what we can and exit without
    running exit handlers.
    """
    logger.info("Worker exiting.")
    sentry_sdk.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
