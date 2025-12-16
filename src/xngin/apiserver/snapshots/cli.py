"""snapshotter collects snapshots."""

import asyncio
import os
from datetime import timedelta
from typing import Annotated

import typer
from loguru import logger
from sentry_sdk.crons import monitor

from xngin.apiserver import customlogging, database
from xngin.apiserver.snapshots import snapshotter
from xngin.ops import sentry
from xngin.xsecrets import secretservice

ENV_CRONJOB_MONITOR_SLUG = "CRONJOB_MONITOR_SLUG"

# Use os.process_cpu_count() whenever we can move to python 3.13
NPROC = max(4, len(os.sched_getaffinity(0)) // 4) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 4

customlogging.setup()
sentry.setup()

app = typer.Typer(help="Collects snapshots as needed.")


async def acollect(snapshot_interval: int, snapshot_timeout: int, parallelism: int):
    """Collects snapshots (async wrapper)."""
    async with database.setup():
        await snapshotter.create_pending_snapshots(snapshot_interval)
        async with asyncio.TaskGroup() as task:
            for i in range(parallelism):
                with logger.contextualize(task=i):
                    _ = task.create_task(snapshotter.process_pending_snapshots(snapshot_timeout), name=f"sn{i}")


@app.command()
def collect(
    snapshot_timeout: Annotated[
        int,
        typer.Option(
            "--max-time",
            min=1,
            help="Maximum duration of a single snapshot (in seconds). "
            "Snapshots that take longer than this will be marked as failures.",
        ),
    ] = snapshotter.SNAPSHOT_TIMEOUT_SECS,
    snapshot_interval: Annotated[
        int, typer.Option("--interval", min=60, help="The target interval between snapshots (in seconds).")
    ] = timedelta(hours=6).seconds,
    parallelism: Annotated[
        int,
        typer.Option(
            "-j",
            min=1,
            help="Number of snapshotting tasks to spawn. This controls the number of potential blocking DWH operations "
            "that may occur simultaneously.",
        ),
    ] = NPROC,
):
    """Collect snapshots from the experiments that need them.

    Experiments with successful snapshots will not be snapshot again before --snapshot-interval has elapsed between
    the latest successful snapshot and the current time.

    To enable automatic retries of failed jobs, and to allow for clock variances, the duration between invocations of
    this job should be some fraction of --snapshot-interval. For example, if --snapshot-interval is set to 6 hours,
    consider running this job every hour. This effectively enables hourly retries, allows for non-monotonic clock
    behaviors, and some tolerance for unpredictable cron scheduling or missed invocations.
    """
    secretservice.setup()

    cronjob_monitor_slug = os.environ.get(ENV_CRONJOB_MONITOR_SLUG, "")
    if cronjob_monitor_slug:
        with monitor(monitor_slug=cronjob_monitor_slug):
            asyncio.run(acollect(snapshot_interval, snapshot_timeout, parallelism))
    else:
        asyncio.run(acollect(snapshot_interval, snapshot_timeout, parallelism))
