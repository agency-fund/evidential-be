"""snapshotter collects snapshots."""

import asyncio
import os
import sys
from datetime import timedelta
from typing import Annotated

import typer
from loguru import logger

from xngin.apiserver import database
from xngin.apiserver.snapshots import snapshotter
from xngin.xsecrets import secretservice

NPROC = max(4, len(os.sched_getaffinity(0)) // 4)

if sentry_dsn := os.environ.get("SENTRY_DSN"):
    import sentry_sdk
    from sentry_sdk import scrubber

    denylist = [
        *scrubber.DEFAULT_DENYLIST,
        "dsn",
    ]
    pii_denylist = [
        *scrubber.DEFAULT_PII_DENYLIST,
        "webhook_token",
        "email",
    ]

    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=os.environ.get("ENVIRONMENT", "local"),
        traces_sample_rate=1.0,
        _experiments={
            "continuous_profiling_auto_start": True,
        },
        send_default_pii=False,
        event_scrubber=sentry_sdk.scrubber.EventScrubber(denylist=denylist, pii_denylist=pii_denylist),
    )

app = typer.Typer(help="Collects snapshots as needed.")


async def acollect(snapshot_interval: int, snapshot_timeout: int, parallelism: int):
    """Collects snapshots (async wrapper)."""
    async with database.setup():
        await snapshotter.create_pending_snapshots(snapshot_interval)
        async with asyncio.TaskGroup() as task:
            for i in range(parallelism):
                _ = task.create_task(snapshotter.process_pending_snapshots(snapshot_timeout), name=f"sn{i}")


@app.command()
def collect(
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Log level",
            envvar="LOGURU_LEVEL",
        ),
    ] = "DEBUG",
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
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    secretservice.setup()

    # Typer doesn't support async.
    asyncio.run(acollect(snapshot_interval, snapshot_timeout, parallelism))
