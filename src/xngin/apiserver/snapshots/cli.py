"""snapshotter collects snapshots."""

import asyncio
import os
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


async def acollect(
    default_max_snapshot_age: int, mab_dwh_max_snapshot_age: int, snapshot_timeout: int, parallelism: int
):
    """Collects snapshots (async wrapper)."""
    async with database.setup():
        await snapshotter.create_pending_snapshots(default_max_snapshot_age, mab_dwh_max_snapshot_age)
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
    default_max_snapshot_age: Annotated[
        int,
        typer.Option(
            "--default-max-snapshot-age",
            min=60,
            help="How old an experiment's newest snapshot may be (in seconds) before this job creates "
            "another one. Applies to every type except MAB-DWH.",
        ),
    ] = snapshotter.DEFAULT_MAX_SNAPSHOT_AGE_SECS,
    mab_dwh_max_snapshot_age: Annotated[
        int,
        typer.Option(
            "--mab-dwh-max-snapshot-age",
            min=60,
            help="The same limit for MAB-DWH experiments, whose snapshots also ingest outcomes from "
            "the org's data warehouse, so for them this also sets how often we pull.",
        ),
    ] = snapshotter.DEFAULT_MAB_DWH_MAX_SNAPSHOT_AGE_SECS,
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

    Experiments with successful snapshots will not be snapshot again until their newest snapshot is older
    than the age limit for their type: --mab-dwh-max-snapshot-age for MAB-DWH, --default-max-snapshot-age
    for the rest.

    Neither flag schedules anything. This job only acts when the cron invokes it (schedule set in
    railway.snapshots.json, currently hourly), so the cron period is the frequency with which an experiment
    will be snapshotted, and the age limits decide what qualifies on a given run.

    Run this job at some fraction of --default-max-snapshot-age so that failures retry promptly and a missed
    invocation does not delay everything to the next full period. Running hourly against the 6 hour default
    gives hourly retries plus tolerance for clock drift and unpredictable cron scheduling, and means an
    experiment qualifies on roughly one run in six.

    MAB-DWH ingests outcomes as part of snapshotting, so keep its limit at or below the cron period to ingest on every
    run.
    """
    secretservice.setup()

    cronjob_monitor_slug = os.environ.get(ENV_CRONJOB_MONITOR_SLUG, "")
    if cronjob_monitor_slug:
        with monitor(monitor_slug=cronjob_monitor_slug):
            asyncio.run(acollect(default_max_snapshot_age, mab_dwh_max_snapshot_age, snapshot_timeout, parallelism))
    else:
        asyncio.run(acollect(default_max_snapshot_age, mab_dwh_max_snapshot_age, snapshot_timeout, parallelism))
    logger.info("collect() finished successfully.")
