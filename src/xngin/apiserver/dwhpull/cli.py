"""dwh-pull reads bandit outcomes from organizations' data warehouses."""

import asyncio
import os
from typing import Annotated

import typer
from loguru import logger
from sentry_sdk.crons import monitor

from xngin.apiserver import customlogging, database
from xngin.apiserver.dwhpull import dwhpull
from xngin.ops import sentry
from xngin.xsecrets import secretservice

ENV_CRONJOB_MONITOR_SLUG = "CRONJOB_MONITOR_SLUG"

customlogging.setup()
sentry.setup()

app = typer.Typer(help="Reads bandit outcomes from organizations' data warehouses.")


async def apull(pull_timeout: int):
    """Pulls outcomes (async wrapper)."""
    async with database.setup():
        await dwhpull.pull_all_experiments(pull_timeout)


@app.command()
def pull(
    pull_timeout: Annotated[
        int,
        typer.Option(
            "--max-time",
            min=1,
            help="Maximum duration of one experiment's pull (in seconds). An experiment that takes "
            "longer than this is abandoned for this run and reported as a failure.",
        ),
    ] = dwhpull.PULL_TIMEOUT_SECS,
):
    """Read outcomes from the data warehouse for every experiment that needs them.

    MAB-DWH experiments have their outcomes read from the organization's data warehouse rather than
    pushed to the API, so this job is what moves those experiments forward: every outcome it applies
    updates the arm's posterior, and later assignments draw on the updated arms.

    This job pulls on every invocation, so the interval between invocations is how quickly an
    outcome that lands in the warehouse reaches the bandit. Outcomes that have not landed yet are
    re-read on the next run, so a missed invocation costs latency and nothing else.
    """
    secretservice.setup()

    cronjob_monitor_slug = os.environ.get(ENV_CRONJOB_MONITOR_SLUG, "")
    if cronjob_monitor_slug:
        with monitor(monitor_slug=cronjob_monitor_slug):
            asyncio.run(apull(pull_timeout))
    else:
        asyncio.run(apull(pull_timeout))
    logger.info("pull() finished successfully.")
