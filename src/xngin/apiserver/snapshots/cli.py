"""snapshotter applies autofail updates."""

import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Annotated

import typer
from loguru import logger
from sentry_sdk.crons import monitor

from xngin.apiserver import customlogging, database
from xngin.apiserver.snapshots import autofail
from xngin.ops import sentry
from xngin.xsecrets import secretservice

ENV_CRONJOB_MONITOR_SLUG = "CRONJOB_MONITOR_SLUG"

customlogging.setup()
sentry.setup()

app = typer.Typer(help="Applies autofail updates as needed.")


def autofail_collect(
    autofail_timeout: float,
    autofail_batch_sleep: float,
    autofail_batch_size: int,
    *,
    database_setup: Callable[[], AbstractContextManager[None]] = database.setup,
    process_autofails: Callable[[float, float, int], None] = autofail.process_autofails,
) -> None:
    """Process eligible autofail updates within a bounded runtime."""
    with database_setup():
        process_autofails(autofail_timeout, autofail_batch_sleep, autofail_batch_size)


@app.command()
def collect(
    autofail_timeout: Annotated[
        int,
        typer.Option(
            "--autofail-timeout",
            min=1,
            help="Maximum total autofail processing time (in seconds). The timeout is checked before each batch.",
        ),
    ] = autofail.DEFAULT_AUTOFAIL_TIMEOUT_SECS,
    autofail_batch_sleep: Annotated[
        float,
        typer.Option(
            "--autofail-batch-sleep",
            min=0,
            help="Delay between committed autofail batches (in seconds).",
        ),
    ] = autofail.DEFAULT_AUTOFAIL_BATCH_SLEEP_SECS,
    autofail_batch_size: Annotated[
        int,
        typer.Option(
            "--autofail-batch-size",
            min=1,
            help="Maximum number of autofail updates to commit in one transaction.",
        ),
    ] = autofail.DEFAULT_AUTOFAIL_BATCH_SIZE,
):
    """Apply autofail outcomes to the draws whose autofail windows have expired."""
    secretservice.setup()

    cronjob_monitor_slug = os.environ.get(ENV_CRONJOB_MONITOR_SLUG, "")
    if cronjob_monitor_slug:
        with monitor(monitor_slug=cronjob_monitor_slug):
            autofail_collect(autofail_timeout, autofail_batch_sleep, autofail_batch_size)
    else:
        autofail_collect(autofail_timeout, autofail_batch_sleep, autofail_batch_size)
    logger.info("collect() finished successfully.")
