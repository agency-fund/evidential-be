"""tq is a simple Postgres task queue daemon."""

import os
import sys
from typing import Annotated

import typer
from loguru import logger
from xngin.tq.handlers import make_webhook_outbound_handler
from xngin.tq.task_payload_types import WEBHOOK_OUTBOUND_TASK_TYPE
from xngin.tq.task_queue import TaskQueue

# The maximum number of times we will try to complete a task before marking it as "dead".
DEFAULT_MAX_RETRIES = 10

# Default polling interval (in seconds).
DEFAULT_POLLING_INTERVAL = 60

if sentry_dsn := os.environ.get("SENTRY_DSN"):
    import sentry_sdk

    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=os.environ.get("ENVIRONMENT", "local"),
        traces_sample_rate=1.0,
        _experiments={
            "continuous_profiling_auto_start": True,
        },
    )

app = typer.Typer(help="Task queue processor for xngin")


@app.command()
def run(
    dsn: Annotated[
        str,
        typer.Option(
            "--dsn",
            help="Database connection string",
            envvar="DATABASE_URL",
        ),
    ],
    max_retries: Annotated[
        int,
        typer.Option(
            "--max-retries",
            min=0,
            help="Maximum number of retries for a task. Note: the task is always tried once.",
        ),
    ] = DEFAULT_MAX_RETRIES,
    poll_interval: Annotated[
        int,
        typer.Option(
            "--poll-interval",
            min=1,
            help="Interval in seconds to poll for tasks when no notifications are received",
        ),
    ] = DEFAULT_POLLING_INTERVAL,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Log level",
            envvar="LOGURU_LEVEL",
        ),
    ] = "DEBUG",
) -> None:
    """Run the task queue processor."""
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.info(f"Starting task queue with DSN: {dsn}")

    queue = TaskQueue(
        dsn=dsn, max_retries=max_retries, poll_interval_secs=poll_interval
    )
    queue.register_handler(
        WEBHOOK_OUTBOUND_TASK_TYPE, make_webhook_outbound_handler(dsn)
    )
    queue.run()
