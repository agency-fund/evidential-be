"""CLI interface for the task queue."""
import os
import signal
import sys
from typing import Annotated

import typer
from loguru import logger
from xngin.tq.handlers import event_created_handler
from xngin.tq.queue import TaskQueue

DEFAULT_MAX_RETRIES = 10

DEFAULT_POLLING_INTERVAL = 5 * 60

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
            help="Maximum number of retries for a task",
        ),
    ] = DEFAULT_MAX_RETRIES,
    poll_interval: Annotated[
        int,
        typer.Option(
            "--poll-interval",
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
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    logger.info(f"Starting task queue with DSN: {dsn}")

    # Create the task queue
    queue = TaskQueue(dsn=dsn, max_retries=max_retries, poll_interval=poll_interval)

    # Register handlers
    queue.register_handler("event.created", event_created_handler)

    # Handle SIGINT and SIGTERM
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        queue.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the queue
    queue.run()


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
