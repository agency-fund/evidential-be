"""CLI interface for the task queue."""

import signal
import sys
from typing import Annotated, Optional

import typer
from loguru import logger

from xngin.tq.handlers import event_created_handler
from xngin.tq.queue import TaskQueue

app = typer.Typer(help="Task queue processor for xngin")


@app.command()
def run(
    dsn: Annotated[
        str,
        typer.Option(
            "--dsn",
            help="Database connection string",
            envvar="XNGIN_TQ_DSN",
        ),
    ],
    max_retries: Annotated[
        int,
        typer.Option(
            "--max-retries",
            help="Maximum number of retries for a task",
        ),
    ] = 3,
    poll_interval: Annotated[
        int,
        typer.Option(
            "--poll-interval",
            help="Interval in seconds to poll for tasks when no notifications are received",
        ),
    ] = 30,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Log level",
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
