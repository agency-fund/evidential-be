"""Runs the task queue on a thread inside the worker."""

from collections.abc import Generator
from contextlib import contextmanager
from threading import Event, Thread

import sentry_sdk
from loguru import logger

from xngin.apiserver import database
from xngin.apiserver.worker.executor import Clock
from xngin.apiserver.worker.tq.handlers import make_turn_journeys_changed_handler, make_webhook_outbound_handler
from xngin.apiserver.worker.tq.task_payload_types import TURN_JOURNEYS_CHANGED_TASK_TYPE, WEBHOOK_OUTBOUND_TASK_TYPE
from xngin.apiserver.worker.tq.task_queue import TaskQueue


@contextmanager
def running_task_queue(
    database_url: str, clock: Clock, stop: Event, *, max_retries: int, poll_interval: int, join_timeout: float
) -> Generator[None]:
    """Run the task queue on its own thread while the context is active.

    The queue's own connection uses psycopg directly, and the handlers use SQLAlchemy engines with psycopg3. Neither
    uses the worker's database pool.

    Leaving the context sets stop and waits up to join_timeout seconds for the queue to finish its current task.

    The queue retries its own loop after database errors and bugs in task handling. If it nonetheless fails, the
    failure is reported and stop is set, so the whole worker exits and Railway restarts it, rather than leaving a
    worker that no longer processes tasks.
    """
    queue = TaskQueue(
        dsn=database.get_psycopg_dsn(database_url),
        max_retries=max_retries,
        poll_interval_secs=poll_interval,
        clock=clock,
    )
    queue.register_handler(WEBHOOK_OUTBOUND_TASK_TYPE, make_webhook_outbound_handler(database_url))
    queue.register_handler(TURN_JOURNEYS_CHANGED_TASK_TYPE, make_turn_journeys_changed_handler(database_url))

    def run() -> None:
        try:
            queue.run(stop=stop)
        except BaseException as exc:
            logger.opt(exception=exc).error("Task queue stopped unexpectedly.")
            sentry_sdk.capture_exception(exc)
            stop.set()

    thread = Thread(target=run, name="tq")
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=join_timeout)
        if thread.is_alive():
            logger.warning(f"Task queue did not stop within {join_timeout} seconds.")
