"""Task queue implementation using Postgres."""

# mypy: disable-error-code="misc"

import asyncio
import inspect
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import psycopg
import sentry_sdk
from loguru import logger
from psycopg.rows import dict_row
from sqlalchemy import make_url

from xngin.apiserver.worker.executor import Clock

TQ_DB_APPLICATION_NAME = f"tq-{os.getpid()}"

CONNECT_TIMEOUT_SECS = 10
RECONNECT_DELAY_SECS = 5

# A task still running after this long is assumed to have been orphaned by a process that exited mid-task, and is
# claimed again. Handlers time out their HTTP requests after 10 seconds, so a live task never runs this long.
DEFAULT_STALE_RUNNING_SECS = 15 * 60


@dataclass
class Task:
    """Represents a task in the queue."""

    id: str
    created_at: datetime
    updated_at: datetime
    task_type: str
    retry_count: int
    status: str
    embargo_until: datetime
    payload: dict[str, Any] | None = None
    event_id: str | None = None
    message: str | None = None

    def __repr__(self) -> str:
        """Return a string representation of the task."""
        return f"Task(id={self.id}, type={self.task_type}, retry_count={self.retry_count})"


class TaskHandler(Protocol):
    """Protocol for task handlers.

    Handlers should raise an exception if there is an error.
    """

    def __call__(
        self,
        task: Task,
    ):
        """Handle a task."""
        ...


class TaskQueue:
    """Task queue implementation using Postgres."""

    def __init__(
        self,
        dsn: str,
        max_retries: int,
        poll_interval_secs: int,
        clock: Clock,
        stale_running_secs: int = DEFAULT_STALE_RUNNING_SECS,
    ):
        """Initialize the task queue.

        Args:
            dsn: Database connection string.
            max_retries: Maximum number of retries for a task. Note: the task is always tried once.
            poll_interval_secs: Interval in seconds to poll for tasks when no notifications are received.
            clock: The clock for the poll interval and reconnect delays, which tests replace to control time.
            stale_running_secs: Seconds after which a running task is assumed orphaned and is claimed again.
        """
        self.dsn = dsn
        self.max_retries = max_retries
        self.poll_interval = poll_interval_secs
        self.clock = clock
        self.stale_running_secs = stale_running_secs
        self.handlers: dict[str, TaskHandler] = {}

    def register_handler(self, task_type: str, handler: TaskHandler) -> None:
        """Register a handler for a task type.

        Args:
            task_type: The task type to handle.
            handler: The handler function.
        """
        logger.info(f"Registering handler for task type: {task_type}")
        self.handlers[task_type] = handler

    def _setup_notifications(self, conn: psycopg.Connection) -> None:
        """Set up notifications for new tasks.

        This will cause any insert into the tasks table to also send a NOTIFY immediately.
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE OR REPLACE FUNCTION notify_task_queue() RETURNS TRIGGER AS
                $$
                BEGIN
                    PERFORM pg_notify('task_queue', 'new_task');
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                """
            )

            cur.execute(
                """
                CREATE OR REPLACE TRIGGER task_queue_notify_trigger
                    AFTER INSERT
                    ON tasks
                    FOR EACH ROW
                EXECUTE FUNCTION notify_task_queue();
                """
            )

            # Listen for notifications
            cur.execute("LISTEN task_queue;")
            conn.commit()

    def _fetch_task(self, conn: psycopg.Connection) -> Task | None:
        """Fetch a task from the queue.

        Also reclaims tasks orphaned in the running state, counting the lost attempt as a retry so that a task that
        keeps getting orphaned eventually exhausts its retries. A reclaimed task may run more than once.
        """
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                t"""
                UPDATE tasks
                SET status = 'running',
                    updated_at = NOW(),
                    retry_count = retry_count + CASE WHEN status = 'running' THEN 1 ELSE 0 END
                WHERE id = (
                    SELECT id FROM tasks
                    WHERE (
                        status = 'pending'
                        AND embargo_until <= NOW()
                        AND retry_count <= {self.max_retries}
                    ) OR (
                        status = 'running'
                        AND updated_at < NOW() - make_interval(secs => {self.stale_running_secs})
                    )
                    ORDER BY created_at
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING *
                """
            )
            row = cur.fetchone()
            conn.commit()
            return Task(**row) if row else None

    def _handle_task(self, conn: psycopg.Connection, task: Task) -> None:
        """Handle a task."""
        if task.retry_count > self.max_retries:
            # Reclaimed from a process that died during the task's last allowed attempt, so it is not run again.
            logger.warning(f"Task {task.id} was orphaned while running with no retries left, marked as dead")
            self._mark_task_dead(conn, task, "Orphaned while running, with no retries left.")
            return

        logger.info(f"Handling task: {task}")

        if task.task_type not in self.handlers:
            logger.error(f"No handler registered for task type: {task.task_type}")
            self._mark_task_failed(conn, task, f"No handler for task type: {task.task_type}")
            return

        handler = self.handlers[task.task_type]

        try:
            result = handler(task)
            if inspect.iscoroutine(result):
                asyncio.run(result)
            self._mark_task_completed(conn, task)
        except Exception as e:
            logger.exception(f"Error handling task {task.id}")
            self._mark_task_failed(conn, task, str(e))

    def _mark_task_completed(self, conn: psycopg.Connection, task: Task) -> None:
        """Mark a task as completed by setting its status to 'success'."""
        with conn.cursor() as cur:
            cur.execute(
                t"""
                UPDATE tasks
                SET status = 'success', updated_at = NOW(), message = null
                WHERE id = {task.id}
                """
            )
            conn.commit()
            logger.info(f"Task {task.id} completed and marked as successful")
        self._count_outcome(task, "succeeded")

    def _mark_task_dead(self, conn: psycopg.Connection, task: Task, message: str) -> None:
        """Mark a task as dead without running it, when its lost attempt has already been counted."""
        with conn.cursor() as cur:
            cur.execute(
                t"""
                UPDATE tasks
                SET status = 'dead', updated_at = NOW(), message = {message}
                WHERE id = {task.id}
                """
            )
            conn.commit()
        self._count_outcome(task, "failed")

    def _mark_task_failed(self, conn: psycopg.Connection, task: Task, err: str) -> None:
        """Mark a task as failed."""
        with conn.cursor() as cur:
            if task.retry_count >= self.max_retries:
                # Mark as dead if max retries reached
                cur.execute(
                    t"""
                    UPDATE tasks
                    SET status = 'dead',
                        retry_count = retry_count + 1,
                        updated_at = NOW(),
                        message = {err}
                    WHERE id = {task.id}
                    """
                )
                logger.warning(f"Task {task.id} failed and reached max retries, marked as dead")
            else:
                # Calculate backoff time for next retry
                backoff_minutes = min(2**task.retry_count, 15)

                # Reset to pending for retry with embargo using Postgres interval
                cur.execute(
                    t"""
                    UPDATE tasks
                    SET status = 'pending',
                        retry_count = retry_count + 1,
                        updated_at = NOW(),
                        embargo_until = NOW() + make_interval(mins => {backoff_minutes}),
                        message = {err}
                    WHERE id = {task.id}
                    """
                )
                logger.info(
                    f"Task {task.id} failed, retry count now {task.retry_count + 1}, next attempt after "
                    f"{backoff_minutes} minutes"
                )
            conn.commit()
        self._count_outcome(task, "failed")

    def _count_outcome(self, task: Task, outcome: str) -> None:
        sentry_sdk.metrics.count(f"worker.tasks.{outcome}", 1, attributes={"task_type": task.task_type})

    def _wait_for_notification(self, conn: psycopg.Connection, stop: threading.Event) -> None:
        """Wait until a task is inserted, poll_interval elapses, or stop is set.

        Waits in slices of at most a second on the same connection, so that stop is noticed promptly without
        reconnecting and re-running the trigger DDL in _setup_notifications.
        """
        deadline = self.clock.monotonic() + self.poll_interval
        while not stop.is_set():
            remaining = deadline - self.clock.monotonic()
            if remaining <= 0:
                logger.debug(f"Waited {self.poll_interval} for notification. Polling again.")
                return
            for _ in conn.notifies(timeout=min(remaining, 1), stop_after=1):
                logger.debug("Received notification about new tasks")
                return

    def run(self, stop: threading.Event) -> None:
        """Run the task queue processor until stop is set.

        The loop exits after the task in progress, if any, finishes.
        """
        logger.info(f"Starting task queue with DSN: {make_url(self.dsn)}")

        # Connect and set up notifications once per connection, since the trigger DDL locks the tasks table. Reconnect
        # only after the loop fails.
        while not stop.is_set():
            try:
                with psycopg.connect(
                    self.dsn,
                    autocommit=False,
                    application_name=TQ_DB_APPLICATION_NAME,
                    connect_timeout=CONNECT_TIMEOUT_SECS,
                ) as conn:
                    self._setup_notifications(conn)

                    # Main task handling loop: Handle any new tasks, then wait for NOTIFY or polling_interval, repeat.
                    while not stop.is_set():
                        task = self._fetch_task(conn)
                        if task:
                            self._handle_task(conn, task)
                            continue

                        logger.debug("No tasks available, waiting for notifications...")
                        self._wait_for_notification(conn, stop)

            except psycopg.OperationalError as e:
                logger.error(f"Database connection error: {e}")
                logger.info(f"Reconnecting in {RECONNECT_DELAY_SECS} seconds...")
                self.clock.wait(stop, RECONNECT_DELAY_SECS)
            except Exception as exc:
                # Any other error, such as a query failing or a tasks row that no longer matches Task, is a bug
                # in the queue itself. Report it and retry on a fresh connection, so the worker's other jobs keep
                # running rather than the whole worker exiting into a restart loop. Only errors outside this loop
                # reach running_task_queue, which then stops the worker.
                logger.opt(exception=exc).error(f"Task queue failed. Reconnecting in {RECONNECT_DELAY_SECS} seconds...")
                sentry_sdk.capture_exception(exc)
                self.clock.wait(stop, RECONNECT_DELAY_SECS)
