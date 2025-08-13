"""Task queue implementation using Postgres."""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import psycopg
from loguru import logger
from psycopg.rows import dict_row


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

    def __init__(self, dsn: str, max_retries: int, poll_interval_secs: int):
        """Initialize the task queue.

        Args:
            dsn: Database connection string.
            max_retries: Maximum number of retries for a task. Note: the task is always tried once.
            poll_interval_secs: Interval in seconds to poll for tasks when no notifications are received.
        """
        self.dsn = dsn
        self.max_retries = max_retries
        self.poll_interval = poll_interval_secs
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
                CREATE OR REPLACE FUNCTION notify_task_queue() RETURNS TRIGGER AS $$
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
                AFTER INSERT ON tasks
                FOR EACH ROW
                EXECUTE FUNCTION notify_task_queue();
                """
            )

            # Listen for notifications
            cur.execute("LISTEN task_queue;")
            conn.commit()

    def _fetch_task(self, conn: psycopg.Connection) -> Task | None:
        """Fetch a task from the queue."""
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE tasks
                SET status = 'running', updated_at = NOW()
                WHERE id IN (
                    SELECT id FROM tasks
                    WHERE status = 'pending'
                    AND embargo_until <= NOW()
                    AND retry_count <= %s
                    ORDER BY created_at
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING *
                """,
                (self.max_retries,),
            )
            row = cur.fetchone()
            if row:
                conn.commit()
                return Task(**row)
            return None

    def _handle_task(self, conn: psycopg.Connection, task: Task) -> None:
        """Handle a task."""
        logger.info(f"Handling task: {task}")

        if task.task_type not in self.handlers:
            logger.error(f"No handler registered for task type: {task.task_type}")
            self._mark_task_failed(conn, task, f"No handler for task type: {task.task_type}")
            return

        handler = self.handlers[task.task_type]

        try:
            handler(task)
            self._mark_task_completed(conn, task)
        except Exception as e:
            logger.exception(f"Error handling task {task.id}")
            self._mark_task_failed(conn, task, str(e))

    def _mark_task_completed(self, conn: psycopg.Connection, task: Task) -> None:
        """Mark a task as completed by setting its status to 'success'."""
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tasks
                SET status = 'success', updated_at = NOW(), message = null
                WHERE id = %s
                """,
                (task.id,),
            )
            conn.commit()
            logger.info(f"Task {task.id} completed and marked as successful")

    def _mark_task_failed(self, conn: psycopg.Connection, task: Task, err: str) -> None:
        """Mark a task as failed."""
        with conn.cursor() as cur:
            if task.retry_count >= self.max_retries:
                # Mark as dead if max retries reached
                cur.execute(
                    """
                    UPDATE tasks
                    SET status = 'dead',
                        retry_count = retry_count + 1,
                        updated_at = NOW(),
                        message = %s
                    WHERE id = %s
                    """,
                    (
                        err,
                        task.id,
                    ),
                )
                logger.warning(f"Task {task.id} failed and reached max retries, marked as dead")
            else:
                # Calculate backoff time for next retry
                backoff_minutes = min(2**task.retry_count, 15)

                # Reset to pending for retry with embargo using Postgres interval
                cur.execute(
                    """
                    UPDATE tasks
                    SET status = 'pending',
                        retry_count = retry_count + 1,
                        updated_at = NOW(),
                        embargo_until = NOW() + INTERVAL '%s minutes',
                        message = %s
                    WHERE id = %s
                    """,
                    (backoff_minutes, err, task.id),
                )
                logger.info(
                    f"Task {task.id} failed, retry count now {task.retry_count + 1}, next attempt after {backoff_minutes} minutes"
                )
            conn.commit()

    def run(self) -> None:
        """Run the task queue processor.

        This method will run indefinitely, processing tasks as they become available.
        """
        logger.info(f"Starting task queue with DSN: {self.dsn}")

        # Main task handling loop: Handle any new tasks, then wait for NOTIFY or polling_interval, repeat.
        while True:
            try:
                with psycopg.connect(self.dsn, autocommit=False) as conn:
                    self._setup_notifications(conn)

                    task = self._fetch_task(conn)
                    if task:
                        self._handle_task(conn, task)
                        continue

                    logger.debug("No tasks available, waiting for notifications...")
                    conn.commit()

                    try:
                        gen = conn.notifies(timeout=self.poll_interval)
                        next(gen)
                        logger.debug("Received notification about new tasks")
                        continue
                    except StopIteration:
                        # StopIteration is raised when poll_interval expires
                        logger.debug(f"Waited {self.poll_interval} for notification. Polling again.")
                        continue
                    finally:
                        gen.close()

            except psycopg.OperationalError as e:
                logger.error(f"Database connection error: {e}")
                logger.info("Reconnecting in 5 seconds...")
                time.sleep(5)
