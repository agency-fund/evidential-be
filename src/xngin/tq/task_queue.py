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

    def __repr__(self) -> str:
        """Return a string representation of the task."""
        return (
            f"Task(id={self.id}, type={self.task_type}, retry_count={self.retry_count})"
        )


class TaskHandler(Protocol):
    """Protocol for task handlers."""

    def __call__(
        self,
        task: Task,
    ):
        """Handle a task.

        Args:
            task: The task to handle.

        Raises:
            Exception: If the task handling fails.
        """
        ...


class TaskQueue:
    """Task queue implementation using Postgres."""

    def __init__(self, dsn: str, max_retries: int = 3, poll_interval: int = 60 * 5):
        """Initialize the task queue.

        Args:
            dsn: Database connection string.
            max_retries: Maximum number of retries for a task.
            poll_interval: Interval in seconds to poll for tasks when no notifications are received.
        """
        self.dsn = dsn
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.handlers: dict[str, TaskHandler] = {}
        self.running = False

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

        Args:
            conn: Database connection.
        """
        with conn.cursor() as cur:
            # Create a trigger function if it doesn't exist
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
            logger.info("Set up notifications for task queue")

    def _fetch_task(self, conn: psycopg.Connection) -> Task | None:
        """Fetch a task from the queue.

        Args:
            conn: Database connection.

        Returns:
            A task if one is available, None otherwise.
        """
        with conn.cursor(row_factory=dict_row) as cur:
            # Use UPDATE with FOR UPDATE SKIP LOCKED to claim a task atomically
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
                conn.commit()  # Commit the status update
                return Task(**row)
            return None

    def _handle_task(self, conn: psycopg.Connection, task: Task) -> None:
        """Handle a task.

        Args:
            conn: Database connection.
            task: The task to handle.
        """
        logger.info(f"Handling task: {task}")

        if task.task_type not in self.handlers:
            logger.error(f"No handler registered for task type: {task.task_type}")
            self._mark_task_failed(
                conn, task, Exception(f"No handler for task type: {task.task_type}")
            )
            return

        handler = self.handlers[task.task_type]

        try:
            handler(task)
            # If we get here, the handler completed without raising an exception
            self._mark_task_completed(conn, task)
        except Exception as e:
            logger.exception(f"Error handling task {task.id}")
            self._mark_task_failed(conn, task, e)

    def _mark_task_completed(self, conn: psycopg.Connection, task: Task) -> None:
        """Mark a task as completed by setting its status to 'success'.

        Args:
            conn: Database connection.
            task: The task to mark as completed.
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tasks
                SET status = 'success', updated_at = NOW()
                WHERE id = %s
                """,
                (task.id,),
            )
            conn.commit()
            logger.info(f"Task {task.id} completed and marked as successful")

    def _mark_task_failed(
        self, conn: psycopg.Connection, task: Task, exc: Exception
    ) -> None:
        """Mark a task as failed.

        Args:
            conn: Database connection.
            task: The task to mark as failed.
            exc: The exception that caused the failure.
        """
        with conn.cursor() as cur:
            # Check if we've reached max retries
            if (
                task.retry_count >= self.max_retries - 1
            ):  # -1 because we're about to increment
                # Mark as dead if max retries reached
                cur.execute(
                    """
                    UPDATE tasks
                    SET status = 'dead',
                        retry_count = retry_count + 1,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (task.id,),
                )
                logger.warning(
                    f"Task {task.id} failed and reached max retries, marked as dead"
                )
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
                        embargo_until = NOW() + INTERVAL '%s minutes'
                    WHERE id = %s
                    """,
                    (backoff_minutes, task.id),
                )
                logger.info(
                    f"Task {task.id} failed, retry count now {task.retry_count + 1}, next attempt after {backoff_minutes} minutes"
                )
            conn.commit()

    def run(self) -> None:
        """Run the task queue processor.

        This method will run indefinitely, processing tasks as they become available.
        """
        self.running = True
        logger.info(f"Starting task queue with DSN: {self.dsn}")

        while self.running:
            try:
                with psycopg.connect(self.dsn, autocommit=False) as conn:
                    self._setup_notifications(conn)

                    while self.running:
                        # Try to fetch and process a task
                        task = self._fetch_task(conn)
                        if task:
                            self._handle_task(conn, task)
                            continue

                        # No task available, wait for notification or timeout
                        logger.debug("No tasks available, waiting for notifications...")
                        conn.commit()  # Release any locks

                        # Wait for notifications with timeout
                        try:
                            gen = conn.notifies(timeout=self.poll_interval)
                            next(gen)
                            logger.debug("Received notification about new tasks")
                            # Continue immediately to process the new task
                            continue
                        except StopIteration:
                            logger.debug("Timeout or other stop.")
                        finally:
                            gen.close()

                        # No notifications received, poll again
                        logger.debug(
                            f"No notifications received in {self.poll_interval}s, polling again"
                        )

            except psycopg.OperationalError as e:
                logger.error(f"Database connection error: {e}")
                logger.info("Reconnecting in 5 seconds...")
                time.sleep(5)

    def stop(self) -> None:
        """Stop the task queue processor."""
        logger.info("Stopping task queue")
        self.running = False
