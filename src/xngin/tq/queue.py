"""Task queue implementation using Postgres."""

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Dict, Optional, Protocol

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
    embargo_until: Optional[datetime] = None
    payload: Optional[Dict[str, Any]] = None
    event_id: Optional[str] = None
    
    def __repr__(self) -> str:
        """Return a string representation of the task."""
        return f"Task(id={self.id}, type={self.task_type}, retry_count={self.retry_count})"


class TaskHandler(Protocol):
    """Protocol for task handlers."""

    def __call__(
        self, task: Task, on_success: Callable[[], None], on_failure: Callable[[Exception], None]
    ) -> None:
        """Handle a task.

        Args:
            task: The task to handle.
            on_success: Callback to call when the task is successfully handled.
            on_failure: Callback to call when the task handling fails.
        """
        ...


class TaskQueue:
    """Task queue implementation using Postgres."""

    def __init__(self, dsn: str, max_retries: int = 3, poll_interval: int = 30):
        """Initialize the task queue.

        Args:
            dsn: Database connection string.
            max_retries: Maximum number of retries for a task.
            poll_interval: Interval in seconds to poll for tasks when no notifications are received.
        """
        self.dsn = dsn
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.handlers: Dict[str, TaskHandler] = {}
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

    def _fetch_task(self, conn: psycopg.Connection) -> Optional[Task]:
        """Fetch a task from the queue.

        Args:
            conn: Database connection.

        Returns:
            A task if one is available, None otherwise.
        """
        now = datetime.now(UTC)
        with conn.cursor(row_factory=dict_row) as cur:
            # Use FOR UPDATE SKIP LOCKED to avoid race conditions
            cur.execute(
                """
                SELECT * FROM tasks
                WHERE (embargo_until IS NULL OR embargo_until <= %s) AND retry_count <= %s
                ORDER BY created_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """,
                (now, self.max_retries),
            )
            row = cur.fetchone()
            if row:
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
            self._mark_task_failed(conn, task, Exception(f"No handler for task type: {task.task_type}"))
            return

        handler = self.handlers[task.task_type]

        def on_success() -> None:
            self._mark_task_completed(conn, task)

        def on_failure(exc: Exception) -> None:
            logger.error(f"Task {task.id} failed: {exc}")
            self._mark_task_failed(conn, task, exc)

        try:
            handler(task, on_success, on_failure)
        except Exception as e:
            logger.exception(f"Error handling task {task.id}")
            on_failure(e)

    def _mark_task_completed(self, conn: psycopg.Connection, task: Task) -> None:
        """Mark a task as completed by deleting it.

        Args:
            conn: Database connection.
            task: The task to mark as completed.
        """
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tasks WHERE id = %s", (task.id,))
            conn.commit()
            logger.info(f"Task {task.id} completed and removed from queue")

    def _mark_task_failed(self, conn: psycopg.Connection, task: Task, exc: Exception) -> None:
        """Mark a task as failed.

        Args:
            conn: Database connection.
            task: The task to mark as failed.
            exc: The exception that caused the failure.
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tasks
                SET retry_count = retry_count + 1,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (task.id,),
            )
            logger.info(f"Task {task.id} failed, retry count now {task.retry_count + 1}")
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
                            logger.debug("Closing generator")
                            gen.close()

                        # No notifications received, poll again
                        logger.debug(f"No notifications received in {self.poll_interval}s, polling again")

            except psycopg.OperationalError as e:
                logger.error(f"Database connection error: {e}")
                logger.info("Reconnecting in 5 seconds...")
                time.sleep(5)


    def stop(self) -> None:
        """Stop the task queue processor."""
        logger.info("Stopping task queue")
        self.running = False

    def enqueue(
        self,
        task_type: str,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        embargo_until: Optional[datetime] = None,
    ) -> str:
        """Enqueue a new task.

        Args:
            task_type: The type of task.
            payload: Optional payload for the task.
            event_id: Optional ID of an event associated with this task.
            embargo_until: Optional time until which the task should not be processed.

        Returns:
            The ID of the newly created task.
        """
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tasks (
                        task_type, payload, event_id, embargo_until
                    ) VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        task_type,
                        json.dumps(payload) if payload else None,
                        event_id,
                        embargo_until,
                    ),
                )
                task_id = cur.fetchone()[0]
                conn.commit()
                logger.info(f"Enqueued task {task_id} of type {task_type}")
                return task_id
