"""Task handlers for the task queue."""

from collections.abc import Callable

import httpx
from loguru import logger
from xngin.tq.task_queue import Task


def event_created_handler(
    task: Task, on_success: Callable[[], None], on_failure: Callable[[Exception], None]
) -> None:
    """Handle an event.created task.

    This handler sends the event data to httpbin.org/post as an example.

    Args:
        task: The task to handle.
        on_success: Callback to call when the task is successfully handled.
        on_failure: Callback to call when the task handling fails.
    """
    logger.info(f"Handling event.created task: {task.id}")

    if not task.payload:
        logger.error("Task payload is empty")
        on_failure(ValueError("Task payload is empty"))
        return

    try:
        # Send the event data to httpbin.org/post
        response = httpx.post(
            "https://httpbin.org/post",
            json=task.payload,
            timeout=10.0,
        )
        response.raise_for_status()

        # Log the response
        logger.info(
            f"Successfully sent event data to httpbin.org: {response.status_code}"
        )
        logger.debug(f"Response: {response.json()}")

        # Mark the task as completed
        on_success()
    except Exception as e:
        logger.exception(f"Failed to send event data to httpbin.org: {e}")
        on_failure(e)
