"""Task handlers for the task queue."""

from collections.abc import Callable

import httpx
from loguru import logger
from xngin.tq.task_queue import Task
from xngin.tq.task_types import WebhookOutboundTask


def webhook_status_handler(
        task: Task,
        on_success: Callable[[], None], on_failure: Callable[[Exception], None]
) -> None:
    """Handle webhook.status task."""
    logger.info(f"Received webhook.status task {task}")
    on_success()


def webhook_outbound_handler(
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

    payload = WebhookOutboundTask.model_validate(task.payload)
    logger.info(f"Processing {payload}")

    try:
        # Send the event data to httpbin.org/post
        response = httpx.request(
            payload.method,
            payload.url,
            json=payload.payload,
            timeout=10.0,
            headers=payload.headers,
        )
        logger.debug(f"Response: {response.content}")
        if response.status_code >= 200 and response.status_code < 300 :
            logger.info(
                f"Successfully sent event data to {payload.url}: {response.status_code}"
            )
            logger.debug(f"Response: {response.json()}")
            on_success()
        else:
            logger.info(f"Outbound webhook failed")
            response.raise_for_status()
    except Exception as e:
        logger.exception(f"Failed to send event data to httpbin.org: {e}")
        on_failure(e)
