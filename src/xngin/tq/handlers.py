"""Task handlers for the task queue."""

import httpx
from loguru import logger
from xngin.tq.task_queue import Task
from xngin.tq.task_types import WebhookOutboundTask


def webhook_status_handler(task: Task) -> bool:
    """Handle webhook.status task.
    
    Args:
        task: The task to handle.
        
    Returns:
        True if successful.
        
    Raises:
        Exception: If the task handling fails.
    """
    logger.info(f"Received webhook.status task {task}")
    return True


def webhook_outbound_handler(task: Task) -> bool:
    """Handle an event.created task.

    This handler sends the event data to httpbin.org/post as an example.

    Args:
        task: The task to handle.
        
    Returns:
        True if successful.
        
    Raises:
        ValueError: If the task payload is empty.
        Exception: If the HTTP request fails.
    """
    logger.info(f"Handling event.created task: {task.id}")

    if not task.payload:
        logger.error("Task payload is empty")
        raise ValueError("Task payload is empty")

    payload = WebhookOutboundTask.model_validate(task.payload)
    logger.info(f"Processing {payload}")

    # Send the event data to the webhook URL
    response = httpx.request(
        payload.method,
        payload.url,
        json=payload.payload,
        timeout=10.0,
        headers=payload.headers,
    )
    logger.debug(f"Response: {response.content}")
    
    if response.status_code >= 200 and response.status_code < 300:
        logger.info(
            f"Successfully sent event data to {payload.url}: {response.status_code}"
        )
        logger.debug(f"Response: {response.json()}")
        return True
    else:
        logger.info(f"Outbound webhook failed with status {response.status_code}")
        response.raise_for_status()  # This will raise an HTTPStatusError
