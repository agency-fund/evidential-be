"""Task handlers for the task queue."""

import httpx
import psycopg
from loguru import logger
from psycopg.types.json import Jsonb
from xngin.events.experiment_created import WebhookSent
from xngin.tq.task_queue import Task
from xngin.tq.task_types import WebhookOutboundTask


def make_webhook_outbound_handler(dsn: str):
    def webhook_outbound_handler(task: Task):
        """Handle a webhook.outbound task."""
        logger.info(f"Handling event.created task: {task.id}")

        if not task.payload:
            logger.error("Task payload is empty")
            raise ValueError("Task payload is empty")

        payload = WebhookOutboundTask.model_validate(task.payload)
        logger.info(f"Processing {payload}")

        with psycopg.connect(dsn) as conn:
            try:
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
                    conn.execute(
                        "INSERT INTO events (organization_id, type, data) VALUES (%s, %s, %s)",
                        (
                            payload.organization_id,
                            "webhook.sent",
                            Jsonb(
                                WebhookSent(
                                    request=payload,
                                    success=True,
                                    response=f"{response.status_code}",
                                ).model_dump()
                            ),
                        ),
                    )
                else:
                    logger.info(
                        f"Outbound webhook failed with status {response.status_code}"
                    )
                    response.raise_for_status()
            except httpx.HTTPError as e:
                logger.exception("Outbound webhook failed")
                conn.execute(
                    "INSERT INTO events (organization_id, type, data) VALUES (%s, %s, %s)",
                    (
                        payload.organization_id,
                        "webhook.sent",
                        Jsonb(
                            WebhookSent(
                                request=payload, success=False, response=str(e)
                            ).model_dump()
                        ),
                    ),
                )

    return webhook_outbound_handler
