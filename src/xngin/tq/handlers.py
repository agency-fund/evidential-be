"""Task handlers for the task queue."""

import httpx
import sqlalchemy
from loguru import logger
from sqlalchemy import NullPool
from sqlalchemy.orm import Session
from xngin.apiserver.models import tables
from xngin.events.webhook_sent import WebhookSentEvent
from xngin.tq.task_queue import Task
from xngin.tq.task_types import WebhookOutboundTask


def make_webhook_outbound_handler(dsn: str):
    """Returns a webhook outbound handler bound with the DSN via a SQLAlchemy engine.

    Also creates an entry in the Event table with information that will be useful for
    customers when debugging.
    """
    engine = sqlalchemy.create_engine(dsn, poolclass=NullPool)

    def webhook_outbound_handler(task: Task):
        """Handle a webhook.outbound task."""
        if not task.payload:
            logger.error("Task payload is empty")
            raise ValueError("Task payload is empty")

        request = WebhookOutboundTask.model_validate(task.payload)
        logger.info(f"Processing {request}")

        with Session(engine) as session:
            try:
                response = httpx.request(
                    request.method,
                    request.url,
                    json=request.body,
                    timeout=10.0,
                    headers=request.headers,
                )
                logger.debug(f"Response: {response.content}")

                if response.status_code >= 200 and response.status_code < 300:
                    logger.info(
                        f"Successfully sent event data to {request.url}: {response.status_code}"
                    )
                    logger.debug(f"Response: {response.json()}")
                    session.add(
                        tables.Event(
                            organization_id=request.organization_id,
                            type="webhook.sent",
                            data=WebhookSentEvent(
                                request=request,
                                success=True,
                                response=f"{response.status_code}",
                            ).model_dump(),
                        )
                    )
                else:
                    logger.info(
                        f"Outbound webhook failed with status {response.status_code}"
                    )
                    response.raise_for_status()
            except httpx.HTTPError as err:
                logger.exception("Outbound webhook failed")
                if "response" in locals():
                    message = f"status={response.status_code} message={err!s}"
                else:
                    message = str(err)
                session.add(
                    tables.Event(
                        organization_id=request.organization_id,
                        type="webhook.sent",
                        data=WebhookSentEvent(
                            request=request,
                            success=False,
                            response=message,
                        ).model_dump(),
                    )
                )
                raise
            finally:
                session.commit()

    return webhook_outbound_handler
