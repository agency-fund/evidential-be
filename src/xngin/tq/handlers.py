"""Task handlers for the task queue."""

from urllib.parse import urlparse, urlunparse

import httpx
import sqlalchemy
from loguru import logger
from sqlalchemy import NullPool
from sqlalchemy.orm import Session

from xngin.apiserver.dns.safe_resolve import DnsLookupError, safe_resolve
from xngin.apiserver.sqla import tables
from xngin.events.webhook_sent import WebhookSentEvent
from xngin.tq.task_payload_types import WebhookOutboundTask
from xngin.tq.task_queue import TQ_DB_APPLICATION_NAME, Task


def _record_webhook_sent_event(
    session: Session,
    request: WebhookOutboundTask,
    *,
    success: bool,
    response_summary: str,
    log_exc_message: str | None = None,
) -> None:
    if log_exc_message is not None:
        logger.exception(log_exc_message)
    session.add(
        tables.Event(
            organization_id=request.organization_id,
            type=WebhookSentEvent.TYPE,
        ).set_data(WebhookSentEvent(request=request, success=success, response=response_summary))
    )


def make_webhook_outbound_handler(dsn: str):
    """Returns a webhook outbound handler bound with the DSN via a SQLAlchemy engine.

    Also creates an entry in the Event table with information that will be useful for
    customers when debugging.
    """
    engine = sqlalchemy.create_engine(
        dsn, connect_args={"application_name": TQ_DB_APPLICATION_NAME}, poolclass=NullPool
    )

    def webhook_outbound_handler(task: Task):
        """Handle a webhook.outbound task.

        The payload is assumed to be a WebhookOutboundTask.
        """
        if not task.payload:
            logger.error("Task payload is empty")
            raise ValueError("Task payload is empty")

        request = WebhookOutboundTask.model_validate(task.payload)
        logger.info(f"Processing {request}")

        parsed = urlparse(request.url)

        with Session(engine) as session:
            try:
                scheme = parsed.scheme
                hostname = parsed.hostname
                safe_ip = safe_resolve(hostname)
                assert hostname is not None  # safe_resolve() checks this but mypy doesn't know that
                port = parsed.port

                ip_host = f"[{safe_ip}]" if ":" in safe_ip else safe_ip
                ip_host_port = f"{ip_host}:{port}" if port else ip_host
                connect_url = urlunparse((
                    scheme,
                    ip_host_port,
                    parsed.path or "/",
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                ))

                host_header = f"{hostname}:{port}" if port else hostname
                outbound_headers = {**request.headers, "Host": host_header}

                extensions = {"sni_hostname": hostname} if scheme == "https" else None

                response: httpx.Response | None = None
                with httpx.Client(timeout=10.0) as client:
                    response = client.request(
                        request.method,
                        connect_url,
                        json=request.body,
                        headers=outbound_headers,
                        extensions=extensions,
                    )

                if 200 <= response.status_code < 300:
                    logger.info(f"Successfully sent event data to {request.url}: {response.status_code}")
                    if response.content:
                        logger.debug(f"Response has content: {response.text}")
                    _record_webhook_sent_event(
                        session,
                        request,
                        success=True,
                        response_summary=f"{response.status_code}",
                    )
                else:
                    logger.info(f"Outbound webhook failed with status {response.status_code}")
                    response.raise_for_status()
            except DnsLookupError as err:
                message = f"Failed to resolve hostname from webhook URL: ({err!s})"
                _record_webhook_sent_event(
                    session,
                    request,
                    success=False,
                    response_summary=message,
                    log_exc_message=message,
                )
                raise
            except httpx.HTTPError as err:
                message = f"status={response.status_code} message={err!s}" if response else str(err)
                _record_webhook_sent_event(
                    session,
                    request,
                    success=False,
                    response_summary=message,
                    log_exc_message="Outbound webhook failed",
                )
                raise
            finally:
                session.commit()

    return webhook_outbound_handler
