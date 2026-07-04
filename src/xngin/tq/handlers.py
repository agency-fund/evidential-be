"""Task handlers for the task queue."""

from urllib.parse import urlparse, urlunparse

import httpx2
import sqlalchemy
from loguru import logger
from sqlalchemy import NullPool
from sqlalchemy.orm import Session

from xngin.apiserver import constants
from xngin.apiserver.dns.safe_resolve import DnsLookupError, safe_resolve
from xngin.apiserver.flags import XNGIN_PUBLIC_API_BASE_URL
from xngin.apiserver.sqla import tables
from xngin.events.turn_journeys_changed import TurnJourneysChangedEvent
from xngin.events.webhook_sent import WebhookSentEvent
from xngin.tq.task_payload_types import TurnJourneysChangedTask, WebhookOutboundTask
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


def _record_turn_journeys_changed_event(
    session: Session,
    request: TurnJourneysChangedTask,
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
            type=TurnJourneysChangedEvent.TYPE,
        ).set_data(
            TurnJourneysChangedEvent(
                organization_id=request.organization_id,
                webhook_id=request.webhook_id,
                success=success,
                response=response_summary,
            )
        )
    )


def make_webhook_outbound_handler(dsn: str, *, transport: httpx2.BaseTransport | None = None):
    """Returns a webhook outbound handler bound with the DSN via a SQLAlchemy engine.

    Also creates an entry in the Event table with information that will be useful for
    customers when debugging.

    ``transport`` is forwarded to ``httpx2.Client`` and exists primarily for testing.
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

                ip_host_port = f"{safe_ip}:{port}" if port else safe_ip
                # Retain any credentials in the reconstruction e.g. user:pw if they were provided
                userinfo = parsed.netloc.rsplit("@", 1)[0] if "@" in parsed.netloc else None
                authority = f"{userinfo}@{ip_host_port}" if userinfo else ip_host_port
                connect_url = urlunparse((
                    scheme,
                    authority,
                    parsed.path or "/",
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                ))

                host_header = f"{hostname}:{port}" if port else hostname
                outbound_headers = {**request.headers, "Host": host_header}

                extensions = {"sni_hostname": hostname} if scheme == "https" else None

                response: httpx2.Response | None = None
                with httpx2.Client(timeout=10.0, transport=transport) as client:
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
            except httpx2.HTTPError as err:
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


def make_turn_journeys_changed_handler(dsn: str, *, transport: httpx2.AsyncBaseTransport | None = None):
    """Returns a turn_journeys_changed handler bound with the DSN via a SQLAlchemy engine.

    Also creates an entry in the Event table with information that will be useful for
    customers when debugging.

    This handler deliberately does not import or call Evidential's business logic (e.g.
    `refresh_journeys_dict`) directly. Instead it makes a real outbound HTTP request back to this
    same server's `/integrations/turn/{webhook_id}/refresh-journeys` endpoint, authenticating with
    the webhook token from the task payload. `tq` is kept free of dependencies on `sqla.tables` and
    Evidential's business logic, and runs with fewer privileges.
    """
    engine = sqlalchemy.create_engine(
        dsn, connect_args={"application_name": TQ_DB_APPLICATION_NAME}, poolclass=NullPool
    )

    async def turn_journeys_changed_handler(task: Task):
        """Handle a turn_journeys_changed task.

        The payload is assumed to be a TurnJourneysChangedTask.
        """
        if not task.payload:
            logger.error("Task payload is empty")
            raise ValueError("Task payload is empty")

        request = TurnJourneysChangedTask.model_validate(task.payload)
        logger.info(f"Processing {request}")
        with Session(engine) as session:
            try:
                async with httpx2.AsyncClient(transport=transport, timeout=10.0) as client:
                    response = await client.request(
                        method="POST",
                        url=f"{XNGIN_PUBLIC_API_BASE_URL}{constants.API_PREFIX_V1}/integrations/turn/{request.webhook_id}/refresh-journeys",
                        headers={constants.HEADER_WEBHOOK_TOKEN: request.webhook_auth_token},
                    )
                    response.raise_for_status()

                _record_turn_journeys_changed_event(
                    session,
                    request,
                    success=True,
                    response_summary="Journeys dict refreshed successfully",
                )
            except httpx2.HTTPStatusError as err:
                message = f"status={response.status_code} message={err!s}"
                _record_turn_journeys_changed_event(
                    session,
                    request,
                    success=False,
                    response_summary=message,
                    log_exc_message="Refreshing journeys dict failed",
                )
                raise
            finally:
                session.commit()

    return turn_journeys_changed_handler
