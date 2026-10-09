"""Attaches per-request diagnostic context to log lines and Sentry events."""

import contextvars
import itertools
import time
import typing

import sentry_sdk
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

if typing.TYPE_CHECKING:
    from loguru import Record as loguru_Record

# Railway's edge proxy sets this on every request it forwards. Its value is the requestId field of Railway's HTTP logs.
RAILWAY_REQUEST_ID_HEADER = "x-railway-request-id"

# Identifies the Railway edge point of presence (POP) that handled the request.
RAILWAY_EDGE_HEADER = "x-railway-edge"

# Set by many other proxies and load balancers. Used when Railway's header is absent, i.e. when not hosted on Railway.
REQUEST_ID_HEADER = "x-request-id"

# Bounds what a client that sends its own header can put in our logs.
_MAX_HEADER_VALUE_LENGTH = 64

# Track the time-since-started and ordering of requests handled by this worker.
_worker_started_at = time.monotonic()
_worker_request_seq = itertools.count(1)

# Request fields added to log records.
#
# Sentry receives these separately below.
#
# Unlike logger.contextualize(), these fields remain set after middleware unwinds, so Uvicorn's unhandled-exception
# logs retain the request context.
_request_context: contextvars.ContextVar[dict[str, typing.Any] | None] = contextvars.ContextVar(
    "request_context", default=None
)


def add_request_context(record: loguru_Record) -> None:
    """Adds request fields to a log record, preserving explicitly bound fields."""
    if fields := _request_context.get():
        for key, value in fields.items():
            record["extra"].setdefault(key, value)


class RequestContextMiddleware:
    """ASGI middleware that adds per-request diagnostic context to log lines and Sentry events.

    If Loguru is used, pass add_request_context as the patcher argument to customlogging.setup() to enable request
    context on log records.

    Log lines emitted while handling a request carry a request_id field, and Sentry events carry a request_id tag, so
    that a failed request in the hosting provider's HTTP logs can be found in the application logs and in Sentry, and
    vice versa. The ID comes from X-Railway-Request-Id, or X-Request-Id when that is absent.

    When X-Railway-Edge is present, its value is added as a railway_edge log field and Sentry tag to identify the
    edge point of presence that handled the request.

    Both also carry the worker's request sequence number and uptime (as worker_request_seq and worker_uptime_seconds
    log fields, and a "worker" Sentry context).
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        worker = {
            "worker_request_seq": next(_worker_request_seq),
            "worker_uptime_seconds": round(time.monotonic() - _worker_started_at),
        }
        # Sentry's ASGI integration provides a separate isolation scope for each request.
        sentry_sdk.set_context("worker", worker)
        fields: dict[str, typing.Any] = dict(worker)

        headers = Headers(scope=scope)
        if request_id := headers.get(RAILWAY_REQUEST_ID_HEADER) or headers.get(REQUEST_ID_HEADER):
            request_id = request_id[:_MAX_HEADER_VALUE_LENGTH]
            fields["request_id"] = request_id
            sentry_sdk.set_tag("request_id", request_id)

        if railway_edge := headers.get(RAILWAY_EDGE_HEADER):
            railway_edge = railway_edge[:_MAX_HEADER_VALUE_LENGTH]
            fields["railway_edge"] = railway_edge
            sentry_sdk.set_tag("railway_edge", railway_edge)

        _request_context.set(fields)
        await self._app(scope, receive, send)
