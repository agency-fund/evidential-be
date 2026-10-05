import asyncio

import pytest
from loguru import logger
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette.types import Message

from xngin.apiserver import request_context_middleware
from xngin.apiserver.request_context_middleware import (
    RAILWAY_EDGE_HEADER,
    RAILWAY_REQUEST_ID_HEADER,
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
)


def _handler(_request):
    logger.info("handled")
    return PlainTextResponse("ok")


def _failing_handler(_request):
    raise RuntimeError("boom")


class _LogsLikeUvicorn:
    """Logs at the points where Uvicorn does.

    Uvicorn writes the access line when the response headers are sent ("access"), which is normally inside the
    application. For an unhandled exception, Starlette's ServerErrorMiddleware sends the 500 response, and Uvicorn logs
    the traceback ("after app"), only after the exception has propagated out of the application's middleware.
    """

    def __init__(self, app):
        self._app = app

    async def __call__(self, scope, receive, send):
        async def logging_send(message):
            if message["type"] == "http.response.start":
                logger.info("access")
            await send(message)

        try:
            await self._app(scope, receive, logging_send)
        finally:
            if scope["type"] == "http":
                logger.info("after app")


async def _receive() -> Message:
    return {"type": "http.request"}


async def _send(_message: Message) -> None:
    pass


@pytest.fixture
def client():
    app = Starlette(routes=[Route("/", _handler), Route("/boom", _failing_handler)])
    app.add_middleware(RequestContextMiddleware)
    return TestClient(_LogsLikeUvicorn(app), raise_server_exceptions=False)


@pytest.fixture
def records():
    logger.configure(patcher=request_context_middleware.add_request_context)
    captured = []
    handler_id = logger.add(lambda message: captured.append(message.record), level="INFO")
    yield captured
    logger.remove(handler_id)


@pytest.fixture
def sentry_tags(monkeypatch):
    tags: dict[str, str] = {}
    monkeypatch.setattr(request_context_middleware.sentry_sdk, "set_tag", tags.__setitem__)
    return tags


def _with_message(records, message):
    return [r for r in records if r["message"] == message]


def _handled(records):
    return _with_message(records, "handled")


def test_request_context_is_added_to_logs_and_sentry(client, records, sentry_tags):
    response = client.get("/", headers={RAILWAY_REQUEST_ID_HEADER: "abc", RAILWAY_EDGE_HEADER: "iad1"})

    assert response.status_code == 200
    (handled,) = _handled(records)
    (access,) = _with_message(records, "access")
    for record in (handled, access):
        assert record["extra"]["request_id"] == "abc"
        assert record["extra"]["railway_edge"] == "iad1"
    assert sentry_tags == {"request_id": "abc", "railway_edge": "iad1"}


def test_request_id_falls_back_to_x_request_id(client, records, sentry_tags):
    client.get("/", headers={REQUEST_ID_HEADER: "from-load-balancer"})

    (record,) = _handled(records)
    assert record["extra"]["request_id"] == "from-load-balancer"
    assert sentry_tags == {"request_id": "from-load-balancer"}


def test_railway_request_id_is_preferred(client, records, sentry_tags):
    client.get("/", headers={RAILWAY_REQUEST_ID_HEADER: "from-railway", REQUEST_ID_HEADER: "from-client"})

    (record,) = _handled(records)
    assert record["extra"]["request_id"] == "from-railway"


def test_request_context_survives_an_unhandled_exception(client, records, sentry_tags):
    response = client.get("/boom", headers={RAILWAY_REQUEST_ID_HEADER: "abc", RAILWAY_EDGE_HEADER: "iad1"})

    assert response.status_code == 500
    (access,) = _with_message(records, "access")
    (after,) = _with_message(records, "after app")
    for record in (access, after):
        assert record["extra"]["request_id"] == "abc"
        assert record["extra"]["railway_edge"] == "iad1"
    assert sentry_tags == {"request_id": "abc", "railway_edge": "iad1"}


def test_request_context_does_not_leak_into_later_requests(client, records, sentry_tags):
    client.get("/", headers={RAILWAY_REQUEST_ID_HEADER: "abc", RAILWAY_EDGE_HEADER: "iad1"})
    sentry_tags.clear()
    response = client.get("/")
    logger.info("after")

    assert response.status_code == 200
    assert sentry_tags == {}

    first, second = _handled(records)
    assert first["extra"]["request_id"] == "abc"
    assert "request_id" not in second["extra"]
    (after,) = [r for r in records if r["message"] == "after"]
    assert "request_id" not in after["extra"]
    assert first["extra"]["railway_edge"] == "iad1"
    assert "railway_edge" not in second["extra"]
    assert "railway_edge" not in after["extra"]


@pytest.mark.parametrize(
    ("header", "field"), [(RAILWAY_REQUEST_ID_HEADER, "request_id"), (RAILWAY_EDGE_HEADER, "railway_edge")]
)
def test_header_values_are_truncated(client, records, sentry_tags, header, field):
    client.get("/", headers={header: "x" * 1000})

    (record,) = _handled(records)
    assert record["extra"][field] == "x" * 64
    assert sentry_tags == {field: "x" * 64}


def test_reused_task_does_not_carry_over_request_fields(records, sentry_tags):
    async def app(_scope, _receive, _send):
        logger.info("handled")

    middleware = RequestContextMiddleware(app)

    async def two_requests_in_one_task():
        await middleware({"type": "http", "headers": [(b"x-request-id", b"first")]}, _receive, _send)
        await middleware({"type": "http", "headers": []}, _receive, _send)

    asyncio.run(two_requests_in_one_task())

    first, second = _handled(records)
    assert first["extra"]["request_id"] == "first"
    assert "request_id" not in second["extra"]


def test_overlapping_requests_keep_their_own_request_ids(records, sentry_tags):
    # Neither request logs until both have set their context.
    both_started = asyncio.Barrier(2)

    async def app(scope, _receive, _send):
        await both_started.wait()
        logger.info("handled", path=scope["path"])

    middleware = RequestContextMiddleware(app)
    request_ids = {"/a": "request-a", "/b": "request-b"}
    scopes = [
        {"type": "http", "path": path, "headers": [(b"x-request-id", request_id.encode())]}
        for path, request_id in request_ids.items()
    ]

    async def overlapping_requests():
        # Like Uvicorn, run each request in its own task.
        await asyncio.gather(*(middleware(scope, _receive, _send) for scope in scopes))

    asyncio.run(overlapping_requests())

    handled = _handled(records)
    assert {r["extra"]["path"]: r["extra"]["request_id"] for r in handled} == request_ids
