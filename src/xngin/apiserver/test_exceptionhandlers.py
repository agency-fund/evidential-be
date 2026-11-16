from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from xngin.apiserver import exceptionhandlers
from xngin.apiserver.exceptions_common import DwhPermissionDeniedError, DwhQueryError
from xngin.apiserver.request_context_middleware import REQUEST_ID_HEADER, RequestContextMiddleware


def test_dwh_errors_answer_with_their_status_and_the_dwh_error_response_shape():
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    exceptionhandlers.setup(app)

    @app.get("/")
    def denied():
        raise DwhPermissionDeniedError("msg", hint="h", table_name="t")

    response = TestClient(app).get("/", headers={REQUEST_ID_HEADER: "req-1"})

    assert response.status_code == 424
    assert response.json() == {
        "message": "msg",
        "code": "dwh_permission_denied",
        "hint": "h",
        "hint_command": None,
        "dwh_message": None,
        "table_name": "t",
        "request_id": "req-1",
    }


def test_dwh_errors_log_the_warehouse_message_and_table_name():
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    exceptionhandlers.setup(app)

    @app.get("/")
    def failed():
        raise DwhQueryError(
            "BigQuery rejected the request (quotaExceeded)",
            dwh_message="Quota exceeded: {project}",
            table_name="dataset.participants",
        ) from ValueError("driver says {braces}")

    records = []
    handler_id = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        response = TestClient(app).get("/", headers={REQUEST_ID_HEADER: "req-1"})
    finally:
        logger.remove(handler_id)

    assert response.status_code == 424
    (record,) = records
    assert record["extra"]["dwh_code"] == "dwh_query_failed"
    assert record["extra"]["dwh_message"] == "Quota exceeded: {project}"
    assert record["extra"]["table_name"] == "dataset.participants"
    assert record["extra"]["cause"] == "ValueError"
    assert record["extra"]["request_id"] == "req-1"
    assert "Quota exceeded: {project}" in record["message"]
    assert "dataset.participants" in record["message"]
    # The driver's exception text may contain SQL and bind parameters, so it must not be logged.
    assert "driver says" not in record["message"]
    assert record["exception"] is None


def test_unhandled_exceptions_are_logged_with_the_exception_and_return_500():
    app = FastAPI()
    exceptionhandlers.setup(app)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("kaboom")

    records = []
    handler_id = logger.add(lambda message: records.append(message.record), level="ERROR")
    try:
        response = TestClient(app, raise_server_exceptions=False).get("/boom")
    finally:
        logger.remove(handler_id)

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    (record,) = records
    assert record["exception"].type is RuntimeError
    assert record["extra"]["http_method"] == "GET"
    assert record["extra"]["http_path"] == "/boom"
