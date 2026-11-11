from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from xngin.apiserver import exceptionhandlers
from xngin.apiserver.exceptions_common import DwhPermissionDeniedError
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
