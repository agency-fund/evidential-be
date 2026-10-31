from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from xngin.apiserver import exceptionhandlers


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
