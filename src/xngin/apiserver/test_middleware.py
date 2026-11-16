import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from xngin.apiserver import flags, middleware

FORWARDED_HEADERS = {"X-Forwarded-For": "203.0.113.9", "X-Forwarded-Proto": "https"}


def _client_reporting_app() -> FastAPI:
    app = FastAPI()
    middleware.setup(app)

    @app.get("/client")
    def client(request: Request):
        return {"host": request.client.host if request.client else None, "scheme": request.url.scheme}

    return app


@pytest.mark.parametrize(
    ("forwarded_allow_ips", "expected"),
    [
        pytest.param("", {"host": "testclient", "scheme": "http"}, id="unset-ignores-headers"),
        pytest.param("10.0.0.1", {"host": "testclient", "scheme": "http"}, id="other-peer-ignored"),
        # Starlette's TestClient connects as the peer "testclient".
        pytest.param("10.0.0.1,testclient", {"host": "203.0.113.9", "scheme": "https"}, id="listed-peer-trusted"),
        pytest.param("*", {"host": "203.0.113.9", "scheme": "https"}, id="any-peer-trusted"),
    ],
)
def test_forwarded_headers_are_trusted_only_from_allowed_peers(monkeypatch, forwarded_allow_ips, expected):
    monkeypatch.setattr(flags, "FORWARDED_ALLOW_IPS", forwarded_allow_ips)

    response = TestClient(_client_reporting_app()).get("/client", headers=FORWARDED_HEADERS)

    assert response.json() == expected
