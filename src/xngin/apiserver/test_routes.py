import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from xngin.apiserver import routes
from xngin.apiserver.main import app


def test_redoc_is_disabled():
    assert TestClient(app).get("/redoc").status_code == 404


def test_swagger_ui_is_served_in_development():
    response = TestClient(app).get(routes.SWAGGER_UI_PATH)
    assert response.status_code == 200
    assert "swagger-ui" in response.text


def test_docs_redirects_to_hosted_docs_outside_development(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    production_app = FastAPI(docs_url=None, redoc_url=None)
    routes.register(production_app)
    response = TestClient(production_app).get(routes.SWAGGER_UI_PATH, follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == routes.HOSTED_API_DOCS_URL
