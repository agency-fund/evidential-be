import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from xngin.apiserver import routes
from xngin.apiserver.main import app
from xngin.apiserver.openapi import custom_openapi, humane_operation_id


def test_redoc_is_disabled():
    assert TestClient(app).get("/redoc").status_code == 404


def test_swagger_ui_is_served_in_development():
    response = TestClient(app).get(routes.SWAGGER_UI_PATH)
    assert response.status_code == 200
    assert "swagger-ui" in response.text


@pytest.mark.parametrize(
    ("environment", "path", "expected_location"),
    [
        ("dev", "/", routes.SWAGGER_UI_PATH),
        ("production", "/", routes.HOSTED_API_DOCS_URL),
        ("production", routes.SWAGGER_UI_PATH, routes.HOSTED_API_DOCS_URL),
    ],
)
def test_docs_redirects(
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
    path: str,
    expected_location: str,
):
    monkeypatch.setenv("ENVIRONMENT", environment)
    test_app = FastAPI(
        docs_url=routes.SWAGGER_UI_PATH if environment == "dev" else None,
        redoc_url=None,
        generate_unique_id_function=humane_operation_id,
    )
    routes.register(test_app)

    response = TestClient(test_app).get(path, follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == expected_location

    schema = custom_openapi(test_app)
    assert path not in schema["paths"]
