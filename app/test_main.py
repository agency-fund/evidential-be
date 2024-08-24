from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from .main import app, settings_dependency
from .settings import SettingsForTesting


def get_settings_for_test():
    with open("xngin.testing.settings.json") as f:
        return TypeAdapter(SettingsForTesting).validate_json(f.read())


# https://fastapi.tiangolo.com/advanced/testing-dependencies/#use-the-appdependency_overrides-attribute
app.dependency_overrides[settings_dependency] = get_settings_for_test

client = TestClient(app)


def test_get_settings_for_test():
    settings = get_settings_for_test()
    assert settings.customer.dwh.user == "user"


def test_settings_api():
    response = client.get("/_settings")
    assert response.status_code == 200, response.content
    assert response.json()["settings"]["customer"]["dwh"]["user"] == "user"


def test_root_get_api():
    response = client.get("/")
    assert response.status_code == 404
