from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from .main import app, classify_data_type, DataTypeClass
from .dependencies import settings_dependency
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


def test_classify_data_type():
    assert classify_data_type("foo_id", "date") == DataTypeClass.DISCRETE
    assert classify_data_type("test", "boolean") == DataTypeClass.DISCRETE
    assert classify_data_type("foo", "date") == DataTypeClass.NUMERIC
