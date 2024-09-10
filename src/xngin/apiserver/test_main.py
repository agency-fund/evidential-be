from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from xngin.apiserver.main import app, classify_data_type
from xngin.apiserver.api_types import DataTypeClass
from xngin.apiserver.dependencies import settings_dependency
from xngin.apiserver.settings import SettingsForTesting, XnginSettings


def get_settings_for_test() -> XnginSettings:
    filename = Path(__file__).parent / "testdata/xngin.testing.settings.json"
    with open(filename) as f:
        try:
            contents = f.read()
            return TypeAdapter(SettingsForTesting).validate_json(contents)
        except ValidationError as pyve:
            print(f"Failed to parse {filename}. Contents:\n{contents}")
            raise pyve


# https://fastapi.tiangolo.com/advanced/testing-dependencies/#use-the-appdependency_overrides-attribute
app.dependency_overrides[settings_dependency] = get_settings_for_test

client = TestClient(app)


def test_get_settings_for_test():
    settings = get_settings_for_test()
    assert settings.get_client_config("customer-test").config.dwh.user == "user"
    assert settings.trusted_ips == ["testclient"]


def test_settings_can_be_overridden_by_tests():
    response = client.get("/_settings")
    assert response.status_code == 200, response.content
    settings = XnginSettings.model_validate(response.json()["settings"])
    assert settings.get_client_config("customer-test").config.dwh.user == "user"
    assert settings.trusted_ips == ["testclient"]


def test_config_injection():
    response = client.get("/_settings", headers={"Config-ID": "customer-test"})
    assert response.status_code == 200, response.content
    assert response.json()["config_id"] == "customer-test"


def test_root_get_api():
    response = client.get("/")
    assert response.status_code == 404


def test_classify_data_type():
    assert classify_data_type("foo_id", "date") == DataTypeClass.DISCRETE
    assert classify_data_type("test", "boolean") == DataTypeClass.DISCRETE
    assert classify_data_type("foo", "date") == DataTypeClass.NUMERIC
