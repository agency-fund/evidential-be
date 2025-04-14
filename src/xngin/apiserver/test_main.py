from fastapi.testclient import TestClient

import pytest
from xngin.apiserver import conftest, constants
from xngin.apiserver.api_types import DataType
from xngin.apiserver.main import app
from xngin.apiserver.routers.experiments_api import generate_field_descriptors
from xngin.apiserver.settings import (
    CannotFindParticipantsError,
    XnginSettings,
    infer_table,
)

conftest.setup(app)
client = TestClient(app)
client.base_url = str(client.base_url) + constants.API_PREFIX_V1


def test_get_settings_for_test():
    settings = conftest.get_settings_for_test()
    assert settings.get_datasource("customer-test").config.dwh.user == "user"
    assert settings.trusted_ips == ["testclient"]


def test_settings_can_be_overridden_by_tests():
    response = client.get("/_settings", headers={constants.HEADER_CONFIG_ID: "testing"})
    assert response.status_code == 200, response.content
    settings = XnginSettings.model_validate(response.json()["settings"])
    assert settings.get_datasource("customer-test").config.dwh.user == "user"
    assert settings.trusted_ips == ["testclient"]


def test_config_injection():
    response = client.get(
        "/_settings", headers={constants.HEADER_CONFIG_ID: "customer-test"}
    )
    assert response.status_code == 200, response.content
    assert response.json()["config_id"] == "customer-test"


def test_root_get_api():
    response = client.get("/")
    assert response.status_code == 404


def test_generate_column_descriptors():
    settings = conftest.get_settings_for_test()
    config = settings.get_datasource("testing").config
    with config.dbsession() as session:
        sa_table = infer_table(session.get_bind(), "dwh", config.supports_reflection())

    db_schema = generate_field_descriptors(sa_table, "last_name")

    # Check a few columns:
    assert db_schema["gender"].field_name == "gender"
    assert db_schema["gender"].data_type == DataType.CHARACTER_VARYING
    assert db_schema["gender"].description == ""
    assert db_schema["gender"].is_unique_id is False
    assert db_schema["gender"].is_strata is False
    assert db_schema["gender"].is_filter is False
    assert db_schema["gender"].is_metric is False
    assert db_schema["gender"].extra is None  # only necessary info loaded
    assert db_schema["last_name"].field_name == "last_name"
    assert db_schema["last_name"].data_type == DataType.CHARACTER_VARYING
    # Next assertion ust because we labeled it that way in settings!
    assert db_schema["last_name"].is_unique_id
    assert db_schema["current_income"].field_name == "current_income"
    assert db_schema["current_income"].data_type == DataType.DOUBLE_PRECISION
    assert db_schema["current_income"].is_unique_id is False
    assert db_schema["is_recruited"].field_name == "is_recruited"
    # sqlite stores booleans as ints, so:
    assert db_schema["is_recruited"].data_type == DataType.INTEGER
    assert db_schema["is_recruited"].is_unique_id is False


def test_find_participants_raises_exception_for_invalid_participant_type():
    settings = conftest.get_settings_for_test()
    config = settings.get_datasource("testing").config
    with pytest.raises(CannotFindParticipantsError):
        config.find_participants("bad_type")
