from fastapi.testclient import TestClient

import pytest
from xngin.apiserver import conftest
from xngin.apiserver.api_types import DataType
from xngin.apiserver.main import app, generate_column_descriptors
from xngin.apiserver.settings import (
    CannotFindParticipantsException,
    XnginSettings,
    infer_table,
)

conftest.setup(app)
client = TestClient(app)


def test_get_settings_for_test():
    settings = conftest.get_settings_for_test()
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


def test_generate_column_descriptors():
    settings = conftest.get_settings_for_test()
    config = settings.get_client_config("testing").config
    with config.dbsession() as session:
        sa_table = infer_table(
            session.get_bind(), "test_participant_type", config.supports_reflection()
        )

    db_schema = generate_column_descriptors(sa_table, "last_name")

    # Check a few columns:
    assert db_schema["gender"].column_name == "gender"
    assert db_schema["gender"].data_type == DataType.CHARACTER_VARYING
    assert db_schema["gender"].column_group == ""
    assert db_schema["gender"].description == ""
    assert not db_schema["gender"].is_unique_id
    assert not db_schema["gender"].is_strata
    assert not db_schema["gender"].is_filter
    assert not db_schema["gender"].is_metric
    assert db_schema["last_name"].column_name == "last_name"
    assert db_schema["last_name"].data_type == DataType.CHARACTER_VARYING
    # Next assertion ust because we labeled it that way in settings!
    assert db_schema["last_name"].is_unique_id
    assert db_schema["current_income"].column_name == "current_income"
    assert db_schema["current_income"].data_type == DataType.DOUBLE_PRECISION
    assert not db_schema["current_income"].is_unique_id
    assert db_schema["is_recruited"].column_name == "is_recruited"
    assert (
        db_schema["is_recruited"].data_type == DataType.INTEGER
    )  # sqlite stores booleans as ints
    assert not db_schema["is_recruited"].is_unique_id

    # Now confirm that if no column was marked, we use a default id.
    db_schema = generate_column_descriptors(sa_table)
    assert db_schema["id"].is_unique_id


def test_find_participants_raises_exception_for_invalid_participant_type():
    settings = conftest.get_settings_for_test()
    config = settings.get_client_config("testing").config
    with pytest.raises(CannotFindParticipantsException):
        config.find_participants("bad_type")
