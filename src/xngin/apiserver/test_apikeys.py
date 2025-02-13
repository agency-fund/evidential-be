import pytest
from fastapi.testclient import TestClient

from xngin.apiserver import conftest, constants
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.main import app

# CONFIG_ID_SECURED refers to a datasource defined in xngin.testing.settings.json
CONFIG_ID_SECURED = "testing-secured"

conftest.setup(app)
client = TestClient(app)
client.base_url = str(client.base_url) + constants.API_PREFIX_V1


@pytest.fixture(scope="module")
def db_session():
    session = next(app.dependency_overrides[xngin_db_session]())
    yield session


def test_secured_requires_apikey():
    """Tests that a config marked as requiring an API key rejects requests w/o API keys."""
    response = client.get(
        "/filters?participant_type=test_participant_type",
        headers={constants.HEADER_CONFIG_ID: CONFIG_ID_SECURED},
    )
    assert response.status_code == 403, response.content


def test_secured_wrong_apikey():
    """Tests that a config marked as requiring an API key rejects requests when no API keys defined."""
    response = client.get(
        "/filters?participant_type=test_participant_type",
        headers={
            constants.HEADER_CONFIG_ID: CONFIG_ID_SECURED,
            constants.HEADER_API_KEY: "nonexistent",
        },
    )
    assert response.status_code == 403, response.content


def test_secured_correct_apikey(secured_datasource):
    """Tests that a config with API keys allows valid API keys and rejects invalid keys."""
    response = client.get(
        "/filters?participant_type=test_participant_type",
        headers={
            constants.HEADER_CONFIG_ID: secured_datasource.ds.id,
            constants.HEADER_API_KEY: secured_datasource.key,
        },
    )
    assert response.status_code == 200, response.content

    for bad_key in (
        "",
        secured_datasource.key + "suffix",
        secured_datasource.key.lower(),
        secured_datasource.key.upper(),
    ):
        response = client.get(
            "/filters?participant_type=test_participant_type",
            headers={
                constants.HEADER_CONFIG_ID: secured_datasource.ds.id,
                constants.HEADER_API_KEY: bad_key,
            },
        )
        assert response.status_code == 403, response.content
