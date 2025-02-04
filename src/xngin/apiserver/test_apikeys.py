from fastapi.testclient import TestClient
from sqlalchemy import delete

from xngin.apiserver import conftest, constants
from xngin.apiserver.apikeys import make_key, hash_key
from xngin.apiserver.models.tables import ApiKeyTable, ApiKeyDatasourceTable
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.main import app

CONFIG_ID_SECURED = "testing-secured"


conftest.setup(app)
client = TestClient(app)
client.base_url = str(client.base_url) + constants.API_PREFIX_V1


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


def test_secured_correct_apikey():
    """Tests that a config with API keys allows valid API keys and rejects invalid keys."""
    session = next(app.dependency_overrides[xngin_db_session]())  # hack
    # emulates behavior of POST /m/apikeys
    id_, key = make_key()
    api_key = ApiKeyTable(id=id_, key=hash_key(key))
    api_key.datasources = [ApiKeyDatasourceTable(datasource_id=CONFIG_ID_SECURED)]
    session.add(api_key)
    session.commit()
    try:
        response = client.get(
            "/filters?participant_type=test_participant_type",
            headers={
                constants.HEADER_CONFIG_ID: CONFIG_ID_SECURED,
                constants.HEADER_API_KEY: key,
            },
        )
        assert response.status_code == 200, response.content

        for bad_key in ("", key + "suffix", key.lower(), key.upper()):
            response = client.get(
                "/filters?participant_type=test_participant_type",
                headers={
                    constants.HEADER_CONFIG_ID: CONFIG_ID_SECURED,
                    constants.HEADER_API_KEY: bad_key,
                },
            )
            assert response.status_code == 403, response.content
    finally:
        assert (
            session.execute(delete(ApiKeyTable).where(ApiKeyTable.id == id_)).rowcount
            == 1
        )
