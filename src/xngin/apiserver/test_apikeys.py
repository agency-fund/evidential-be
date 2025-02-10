import dataclasses
import secrets

import pytest
from fastapi.testclient import TestClient

from xngin.apiserver import conftest, constants
from xngin.apiserver.apikeys import make_key, hash_key
from xngin.apiserver.conftest import get_settings_for_test
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.main import app
from xngin.apiserver.models.tables import ApiKey, Datasource, Organization

# CONFIG_ID_SECURED refers to a datasource defined in xngin.testing.settings.json
CONFIG_ID_SECURED = "testing-secured"

conftest.setup(app)
client = TestClient(app)
client.base_url = str(client.base_url) + constants.API_PREFIX_V1


@pytest.fixture(scope="module")
def db_session():
    session = next(app.dependency_overrides[xngin_db_session]())
    yield session


@dataclasses.dataclass
class DatasourceMetadata:
    """Describes an ephemeral datasource, organization, and API key."""

    org: Organization
    ds: Datasource
    key: str


@pytest.fixture(scope="function")
def secured_datasource(db_session):
    """Creates a new test datasource with its associated organization."""
    run_id = secrets.token_hex(8)
    datasource_id = "ds" + run_id

    # We derive a new test datasource from the standard static "testing" datasource by
    # randomizing its unique ID and marking it as requiring an API key.
    test_ds = get_settings_for_test().get_datasource("testing")
    test_ds.id = datasource_id
    test_ds.require_api_key = True

    org = Organization(id="org" + run_id, name="test organization")

    datasource = Datasource(
        id=datasource_id, name="test ds", config=test_ds.model_dump()
    )
    datasource.organization = org

    key_id, key = make_key()
    kt = ApiKey(id=key_id, key=hash_key(key))
    kt.datasource = datasource

    db_session.add_all([org, datasource, kt])
    db_session.commit()
    assert db_session.get(Datasource, datasource_id) is not None
    return DatasourceMetadata(org=org, ds=datasource, key=key)


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
