import dataclasses
import secrets

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from xngin.apiserver import conftest, constants
from xngin.apiserver.apikeys import make_key, hash_key
from xngin.apiserver.models.tables import ApiKey, Datasource, Organization
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.main import app
from xngin.apiserver.settings import SqliteLocalConfig

CONFIG_ID_SECURED = "testing-secured"

conftest.setup(app)
client = TestClient(app)
client.base_url = str(client.base_url) + constants.API_PREFIX_V1


@pytest.fixture(scope="function")
def gen_session():
    session = next(app.dependency_overrides[xngin_db_session]())
    yield session


@dataclasses.dataclass
class SessionData:
    session: Session
    org: Organization
    ds: Datasource
    key: str


@pytest.fixture(scope="function")
def gen_datasource(gen_session):
    """Creates and returns a test datasource with its associated organization."""
    testid = secrets.token_hex(16)
    org = Organization(id="test-organization" + testid, name="test organization")
    ds = Datasource(
        id="test-datasource" + testid,
        name="test datasource",
        config=SqliteLocalConfig(
            type="sqlite_local", sqlite_filename="sqlite:///:memory:", participants=[]
        ).model_dump(),
        organization_id=org.id,
    )
    key_id, key = make_key()
    kt = ApiKey(id=key_id, key=hash_key(key), datasource_id=ds.id)
    gen_session.add(kt)
    gen_session.commit()
    return SessionData(session=gen_session, org=org, ds=ds, key=key)


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


def test_secured_correct_apikey(gen_session, gen_datasource):
    """Tests that a config with API keys allows valid API keys and rejects invalid keys."""
    key = gen_datasource.key
    response = client.get(
        "/filters?participant_type=test_participant_type",
        headers={
            constants.HEADER_CONFIG_ID: gen_datasource.ds.id,
            constants.HEADER_API_KEY: key,
        },
    )
    assert response.status_code == 200, response.content

    for bad_key in ("", key + "suffix", key.lower(), key.upper()):
        response = client.get(
            "/filters?participant_type=test_participant_type",
            headers={
                constants.HEADER_CONFIG_ID: gen_datasource.ds.id,
                constants.HEADER_API_KEY: bad_key,
            },
        )
        assert response.status_code == 403, response.content
