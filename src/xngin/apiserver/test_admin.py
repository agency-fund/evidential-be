import dataclasses
import secrets

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from xngin.apiserver import conftest
from xngin.apiserver import main as main_module
from xngin.apiserver.apikeys import make_key, hash_key
from xngin.apiserver.conftest import get_settings_for_test
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.main import app
from xngin.apiserver.models.tables import ApiKey, Organization, Datasource
from xngin.apiserver.routers import oidc_dependencies
from xngin.apiserver.routers.admin import (
    CreateDatasourceRequest,
    CreateDatasourceResponse,
    ListDatasourcesResponse,
)
from xngin.apiserver.routers.oidc_dependencies import (
    PRIVILEGED_TOKEN_FOR_TESTING,
    PRIVILEGED_EMAIL,
    UNPRIVILEGED_EMAIL,
    UNPRIVILEGED_TOKEN_FOR_TESTING,
)
from xngin.apiserver.settings import RemoteDatabaseConfig, Dsn

conftest.setup(app)
client = TestClient(app)


@pytest.fixture(scope="module")
def db_session():
    session = next(app.dependency_overrides[xngin_db_session]())
    yield session


@pytest.fixture(scope="module", autouse=True)
def set_env_foo():
    # TODO: Calling enable_*() has the minor side effect of enabling these APIs for all subsequent unit tests.
    main_module.enable_oidc_api()
    main_module.enable_admin_api()
    oidc_dependencies.TESTING_TOKENS_ENABLED = True


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
    # TODO: replace this with a non-sqliteconfig value.
    test_ds = get_settings_for_test().get_datasource("testing").config

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


def test_list_orgs_unauthenticated():
    response = client.get("/v1/m/organizations")
    assert response.status_code == 403, response.content


def test_list_orgs_privileged(secured_datasource):
    response = client.get(
        "/v1/m/organizations",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 200, response.content


def test_list_orgs_unprivileged(secured_datasource):
    response = client.get(
        "/v1/m/organizations",
        headers={"Authorization": f"Bearer {UNPRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 403, response.content


def test_lifecycle(secured_datasource):
    # Add privileged user to existing organization
    response = client.post(
        f"/v1/m/organization/{secured_datasource.org.id}/members",
        json={"email": PRIVILEGED_EMAIL},
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 204, response.content

    # Add unprivileged user to existing organization
    response = client.post(
        f"/v1/m/organization/{secured_datasource.org.id}/members",
        json={"email": UNPRIVILEGED_EMAIL},
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 204, response.content

    # List organizations
    response = client.get(
        "/v1/m/organizations",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 200, response.content
    assert response.json()["items"][0]["id"] == secured_datasource.org.id

    # Create datasource
    response = client.post(
        "/v1/m/datasources",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
        content=CreateDatasourceRequest(
            organization_id=secured_datasource.org.id,
            name="test remote ds",
            config=RemoteDatabaseConfig(
                type="remote",
                dwh=Dsn(
                    driver="postgresql+psycopg",
                    host="db.example.com",
                    user="u",
                    password=SecretStr("p"),
                    dbname="db",
                ),
                participants=[],
            ),
        ).model_dump_json(),
    )
    assert response.status_code == 200, response.content
    parsed = CreateDatasourceResponse.model_validate(response.json())
    datasource_id = parsed.id

    # List datasources
    response = client.get(
        "/v1/m/datasources",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 200, response.content
    parsed = ListDatasourcesResponse.model_validate(response.json())
    assert len(parsed.items) == 2
    assert {i.driver for i in parsed.items} == {"postgresql+psycopg", "sqlite"}
    assert parsed.items[0].organization_id == parsed.items[1].organization_id

    # Delete datasources
    response = client.delete(
        f"/v1/m/datasources/{datasource_id}",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 204, response.content
