import base64

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from xngin.apiserver import conftest
from xngin.apiserver import main as main_module
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.main import app
from xngin.apiserver.routers import oidc_dependencies
from xngin.apiserver.routers.admin import (
    CreateDatasourceRequest,
    CreateDatasourceResponse,
    ListDatasourcesResponse,
    UpdateDatasourceRequest,
    InspectDatasourceTableResponse,
    FieldDescription,
    InspectDatasourceResponse,
)
from xngin.apiserver.routers.oidc_dependencies import (
    PRIVILEGED_TOKEN_FOR_TESTING,
    PRIVILEGED_EMAIL,
    UNPRIVILEGED_EMAIL,
    UNPRIVILEGED_TOKEN_FOR_TESTING,
)
from xngin.apiserver.settings import Dsn, BqDsn, GcpServiceAccountInfo
from xngin.cli.main import create_testing_dwh

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
    """Exercises the admin API methods that can operate purely in-process w/o an external database."""
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
            # These settings correspond to the Postgres spun up in GHA or via localpg.py.
            dwh=Dsn(
                driver="postgresql+psycopg",
                host="127.0.0.1",
                user="postgres",
                port=5499,
                password=SecretStr("postgres"),
                dbname="postgres",
                sslmode="disable",
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
    assert {i.driver for i in parsed.items} == {
        "postgresql+psycopg2",
        "postgresql+psycopg",
    }
    assert parsed.items[0].organization_id == parsed.items[1].organization_id

    # Update datasource name
    response = client.patch(
        f"/v1/m/datasources/{datasource_id}",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
        content=UpdateDatasourceRequest(
            name="updated name",
        ).model_dump_json(),
    )
    assert response.status_code == 204, response.content

    # List datasources to confirm update
    response = client.get(
        "/v1/m/datasources",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 200, response.content
    assert "updated name" in {i["name"] for i in response.json()["items"]}, (
        response.json()
    )

    # Update DWH on the datasource
    response = client.patch(
        f"/v1/m/datasources/{datasource_id}",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
        content=UpdateDatasourceRequest(
            dwh=BqDsn(
                driver="bigquery",
                project_id="1234",
                dataset_id="ds",
                credentials=GcpServiceAccountInfo(
                    type="serviceaccountinfo",
                    content_base64=base64.b64encode(b"key").decode(),
                ),
            )
        ).model_dump_json(),
    )
    assert response.status_code == 204, response.content

    # List datasources to confirm update
    response = client.get(
        "/v1/m/datasources",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 200, response.content
    assert "bigquery" in {i["driver"] for i in response.json()["items"]}, (
        response.json()
    )

    # Delete datasources
    response = client.delete(
        f"/v1/m/datasources/{secured_datasource.ds.id}",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 204, response.content


@pytest.mark.pgintegration
def test_lifecycle_with_pg(secured_datasource):
    """Exercises the admin API methods that require an external database."""
    # Add the privileged user to the organization.
    response = client.post(
        f"/v1/m/organization/{secured_datasource.org.id}/members",
        json={"email": PRIVILEGED_EMAIL},
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 204, response.content

    # Populate the testing data warehouse.
    create_testing_dwh(dsn=secured_datasource.dsn, nrows=100)

    # Inspect the datasource.
    response = client.post(
        f"/v1/m/datasources/{secured_datasource.ds.id}/inspect",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 200, response.content
    parsed = InspectDatasourceResponse.model_validate(response.json())
    assert parsed.tables == ["dwh"], response.json()

    # Inspect one table in the datasource.
    response = client.post(
        f"/v1/m/datasources/{secured_datasource.ds.id}/inspect/dwh",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 200, response.content
    parsed = InspectDatasourceTableResponse.model_validate(response.json())
    assert parsed == InspectDatasourceTableResponse(
        detected_unique_id_fields=[],
        fields=[
            FieldDescription(
                field_name="baseline_income", data_type="numeric", description=None
            ),
            FieldDescription(
                field_name="current_income", data_type="numeric", description=None
            ),
            FieldDescription(
                field_name="ethnicity", data_type="character varying", description=None
            ),
            FieldDescription(
                field_name="first_name", data_type="character varying", description=None
            ),
            FieldDescription(
                field_name="gender", data_type="character varying", description=None
            ),
            FieldDescription(field_name="id", data_type="integer", description=None),
            FieldDescription(
                field_name="income", data_type="numeric", description=None
            ),
            FieldDescription(
                field_name="is_engaged", data_type="boolean", description=None
            ),
            FieldDescription(
                field_name="is_onboarded", data_type="boolean", description=None
            ),
            FieldDescription(
                field_name="is_recruited", data_type="boolean", description=None
            ),
            FieldDescription(
                field_name="is_registered", data_type="boolean", description=None
            ),
            FieldDescription(
                field_name="is_retained", data_type="boolean", description=None
            ),
            FieldDescription(
                field_name="last_name", data_type="character varying", description=None
            ),
            FieldDescription(
                field_name="potential_0", data_type="numeric", description=None
            ),
            FieldDescription(
                field_name="potential_1", data_type="integer", description=None
            ),
        ],
    )
