import base64

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from xngin.apiserver import conftest
from xngin.apiserver import main as main_module
from xngin.apiserver.api_types import DataType
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.main import app
from xngin.apiserver.routers import oidc_dependencies
from xngin.apiserver.routers.admin import (
    CreateDatasourceRequest,
    CreateDatasourceResponse,
    ListDatasourcesResponse,
    UpdateDatasourceRequest,
    InspectDatasourceTableResponse,
    FieldMetadata,
    InspectDatasourceResponse,
    CreateParticipantsTypeRequest,
    CreateParticipantsTypeResponse,
    ListParticipantsTypeResponse,
    UpdateParticipantsTypeRequest,
    UpdateParticipantsTypeResponse,
)
from xngin.apiserver.routers.oidc_dependencies import (
    PRIVILEGED_TOKEN_FOR_TESTING,
    PRIVILEGED_EMAIL,
    UNPRIVILEGED_EMAIL,
    UNPRIVILEGED_TOKEN_FOR_TESTING,
)
from xngin.apiserver.settings import (
    Dsn,
    BqDsn,
    GcpServiceAccountInfo,
    SheetParticipantsRef,
    ParticipantsDef,
)
from xngin.cli.main import create_testing_dwh
from xngin.schema.schema_types import ParticipantsSchema, FieldDescriptor

conftest.setup(app)
client = TestClient(app)


@pytest.fixture(scope="module")
def db_session():
    session = next(app.dependency_overrides[xngin_db_session]())
    yield session


@pytest.fixture(scope="module", autouse=True)
def enable_apis_under_test():
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
        f"/v1/m/organizations/{secured_datasource.org.id}/members",
        json={"email": PRIVILEGED_EMAIL},
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 204, response.content

    # Add unprivileged user to existing organization
    response = client.post(
        f"/v1/m/organizations/{secured_datasource.org.id}/members",
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

    # Get participants (sheet version)
    response = client.get(
        f"/v1/m/datasources/{secured_datasource.ds.id}/participants/test_participant_type",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 200, response.content
    parsed = SheetParticipantsRef.model_validate(response.json())
    assert parsed.participant_type == "test_participant_type"
    assert parsed.sheet.worksheet == "Sheet1"

    # Create participant
    response = client.post(
        f"/v1/m/datasources/{secured_datasource.ds.id}/participants",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
        content=CreateParticipantsTypeRequest(
            participant_type="newpt",
            schema_def=ParticipantsSchema(
                table_name="newps",
                fields=[
                    FieldDescriptor(
                        field_name="newf",
                        data_type=DataType.INTEGER,
                        description="test",
                        is_unique_id=True,
                        is_strata=False,
                        is_filter=False,
                        is_metric=False,
                    )
                ],
            ),
        ).model_dump_json(),
    )
    assert response.status_code == 200, response.content
    parsed = CreateParticipantsTypeResponse.model_validate(response.json())
    assert parsed.participant_type == "newpt"

    # List participants
    response = client.get(
        f"/v1/m/datasources/{secured_datasource.ds.id}/participants",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 200, response.content
    parsed = ListParticipantsTypeResponse.model_validate(response.json())
    assert len(parsed.items) == 2, parsed

    # Update participant
    response = client.patch(
        f"/v1/m/datasources/{secured_datasource.ds.id}/participants/newpt",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
        content=UpdateParticipantsTypeRequest(
            participant_type="renamedpt"
        ).model_dump_json(),
    )
    assert response.status_code == 200
    parsed = UpdateParticipantsTypeResponse.model_validate(response.json())
    assert parsed.participant_type == "renamedpt"

    # List participants (again)
    response = client.get(
        f"/v1/m/datasources/{secured_datasource.ds.id}/participants",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 200, response.content
    parsed = ListParticipantsTypeResponse.model_validate(response.json())
    assert len(parsed.items) == 2, parsed

    # Get the named participant type
    response = client.get(
        f"/v1/m/datasources/{secured_datasource.ds.id}/participants/renamedpt",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 200, response.content
    parsed = ParticipantsDef.model_validate(response.json())
    assert parsed.participant_type == "renamedpt"

    # Delete the renamed participant type.
    response = client.delete(
        f"/v1/m/datasources/{secured_datasource.ds.id}/participants/renamedpt",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 204, response.content

    # Get the named participant type after it has been deleted
    response = client.get(
        f"/v1/m/datasources/{secured_datasource.ds.id}/participants/renamedpt",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 404, response.content

    # Delete the testing participant type.
    response = client.delete(
        f"/v1/m/datasources/{secured_datasource.ds.id}/participants/test_participant_type",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 204, response.content

    # Delete the testing participant type a 2nd time.
    response = client.delete(
        f"/v1/m/datasources/{secured_datasource.ds.id}/participants/test_participant_type",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 404, response.content

    # Delete datasources
    response = client.delete(
        f"/v1/m/datasources/{secured_datasource.ds.id}",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert response.status_code == 204, response.content


def test_create_participants_type_invalid(secured_datasource):
    response = client.post(
        f"/v1/m/datasources/{secured_datasource.ds.id}/participants",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
        content=CreateParticipantsTypeRequest.model_construct(
            participant_type="newpt",
            schema_def=ParticipantsSchema.model_construct(
                table_name="newps",
                fields=[
                    FieldDescriptor(
                        field_name="newf",
                        data_type=DataType.INTEGER,
                        description="test",
                        is_unique_id=False,
                        is_strata=False,
                        is_filter=False,
                        is_metric=False,
                    )
                ],
            ),
        ).model_dump_json(),
    )
    assert response.status_code == 422, response.content
    assert "no columns marked as unique ID." in response.json()["detail"][0]["msg"], (
        response.content
    )


@pytest.mark.pgintegration
def test_lifecycle_with_pg(secured_datasource):
    """Exercises the admin API methods that require an external database."""
    # Add the privileged user to the organization.
    response = client.post(
        f"/v1/m/organizations/{secured_datasource.org.id}/members",
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
            FieldMetadata(
                field_name="baseline_income", data_type="numeric", description=None
            ),
            FieldMetadata(
                field_name="current_income", data_type="numeric", description=None
            ),
            FieldMetadata(
                field_name="ethnicity", data_type="character varying", description=None
            ),
            FieldMetadata(
                field_name="first_name", data_type="character varying", description=None
            ),
            FieldMetadata(
                field_name="gender", data_type="character varying", description=None
            ),
            FieldMetadata(field_name="id", data_type="integer", description=None),
            FieldMetadata(field_name="income", data_type="numeric", description=None),
            FieldMetadata(
                field_name="is_engaged", data_type="boolean", description=None
            ),
            FieldMetadata(
                field_name="is_onboarded", data_type="boolean", description=None
            ),
            FieldMetadata(
                field_name="is_recruited", data_type="boolean", description=None
            ),
            FieldMetadata(
                field_name="is_registered", data_type="boolean", description=None
            ),
            FieldMetadata(
                field_name="is_retained", data_type="boolean", description=None
            ),
            FieldMetadata(
                field_name="last_name", data_type="character varying", description=None
            ),
            FieldMetadata(
                field_name="potential_0", data_type="numeric", description=None
            ),
            FieldMetadata(
                field_name="potential_1", data_type="integer", description=None
            ),
        ],
    )
