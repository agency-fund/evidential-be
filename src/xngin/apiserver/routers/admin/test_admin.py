import base64
import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import flags
from xngin.apiserver.conftest import delete_seeded_users
from xngin.apiserver.dns import safe_resolve
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.dwh.inspection_types import FieldDescriptor, ParticipantsSchema
from xngin.apiserver.models import tables
from xngin.apiserver.routers.admin.admin_api import user_from_token
from xngin.apiserver.routers.admin.admin_api_types import (
    AddWebhookToOrganizationRequest,
    AddWebhookToOrganizationResponse,
    CreateApiKeyResponse,
    CreateDatasourceRequest,
    CreateDatasourceResponse,
    CreateOrganizationRequest,
    CreateOrganizationResponse,
    CreateParticipantsTypeRequest,
    CreateParticipantsTypeResponse,
    DatasourceSummary,
    FieldMetadata,
    GetOrganizationResponse,
    InspectDatasourceResponse,
    InspectDatasourceTableResponse,
    ListApiKeysResponse,
    ListDatasourcesResponse,
    ListOrganizationsResponse,
    ListParticipantsTypeResponse,
    ListWebhooksResponse,
    UpdateDatasourceRequest,
    UpdateOrganizationWebhookRequest,
    UpdateParticipantsTypeRequest,
    UpdateParticipantsTypeResponse,
)
from xngin.apiserver.routers.auth.auth_dependencies import (
    PRIVILEGED_EMAIL,
    PRIVILEGED_TOKEN_FOR_TESTING,
    TESTING_TOKENS,
    UNPRIVILEGED_EMAIL,
    UNPRIVILEGED_TOKEN_FOR_TESTING,
)
from xngin.apiserver.routers.auth.principal import Principal
from xngin.apiserver.routers.common_api_types import (
    Arm,
    CreateExperimentRequest,
    CreateExperimentResponse,
    DataType,
    DesignSpecMetricRequest,
    ExperimentsType,
    FreqExperimentAnalysis,
    GetExperimentAssignmentsResponse,
    GetParticipantAssignmentResponse,
    ListExperimentsResponse,
    PreassignedFrequentistExperimentSpec,
)
from xngin.apiserver.routers.common_enums import ExperimentState, StopAssignmentReason
from xngin.apiserver.routers.experiments.test_experiments_common import (
    insert_experiment_and_arms,
    make_create_online_experiment_request,
    make_create_preassigned_experiment_request,
    make_createexperimentrequest_json,
    make_insertable_experiment,
)
from xngin.apiserver.settings import (
    BqDsn,
    Dsn,
    GcpServiceAccountInfo,
    NoDwh,
    ParticipantsDef,
)
from xngin.apiserver.storage.bootstrap import (
    DEFAULT_DWH_SOURCE_NAME,
    DEFAULT_NO_DWH_SOURCE_NAME,
    DEFAULT_ORGANIZATION_NAME,
)
from xngin.apiserver.testing.assertions import assert_dates_equal
from xngin.cli.main import create_testing_dwh

SAMPLE_GCLOUD_SERVICE_ACCOUNT_KEY = {
    "auth_provider_x509_cert_url": "",
    "auth_uri": "",
    "client_email": "",
    "client_id": "",
    "client_x509_cert_url": "",
    "private_key": "",
    "private_key_id": "",
    "project_id": "",
    "token_uri": "",
    "type": "service_account",
    "universe_domain": "googleapis.com",
}


def find_ds_with_name(
    datasources: list[tables.Datasource] | list[DatasourceSummary], name: str
) -> tables.Datasource | DatasourceSummary:
    """Helper function to find a datasource with a specific name from an iterable.

    Raises StopIteration if the datasource is not found.
    """
    return next(ds for ds in datasources if ds.name == name)


@pytest.fixture(name="testing_experiment")
async def fixture_testing_experiment(
    xngin_session: AsyncSession, testing_datasource_with_user
):
    """Create an experiment on a test inline schema datasource with proper user permissions."""
    datasource = testing_datasource_with_user.ds
    experiment = await insert_experiment_and_arms(
        xngin_session, datasource, ExperimentsType.FREQ_PREASSIGNED
    )
    # Add fake assignments for each arm for real participant ids in our test data.
    arm_ids = [arm.id for arm in experiment.arms]
    # NOTE: id = 0 doesn't exist in the test data, so we'll have 1 missing participant.
    for i in range(10):
        assignment = tables.ArmAssignment(
            experiment_id=experiment.id,
            participant_id=str(i),
            participant_type=experiment.participant_type,
            arm_id=arm_ids[i % 2],  # Alternate between the two arms
            strata=[],
        )
        xngin_session.add(assignment)
    await xngin_session.commit()
    await xngin_session.refresh(experiment, ["arm_assignments"])
    return experiment


async def test_user_from_token_when_users_exist(xngin_session: AsyncSession):
    unpriv = await user_from_token(
        xngin_session, TESTING_TOKENS[UNPRIVILEGED_TOKEN_FOR_TESTING]
    )
    assert not unpriv.is_privileged
    priv = await user_from_token(
        xngin_session, TESTING_TOKENS[PRIVILEGED_TOKEN_FOR_TESTING]
    )
    assert priv.is_privileged

    with pytest.raises(HTTPException, match="No user found with email") as e:
        await user_from_token(
            xngin_session,
            Principal(email="usernotfound@example.com", iss="", sub="", hd=""),
        )
    assert e.value.status_code == 403


async def test_user_from_token_initial_setup(xngin_session: AsyncSession):
    # emulate first time developer experience by deleting the seeded users
    await delete_seeded_users(xngin_session)

    first_user = await user_from_token(
        xngin_session, Principal(email="firstuser@example.com", iss="", sub="", hd="")
    )
    assert first_user.is_privileged
    await xngin_session.refresh(first_user, ["organizations"])
    assert len(first_user.organizations) == 1

    with pytest.raises(HTTPException, match="No user found with email") as e:
        await user_from_token(
            xngin_session,
            Principal(email="seconduser@example.com", iss="", sub="", hd=""),
        )
    assert e.value.status_code == 403


async def test_initial_user_setup_matches_testing_dwh(xngin_session):
    await delete_seeded_users(xngin_session)

    first_user = await user_from_token(
        xngin_session, Principal(email="firstuser@example.com", iss="", sub="", hd="")
    )

    # Validate directly from the db that our default org was created with datasources.
    await xngin_session.refresh(first_user, ["organizations"])
    organization = first_user.organizations[0]
    assert organization.name == DEFAULT_ORGANIZATION_NAME
    datasources = await organization.awaitable_attrs.datasources
    assert len(datasources) == 2

    # Validate that we added the testing dwh datasource.
    ds = find_ds_with_name(datasources, DEFAULT_DWH_SOURCE_NAME)
    assert isinstance(ds, tables.Datasource)
    ds_config = ds.get_config()
    pt_def = ds_config.participants[0]
    # Assert it's a "schema" type, not the old "sheets" type.
    assert isinstance(pt_def, ParticipantsDef)
    # Check auto-generated ParticipantsDef is aligned with the test dwh.
    async with DwhSession(ds_config.dwh) as dwh:
        sa_table = await dwh.inspect_table(pt_def.table_name)
    col_names = {c.name for c in sa_table.columns}
    field_names = {f.field_name for f in pt_def.fields}
    assert col_names == field_names
    for field in pt_def.fields:
        col = sa_table.columns[field.field_name]
        assert DataType.match(col.type) == field.data_type

    # Autogenerated NoDwh source should also exist.
    ds = find_ds_with_name(datasources, DEFAULT_NO_DWH_SOURCE_NAME)
    assert isinstance(ds, tables.Datasource)
    assert isinstance(ds.get_config().dwh, NoDwh)


@pytest.mark.skipif(
    flags.AIRPLANE_MODE,
    reason="This test will fail in airplane mode because airplane mode treats all Admin API calls as authenticated.",
)
def test_list_orgs_unauthenticated(client):
    response = client.get("/v1/m/organizations")
    assert response.status_code == 403, response.content


def test_list_orgs_privileged(pget):
    response = pget("/v1/m/organizations")
    assert response.status_code == 200, response.content
    assert ListOrganizationsResponse.model_validate(response.json()).items == []


def test_create_and_get_organization(ppost, pget):
    """Test basic organization creation."""
    # Create an organization
    org_name = "New Organization"
    response = ppost(
        "/v1/m/organizations",
        json=CreateOrganizationRequest(name=org_name).model_dump(),
    )
    assert response.status_code == 200, response.content
    create_response = CreateOrganizationResponse.model_validate(response.json())

    # Fetch the organization
    response = pget(f"/v1/m/organizations/{create_response.id}")
    assert response.status_code == 200, response.content
    # Verify its contents
    org_response = GetOrganizationResponse.model_validate(response.json())
    assert org_response.id == create_response.id
    assert org_response.name == org_name
    assert org_response.users[0].email == PRIVILEGED_EMAIL

    # Verify that the NoDwh datasource is present.
    assert len(org_response.datasources) == 1
    nodwh_summary = org_response.datasources[0]
    assert nodwh_summary.type == "remote"
    assert nodwh_summary.driver == "none"
    assert nodwh_summary.name == DEFAULT_NO_DWH_SOURCE_NAME
    assert nodwh_summary.organization_id == create_response.id
    assert nodwh_summary.organization_name == org_name

    # Inspect the default NoDwh datasource; should have no tables.
    response = pget(f"/v1/m/datasources/{nodwh_summary.id}/inspect")
    assert response.status_code == 200, response.content
    assert InspectDatasourceResponse.model_validate(response.json()).tables == []


@pytest.mark.skipif(
    flags.AIRPLANE_MODE,
    reason="This test will fail in airplane mode because airplane mode treats all Admin API calls as authenticated.",
)
def test_list_orgs_unprivileged(uget):
    response = uget("/v1/m/organizations")
    assert response.status_code == 200, response.content
    assert ListOrganizationsResponse.model_validate(response.json()).items == []


def test_create_datasource_invalid_dns(testing_datasource, ppost):
    """Tests that we reject insecure hostnames with a 400."""
    response = ppost(
        f"/v1/m/organizations/{testing_datasource.org.id}/members",
        json={"email": PRIVILEGED_EMAIL},
    )
    assert response.status_code == 204, response.content

    response = ppost(
        "/v1/m/datasources",
        content=CreateDatasourceRequest(
            organization_id=testing_datasource.org.id,
            name="test remote ds",
            dwh=Dsn(
                driver="postgresql+psycopg",
                host=safe_resolve.UNSAFE_IP_FOR_TESTING,
                user="postgres",
                port=5499,
                password="postgres",
                dbname="postgres",
                sslmode="disable",
            ),
        ).model_dump_json(),
    )
    assert response.status_code == 400, response.content
    assert "DNS resolution failed" in str(response.content)


def test_add_member_to_org(testing_datasource, ppost):
    """Test adding a user to an org."""
    # Add privileged user to existing organization
    response = ppost(
        f"/v1/m/organizations/{testing_datasource.org.id}/members",
        json={"email": PRIVILEGED_EMAIL},
    )
    assert response.status_code == 204, response.content

    # Add unprivileged user to existing organization
    response = ppost(
        f"/v1/m/organizations/{testing_datasource.org.id}/members",
        json={"email": UNPRIVILEGED_EMAIL},
    )
    assert response.status_code == 204, response.content


def test_list_orgs(testing_datasource_with_user, pget):
    """Test listing the orgs the user is a member of."""
    response = pget("/v1/m/organizations")
    assert response.status_code == 200, response.content
    # User was added to the test fixture org already, so no extra org was created.
    response_json = response.json()
    assert len(response_json["items"]) == 1
    assert response_json["items"][0]["id"] == testing_datasource_with_user.org.id
    assert response_json["items"][0]["name"] == "test organization"


async def test_first_user_has_an_organization_created_at_login(xngin_session, pget):
    """Test listing the orgs by the first user of the system using pget."""
    await delete_seeded_users(xngin_session)

    response = pget("/v1/m/organizations")
    assert response.status_code == 200, response.content
    assert len(response.json()["items"]) == 1, response.json()
    assert response.json()["items"][0]["name"] == "My Organization"


async def test_first_user_has_an_organization_created_at_login_unprivileged(
    xngin_session, uget
):
    """Test listing the orgs by the first user of the system using uget."""
    await delete_seeded_users(xngin_session)

    response = uget("/v1/m/organizations")
    assert response.status_code == 200, response.content
    assert len(response.json()["items"]) == 1, response.json()
    assert response.json()["items"][0]["name"] == "My Organization"


def test_datasource_lifecycle(ppost, pget, ppatch):
    """Test creating, listing, updating a datasource."""
    # The user does not initially have any organizations.
    response = pget("/v1/m/organizations")
    assert response.status_code == 200, response.content
    assert not ListOrganizationsResponse.model_validate(response.json()).items

    # Create an organization.
    response = ppost(
        "/v1/m/organizations",
        json=CreateOrganizationRequest(name="test_datasource_lifecycle").model_dump(),
    )
    assert response.status_code == 200, response.content
    org_id = CreateOrganizationResponse.model_validate(response.json()).id

    # Create datasource
    new_ds_name = "test remote ds"
    response = ppost(
        "/v1/m/datasources",
        content=CreateDatasourceRequest(
            organization_id=org_id,
            name=new_ds_name,
            # These settings correspond to the Postgres spun up in GHA or via localpg.py.
            dwh=Dsn(
                driver="postgresql+psycopg",
                host="127.0.0.1",
                user="postgres",
                port=5499,
                password="postgres",
                dbname="postgres",
                sslmode="disable",
            ),
        ).model_dump_json(),
    )
    assert response.status_code == 200, response.content
    datasource_response = CreateDatasourceResponse.model_validate(response.json())
    datasource_id = datasource_response.id

    # List datasources
    response = pget(f"/v1/m/organizations/{org_id}/datasources")
    assert response.status_code == 200, response.content
    list_ds_response = ListDatasourcesResponse.model_validate(response.json())

    assert len(list_ds_response.items) == 2
    # Ensure we have a NoDWH source
    no_dwh = find_ds_with_name(
        list_ds_response.items,
        DEFAULT_NO_DWH_SOURCE_NAME,
    )
    assert isinstance(no_dwh, DatasourceSummary)
    assert no_dwh.driver == "none"
    # Ensure we have a test dwh source
    test_dwh = find_ds_with_name(
        list_ds_response.items,
        new_ds_name,
    )
    assert isinstance(test_dwh, DatasourceSummary)
    assert test_dwh.id == datasource_id
    assert test_dwh.organization_id == org_id
    assert test_dwh.driver == "postgresql+psycopg"

    # Update datasource name
    updated_ds_name = "updated name"
    response = ppatch(
        f"/v1/m/datasources/{datasource_id}",
        content=UpdateDatasourceRequest(
            name=updated_ds_name,
        ).model_dump_json(),
    )
    assert response.status_code == 204, response.content

    # List datasources to confirm update
    response = pget(f"/v1/m/organizations/{org_id}/datasources")
    assert response.status_code == 200, response.content
    list_ds_response = ListDatasourcesResponse.model_validate(response.json())
    test_dwh = find_ds_with_name(list_ds_response.items, updated_ds_name)
    # Ensure driver didn't change, just name
    assert isinstance(test_dwh, DatasourceSummary)
    assert test_dwh.id == datasource_id
    assert test_dwh.driver == "postgresql+psycopg"

    # Update DWH on the datasource
    response = ppatch(
        f"/v1/m/datasources/{datasource_id}",
        content=UpdateDatasourceRequest(
            dwh=BqDsn(
                driver="bigquery",
                project_id="123456",
                dataset_id="ds",
                credentials=GcpServiceAccountInfo(
                    type="serviceaccountinfo",
                    content_base64=base64.b64encode(
                        json.dumps(SAMPLE_GCLOUD_SERVICE_ACCOUNT_KEY).encode("utf-8")
                    ).decode(),
                ),
            )
        ).model_dump_json(),
    )
    assert response.status_code == 204, response.content

    # List datasources to confirm update
    response = pget(
        f"/v1/m/organizations/{org_id}/datasources",
    )
    assert response.status_code == 200, response.content
    list_ds_response = ListDatasourcesResponse.model_validate(response.json())
    test_dwh = find_ds_with_name(list_ds_response.items, updated_ds_name)
    # Ensure driver changed, name didn't
    assert isinstance(test_dwh, DatasourceSummary)
    assert test_dwh.id == datasource_id
    assert test_dwh.driver == "bigquery"


def test_delete_datasource(testing_datasource_with_user, pget, udelete, pdelete):
    """Test deleting a datasource a few different ways."""
    ds_id = testing_datasource_with_user.ds.id
    org_id = testing_datasource_with_user.org.id

    # udelete() authenticates as a user that is not in the same organization as the datasource but the delete
    # endpoint always sends a 204.
    response = udelete(f"/v1/m/datasources/{ds_id}")
    assert response.status_code == 204, response.content

    response = pget(f"/v1/m/organizations/{org_id}/datasources")
    assert response.status_code == 200, response.content
    assert ListDatasourcesResponse.model_validate(response.json()).items, (
        response.content
    )  # non-empty list

    # Delete the datasource as a privileged user.
    response = pdelete(f"/v1/m/datasources/{ds_id}")
    assert response.status_code == 204, response.content

    # Assure the datasource was deleted.
    response = pget(f"/v1/m/organizations/{org_id}/datasources")
    assert response.status_code == 200, response.content
    assert ListDatasourcesResponse.model_validate(response.json()).items == []

    # Delete the datasource a 2nd time returns same code.
    response = pdelete(f"/v1/m/datasources/{ds_id}")
    assert response.status_code == 204, response.content


async def test_webhook_lifecycle(pdelete, ppost, ppatch, pget):
    """Test creating, updating, and deleting a webhook."""
    # Create an organization.
    response = ppost(
        "/v1/m/organizations",
        json=CreateOrganizationRequest(name="test_webhook_lifecycle").model_dump(),
    )
    assert response.status_code == 200, response.content
    org_id = CreateOrganizationResponse.model_validate(response.json()).id

    # Create a webhook
    response = ppost(
        f"/v1/m/organizations/{org_id}/webhooks",
        json=AddWebhookToOrganizationRequest(
            type="experiment.created",
            url="https://example.com/webhook",
            name="test webhook",
        ).model_dump(),
    )
    assert response.status_code == 200, response.content
    webhook_data = AddWebhookToOrganizationResponse.model_validate(response.json())
    assert webhook_data.name == "test webhook"
    assert webhook_data.type == "experiment.created"
    assert webhook_data.url == "https://example.com/webhook"
    assert webhook_data.auth_token is not None
    webhook_id = webhook_data.id
    original_auth_token = webhook_data.auth_token

    # List webhooks to verify creation
    response = pget(f"/v1/m/organizations/{org_id}/webhooks")
    assert response.status_code == 200, response.content
    webhooks = ListWebhooksResponse.model_validate(response.json()).items
    assert len(webhooks) == 1
    assert webhooks[0].id == webhook_id
    assert webhooks[0].url == "https://example.com/webhook"
    assert webhooks[0].auth_token == original_auth_token

    # Regenerate the auth token
    response = ppost(f"/v1/m/organizations/{org_id}/webhooks/{webhook_id}/authtoken")
    assert response.status_code == 204, response.content

    # List webhooks to verify auth token was changed
    response = pget(f"/v1/m/organizations/{org_id}/webhooks")
    assert response.status_code == 200, response.content
    webhooks = ListWebhooksResponse.model_validate(response.json()).items
    assert len(webhooks) == 1
    assert webhooks[0].auth_token != original_auth_token
    assert webhooks[0].auth_token is not None

    # Update the webhook URL
    new_url = "https://updated-example.com/webhook"
    new_name = "new name"
    response = ppatch(
        f"/v1/m/organizations/{org_id}/webhooks/{webhook_id}",
        json=UpdateOrganizationWebhookRequest(url=new_url, name=new_name).model_dump(),
    )
    assert response.status_code == 204, response.content

    # List webhooks to verify update
    response = pget(f"/v1/m/organizations/{org_id}/webhooks")
    assert response.status_code == 200, response.content
    webhooks = ListWebhooksResponse.model_validate(response.json()).items
    assert len(webhooks) == 1
    assert webhooks[0].url == new_url
    assert webhooks[0].name == new_name

    # Delete the webhook
    response = pdelete(f"/v1/m/organizations/{org_id}/webhooks/{webhook_id}")
    assert response.status_code == 204, response.content

    # List webhooks to verify deletion
    response = pget(f"/v1/m/organizations/{org_id}/webhooks")
    assert response.status_code == 200, response.content
    webhooks = ListWebhooksResponse.model_validate(response.json()).items
    assert len(webhooks) == 0

    # Try to regenerate auth token for a non-existent webhook
    response = ppost(f"/v1/m/organizations/{org_id}/webhooks/{webhook_id}/authtoken")
    assert response.status_code == 404, response.content

    # Try to update a non-existent webhook
    response = ppatch(
        f"/v1/m/organizations/{org_id}/webhooks/{webhook_id}",
        json=UpdateOrganizationWebhookRequest(
            url="https://should-fail.com/webhook", name="fail"
        ).model_dump(),
    )
    assert response.status_code == 404, response.content

    # Try to delete a non-existent webhook
    response = pdelete(f"/v1/m/organizations/{org_id}/webhooks/{webhook_id}")
    assert response.status_code == 404, response.content


def test_participants_lifecycle(
    testing_datasource_with_user, pget, ppost, ppatch, pdelete
):
    """Test getting, creating, listing, updating, and deleting a participant type."""
    ds_id = testing_datasource_with_user.ds.id

    # Get participants
    response = pget(
        f"/v1/m/datasources/{ds_id}/participants/test_participant_type",
    )
    assert response.status_code == 200, response.content
    parsed = ParticipantsDef.model_validate(response.json())
    assert parsed.type == "schema"
    assert parsed.participant_type == "test_participant_type"
    assert parsed.table_name == "dwh"

    # Create participant
    response = ppost(
        f"/v1/m/datasources/{ds_id}/participants",
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
    create_pt_response = CreateParticipantsTypeResponse.model_validate(response.json())
    assert create_pt_response.participant_type == "newpt"

    # List participants
    response = pget(
        f"/v1/m/datasources/{ds_id}/participants",
    )
    assert response.status_code == 200, response.content
    list_pt_response = ListParticipantsTypeResponse.model_validate(response.json())
    assert len(list_pt_response.items) == 2, list_pt_response

    # Update participant
    response = ppatch(
        f"/v1/m/datasources/{ds_id}/participants/newpt",
        content=UpdateParticipantsTypeRequest(
            participant_type="renamedpt"
        ).model_dump_json(),
    )
    assert response.status_code == 200, response.content
    update_pt_response = UpdateParticipantsTypeResponse.model_validate(response.json())
    assert update_pt_response.participant_type == "renamedpt"

    # List participants (again)
    response = pget(f"/v1/m/datasources/{ds_id}/participants")
    assert response.status_code == 200, response.content
    list_pt_response = ListParticipantsTypeResponse.model_validate(response.json())
    assert len(list_pt_response.items) == 2, list_pt_response

    # Get the named participant type
    response = pget(
        f"/v1/m/datasources/{ds_id}/participants/renamedpt",
    )
    assert response.status_code == 200, response.content
    participants_def = ParticipantsDef.model_validate(response.json())
    assert participants_def.participant_type == "renamedpt"

    # Delete the renamed participant type.
    response = pdelete(f"/v1/m/datasources/{ds_id}/participants/renamedpt")
    assert response.status_code == 204, response.content

    # Get the named participant type after it has been deleted
    response = pget(f"/v1/m/datasources/{ds_id}/participants/renamedpt")
    assert response.status_code == 404, response.content

    # Delete the testing participant type.
    response = pdelete(
        f"/v1/m/datasources/{ds_id}/participants/test_participant_type",
    )
    assert response.status_code == 204, response.content

    # Delete the testing participant type a 2nd time.
    response = pdelete(
        f"/v1/m/datasources/{ds_id}/participants/test_participant_type",
    )
    assert response.status_code == 404, response.content


def test_create_participants_type_invalid(testing_datasource, ppost):
    response = ppost(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants",
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


async def test_lifecycle_with_db(testing_datasource, ppost, pget, pdelete):
    """Exercises the admin API methods that require an external database."""
    # Add the privileged user to the organization.
    response = ppost(
        f"/v1/m/organizations/{testing_datasource.org.id}/members",
        json={"email": PRIVILEGED_EMAIL},
    )
    assert response.status_code == 204, response.content

    # Populate the testing data warehouse. NOTE: This will drop and recreate the database!
    create_testing_dwh(dsn=testing_datasource.dsn, nrows=100)

    # Inspect the datasource.
    response = pget(f"/v1/m/datasources/{testing_datasource.ds.id}/inspect")
    assert response.status_code == 200, response.content
    datasource_inspection = InspectDatasourceResponse.model_validate(response.json())
    assert datasource_inspection.tables == ["dwh"], response.json()

    # Inspect one table in the datasource.
    response = pget(f"/v1/m/datasources/{testing_datasource.ds.id}/inspect/dwh")
    assert response.status_code == 200, response.content
    table_inspection = InspectDatasourceTableResponse.model_validate(response.json())
    assert table_inspection == InspectDatasourceTableResponse(
        # Note: create_inspect_table_response_from_table() doesn't explicitly check for uniqueness.
        detected_unique_id_fields=["id", "uuid_filter"],
        fields=[
            FieldMetadata(
                field_name="baseline_income",
                data_type=DataType.NUMERIC,
                description="",
            ),
            FieldMetadata(
                field_name="current_income",
                data_type=DataType.NUMERIC,
                description="",
            ),
            FieldMetadata(
                field_name="ethnicity",
                data_type=DataType.CHARACTER_VARYING,
                description="",
            ),
            FieldMetadata(
                field_name="first_name",
                data_type=DataType.CHARACTER_VARYING,
                description="",
            ),
            FieldMetadata(
                field_name="gender",
                data_type=DataType.CHARACTER_VARYING,
                description="",
            ),
            FieldMetadata(field_name="id", data_type=DataType.BIGINT, description=""),
            FieldMetadata(
                field_name="income", data_type=DataType.NUMERIC, description=""
            ),
            FieldMetadata(
                field_name="is_engaged", data_type=DataType.BOOLEAN, description=""
            ),
            FieldMetadata(
                field_name="is_onboarded", data_type=DataType.BOOLEAN, description=""
            ),
            FieldMetadata(
                field_name="is_recruited", data_type=DataType.BOOLEAN, description=""
            ),
            FieldMetadata(
                field_name="is_registered",
                data_type=DataType.BOOLEAN,
                description="",
            ),
            FieldMetadata(
                field_name="is_retained", data_type=DataType.BOOLEAN, description=""
            ),
            FieldMetadata(
                field_name="last_name",
                data_type=DataType.CHARACTER_VARYING,
                description="",
            ),
            FieldMetadata(
                field_name="potential_0", data_type=DataType.NUMERIC, description=""
            ),
            FieldMetadata(
                field_name="potential_1", data_type=DataType.BIGINT, description=""
            ),
            FieldMetadata(
                field_name="sample_date", data_type=DataType.DATE, description=""
            ),
            FieldMetadata(
                field_name="sample_timestamp",
                data_type=DataType.TIMESTAMP_WITHOUT_TIMEZONE,
                description="",
            ),
            FieldMetadata(
                field_name="timestamp_with_tz",
                data_type=DataType.TIMESTAMP_WITH_TIMEZONE,
                description="",
            ),
            FieldMetadata(
                field_name="uuid_filter", data_type=DataType.UUID, description=""
            ),
        ],
    )

    # Create participant
    participant_type = "participant_type_dwh"
    response = ppost(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants",
        content=CreateParticipantsTypeRequest(
            participant_type=participant_type,
            schema_def=ParticipantsSchema(
                table_name="dwh",
                fields=[
                    FieldDescriptor(
                        field_name="id",
                        data_type=DataType.INTEGER,
                        description="test",
                        is_unique_id=True,
                        is_strata=False,
                        is_filter=False,
                        is_metric=False,
                    ),
                    FieldDescriptor(
                        field_name="current_income",
                        data_type=DataType.NUMERIC,
                        description="test",
                        is_unique_id=False,
                        is_strata=False,
                        is_filter=False,
                        is_metric=True,
                    ),
                ],
            ),
        ).model_dump_json(),
    )
    assert response.status_code == 200, response.content
    created_participant_type = CreateParticipantsTypeResponse.model_validate(
        response.json()
    )
    assert created_participant_type.participant_type == participant_type

    # Create experiment using that participant type.
    response = ppost(
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments",
        params={"chosen_n": 100},
        json=make_createexperimentrequest_json(participant_type),
    )
    assert response.status_code == 200, response.content
    created_experiment = CreateExperimentResponse.model_validate(response.json())
    parsed_experiment_id = created_experiment.design_spec.experiment_id
    assert parsed_experiment_id is not None
    assert created_experiment.stopped_assignments_at is not None
    assert (
        created_experiment.stopped_assignments_reason
        == StopAssignmentReason.PREASSIGNED
    )
    parsed_arm_ids = {arm.arm_id for arm in created_experiment.design_spec.arms}
    assert len(parsed_arm_ids) == 2

    # Get that experiment.
    response = pget(
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}"
    )

    assert response.status_code == 200, response.content
    experiment_config = CreateExperimentResponse.model_validate(response.json())
    assert experiment_config.design_spec.experiment_id == parsed_experiment_id

    # List org experiments.
    response = pget(f"/v1/m/organizations/{testing_datasource.org.id}/experiments")
    assert response.status_code == 200, response.content
    experiment_list = ListExperimentsResponse.model_validate(response.json())
    assert len(experiment_list.items) == 1, experiment_list
    experiment_config = CreateExperimentResponse.model_validate(
        experiment_list.items[0].model_dump()
    )
    assert experiment_config.design_spec.experiment_id == parsed_experiment_id

    # Analyze experiment
    response = pget(
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}/analyze"
    )
    assert response.status_code == 200, response.content
    experiment_analysis = FreqExperimentAnalysis.model_validate(response.json())
    assert experiment_analysis.experiment_id == parsed_experiment_id

    # Get assignments for the experiment.
    response = pget(
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}/assignments"
    )
    assert response.status_code == 200, response.content
    assignments = GetExperimentAssignmentsResponse.model_validate(response.json())
    assert assignments.experiment_id == parsed_experiment_id
    assert assignments.sample_size == 100
    assert assignments.balance_check is not None
    assert len(assignments.assignments) == 100
    assert {arm.arm_name for arm in assignments.assignments} == {"control", "treatment"}
    assert {arm.arm_id for arm in assignments.assignments} == parsed_arm_ids

    # Delete the experiment.
    response = pdelete(
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}"
    )
    assert response.status_code == 204, response.content


def test_create_experiment_with_assignment_validation_errors(
    testing_datasource_with_user, testing_sheet_datasource_with_user, ppost
):
    """Test LateValidationError cases in create_experiment_with_assignment."""
    # Create a basic experiment request
    # Test 1: IDs present in design spec trigger LateValidationError
    base_request = make_create_preassigned_experiment_request(with_ids=True)
    base_request.design_spec.experiment_id = "123e4567-e89b-12d3-a456-426614174000"
    testing_datasource = testing_datasource_with_user
    response = ppost(
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments",
        params={"chosen_n": 100},
        content=base_request.model_dump_json(),
    )
    assert response.status_code == 422, response.content
    assert "UUIDs must not be set" in response.json()["message"]

    # Test 2: Invalid participants config (sheet instead of schema)
    # This datasource is a "remote" config, but the participants is of type "sheet".
    testing_datasource = testing_sheet_datasource_with_user
    response = ppost(
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments",
        params={"chosen_n": 100},
        content=make_create_preassigned_experiment_request().model_dump_json(),
    )
    assert response.status_code == 422, response.content
    assert "Participants must be of type schema" in response.json()["message"]


async def test_create_preassigned_experiment_using_inline_schema_ds(
    xngin_session: AsyncSession,
    testing_datasource_with_user,
    use_deterministic_random,
    ppost,
):
    datasource_id = testing_datasource_with_user.ds.id
    request_obj = make_create_preassigned_experiment_request()

    response = ppost(
        f"/v1/m/datasources/{datasource_id}/experiments",
        params={"chosen_n": 100, "random_state": 42},
        content=request_obj.model_dump_json(),
    )
    assert response.status_code == 200, response.content
    created_experiment = CreateExperimentResponse.model_validate(response.json())
    parsed_experiment_id = created_experiment.design_spec.experiment_id
    assert parsed_experiment_id is not None
    parsed_arm_ids = {arm.arm_id for arm in created_experiment.design_spec.arms}
    assert len(parsed_arm_ids) == 2

    # Verify basic response
    assert created_experiment.stopped_assignments_at is not None
    assert (
        created_experiment.stopped_assignments_reason
        == StopAssignmentReason.PREASSIGNED
    )
    assert created_experiment.design_spec.experiment_id is not None
    assert created_experiment.design_spec.arms[0].arm_id is not None
    assert created_experiment.design_spec.arms[1].arm_id is not None
    assert created_experiment.state == ExperimentState.ASSIGNED
    assign_summary = created_experiment.assign_summary
    assert assign_summary is not None
    assert assign_summary.sample_size == 100
    assert assign_summary.balance_check is not None
    assert assign_summary.balance_check.balance_ok is True

    # No power check was added in our request_obj.
    assert created_experiment.power_analyses is None
    # Check if the representations are equivalent; scrub ids from the config for comparison
    actual_design_spec = created_experiment.design_spec.model_copy(deep=True)
    actual_design_spec.experiment_id = None
    actual_design_spec.arms[0].arm_id = None
    actual_design_spec.arms[1].arm_id = None
    assert actual_design_spec == request_obj.design_spec

    experiment_id = created_experiment.design_spec.experiment_id
    (arm1_id, arm2_id) = [arm.arm_id for arm in created_experiment.design_spec.arms]

    # Verify database state using the ids in the returned DesignSpec.
    experiment = (
        await xngin_session.scalars(
            select(tables.Experiment).where(tables.Experiment.id == experiment_id)
        )
    ).one()
    assert experiment.state == ExperimentState.ASSIGNED
    assert experiment.datasource_id == datasource_id
    assert experiment.experiment_type == "freq_preassigned"
    assert experiment.participant_type == "test_participant_type"
    assert experiment.name == request_obj.design_spec.experiment_name
    assert experiment.description == request_obj.design_spec.description
    assert_dates_equal(experiment.start_date, request_obj.design_spec.start_date)
    assert_dates_equal(experiment.end_date, request_obj.design_spec.end_date)
    # Verify assignments were created
    assignments = (
        await xngin_session.scalars(
            select(tables.ArmAssignment).where(
                tables.ArmAssignment.experiment_id == experiment_id
            )
        )
    ).all()
    assert len(assignments) == 100, {
        e.name: getattr(experiment, e.name) for e in tables.Experiment.__table__.columns
    }

    # Check one assignment to see if it looks roughly right
    sample_assignment: tables.ArmAssignment = assignments[0]
    assert sample_assignment.participant_type == "test_participant_type"
    assert sample_assignment.experiment_id == experiment_id
    assert sample_assignment.arm_id in {arm1_id, arm2_id}
    for stratum in sample_assignment.strata:
        assert stratum["field_name"] in {"is_onboarded", "gender"}

    # Check for approximate balance in arm assignment
    num_control = sum(1 for a in assignments if a.arm_id == arm1_id)
    num_treat = sum(1 for a in assignments if a.arm_id == arm2_id)
    assert abs(num_control - num_treat) <= 5  # Allow some wiggle room


def test_create_online_experiment_using_inline_schema_ds(
    testing_datasource_with_user, use_deterministic_random, ppost
):
    datasource_id = testing_datasource_with_user.ds.id
    request_obj = make_create_online_experiment_request()

    response = ppost(
        f"/v1/m/datasources/{datasource_id}/experiments",
        params={"random_state": 42},
        content=request_obj.model_dump_json(),
    )
    assert response.status_code == 200, response.content
    created_experiment = CreateExperimentResponse.model_validate(response.json())
    parsed_experiment_id = created_experiment.design_spec.experiment_id
    assert parsed_experiment_id is not None
    parsed_arm_ids = {arm.arm_id for arm in created_experiment.design_spec.arms}
    assert len(parsed_arm_ids) == 2

    # Verify basic response
    assert created_experiment.stopped_assignments_at is None
    assert created_experiment.stopped_assignments_reason is None
    assert created_experiment.design_spec.experiment_id is not None
    assert created_experiment.design_spec.arms[0].arm_id is not None
    assert created_experiment.design_spec.arms[1].arm_id is not None
    assert created_experiment.state == ExperimentState.ASSIGNED
    assign_summary = created_experiment.assign_summary
    assert assign_summary is not None
    assert assign_summary.balance_check is None
    assert assign_summary.sample_size == 0
    assert assign_summary.arm_sizes is not None
    assert all(a.size == 0 for a in assign_summary.arm_sizes)
    assert created_experiment.power_analyses is None
    # Check if the representations are equivalent
    # scrub the ids from the config for comparison
    actual_design_spec = created_experiment.design_spec.model_copy(deep=True)
    actual_design_spec.experiment_id = None
    actual_design_spec.arms[0].arm_id = None
    actual_design_spec.arms[1].arm_id = None
    assert actual_design_spec == request_obj.design_spec


def test_get_experiment_assignment_for_preassigned_participant(
    testing_experiment, pget
):
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id
    assignments = testing_experiment.arm_assignments

    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/unassigned_id"
    )
    assert response.status_code == 200, response.content
    assignment_response = GetParticipantAssignmentResponse.model_validate(
        response.json()
    )
    assert assignment_response.experiment_id == experiment_id
    assert assignment_response.participant_id == "unassigned_id"
    assert assignment_response.assignment is None

    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/{assignments[0].participant_id}"
    )
    assert response.status_code == 200, response.content
    assignment_response = GetParticipantAssignmentResponse.model_validate(
        response.json()
    )
    assert assignment_response.experiment_id == experiment_id
    assert assignment_response.participant_id == assignments[0].participant_id
    assert assignment_response.assignment is not None


async def test_get_experiment_assignment_for_online_participant(
    xngin_session: AsyncSession, testing_datasource_with_user, pget
):
    test_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource_with_user.ds, ExperimentsType.FREQ_ONLINE
    )
    datasource_id = test_experiment.datasource_id
    experiment_id = test_experiment.id

    # Check for an assignment that doesn't exist, but don't create it.
    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/new_id?create_if_none=false"
    )
    assert response.status_code == 200, response.content
    assignment_response = GetParticipantAssignmentResponse.model_validate(
        response.json()
    )
    assert assignment_response.experiment_id == experiment_id
    assert assignment_response.participant_id == "new_id"
    assert assignment_response.assignment is None

    # Create a new participant assignment.
    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/new_id"
    )
    assert response.status_code == 200, response.content
    assignment_response = GetParticipantAssignmentResponse.model_validate(
        response.json()
    )
    assert assignment_response.experiment_id == experiment_id
    assert assignment_response.participant_id == "new_id"
    assert assignment_response.assignment is not None
    assert str(assignment_response.assignment.arm_id) in {
        arm.id for arm in test_experiment.arms
    }

    # Get back the same assignment.
    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/new_id"
    )
    assert response.status_code == 200, response.content
    assignment_response2 = GetParticipantAssignmentResponse.model_validate(
        response.json()
    )
    assert assignment_response2 == assignment_response

    # Make sure there's only one db entry.
    scalars = await xngin_session.scalars(
        select(tables.ArmAssignment).where(
            tables.ArmAssignment.experiment_id == experiment_id
        )
    )
    assignment = scalars.one()
    assert assignment.participant_id == "new_id"
    assert assignment.arm_id == str(assignment_response.assignment.arm_id)


async def test_get_experiment_assignment_for_online_participant_past_end_date(
    xngin_session: AsyncSession, testing_datasource_with_user, pget
):
    new_exp = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource_with_user.ds,
        ExperimentsType.FREQ_ONLINE,
        end_date=datetime.now(UTC) - timedelta(days=1),
    )
    datasource_id = new_exp.datasource_id
    experiment_id = new_exp.id

    # Verify no new assignment is created for the ended experiment.
    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/new_id"
    )
    assert response.status_code == 200, response.content
    assignment_response = GetParticipantAssignmentResponse.model_validate(
        response.json()
    )
    assert assignment_response.experiment_id == experiment_id
    assert assignment_response.participant_id == "new_id"
    assert assignment_response.assignment is None, assignment_response.model_dump_json()
    # Verify that the experiment state was updated.
    await xngin_session.refresh(new_exp)
    assert new_exp.stopped_assignments_at is not None
    assert new_exp.stopped_assignments_reason == StopAssignmentReason.END_DATE


def test_experiments_analyze(testing_experiment, pget):
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id

    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/analyze"
    )

    assert response.status_code == 200, response.content
    experiment_analysis = FreqExperimentAnalysis.model_validate(response.json())
    assert experiment_analysis.experiment_id == experiment_id
    assert len(experiment_analysis.metric_analyses) == 1
    assert experiment_analysis.num_participants == 10
    # testing_experiment assignment ids start from 0, but id=0 doesn't exist in our test data.
    assert experiment_analysis.num_missing_participants == 1
    assert datetime.now(UTC) - experiment_analysis.created_at < timedelta(seconds=5)
    # Verify that only the first arm is marked as baseline by default
    metric_analysis = experiment_analysis.metric_analyses[0]
    baseline_arms = [arm for arm in metric_analysis.arm_analyses if arm.is_baseline]
    assert len(baseline_arms) == 1
    assert baseline_arms[0].is_baseline
    for analysis in experiment_analysis.metric_analyses:
        # Verify arm_ids match the database model.
        assert {arm.arm_id for arm in analysis.arm_analyses} == {
            arm.id for arm in testing_experiment.arms
        }
        # id=0 doesn't exist in our test data, so we'll have 1 missing value across all arms.
        assert sum([arm.num_missing_values for arm in analysis.arm_analyses]) == 1


async def test_experiments_analyze_for_experiment_with_no_participants(
    xngin_session: AsyncSession, testing_datasource_with_user, pget
):
    test_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource_with_user.ds, ExperimentsType.FREQ_ONLINE
    )
    datasource_id = test_experiment.datasource_id
    experiment_id = test_experiment.id

    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/analyze"
    )
    assert response.status_code == 422, response.content
    assert response.json()["message"] == "No participants found for experiment."


@pytest.mark.parametrize(
    "endpoint,initial_state,expected_status,expected_detail",
    [
        ("commit", ExperimentState.ASSIGNED, 204, None),  # Success case
        ("commit", ExperimentState.COMMITTED, 304, None),  # No-op
        ("commit", ExperimentState.DESIGNING, 400, "Invalid state: designing"),
        ("commit", ExperimentState.ABORTED, 400, "Invalid state: aborted"),
        ("abandon", ExperimentState.DESIGNING, 204, None),  # Success case
        ("abandon", ExperimentState.ASSIGNED, 204, None),  # Success case
        ("abandon", ExperimentState.ABANDONED, 304, None),  # No-op
        ("abandon", ExperimentState.COMMITTED, 400, "Invalid state: committed"),
    ],
)
async def test_admin_experiment_state_setting(
    xngin_session: AsyncSession,
    testing_datasource_with_user,
    endpoint,
    initial_state,
    expected_status,
    expected_detail,
    ppost,
):
    # Initialize our state with an existing experiment who's state we want to modify.
    datasource = testing_datasource_with_user.ds
    experiment, _ = make_insertable_experiment(datasource, initial_state)
    xngin_session.add(experiment)
    await xngin_session.commit()

    response = ppost(
        f"/v1/m/datasources/{datasource.id}/experiments/{experiment.id!s}/{endpoint}"
    )

    # Verify
    assert response.status_code == expected_status
    # If success case, verify state was updated
    if expected_status == 204:
        expected_state = (
            ExperimentState.ABANDONED
            if endpoint == "abandon"
            else ExperimentState.COMMITTED
        )
        await xngin_session.refresh(experiment)
        assert experiment.state == expected_state
    # If failure case, verify the error message
    if expected_detail:
        assert response.json()["detail"] == expected_detail


async def test_manage_apikeys(testing_datasource_with_user, ppost, pget, pdelete):
    ds = testing_datasource_with_user.ds

    response = pget(f"/v1/m/datasources/{ds.id}/apikeys")
    assert response.status_code == 200
    list_api_keys_response = ListApiKeysResponse.model_validate(response.json())
    assert len(list_api_keys_response.items) == 1

    response = ppost(f"/v1/m/datasources/{ds.id}/apikeys/")
    assert response.status_code == 200
    create_api_key_response = CreateApiKeyResponse.model_validate(response.json())
    assert create_api_key_response.datasource_id == ds.id
    created_api_key_id = create_api_key_response.id

    response = pget(f"/v1/m/datasources/{ds.id}/apikeys")
    assert response.status_code == 200
    list_api_keys_response = ListApiKeysResponse.model_validate(response.json())
    assert len(list_api_keys_response.items) == 2

    response = pdelete(f"/v1/m/datasources/{ds.id}/apikeys/{created_api_key_id}")
    assert response.status_code == 204

    response = pget(f"/v1/m/datasources/{ds.id}/apikeys")
    assert response.status_code == 200
    list_api_keys_response = ListApiKeysResponse.model_validate(response.json())
    assert len(list_api_keys_response.items) == 1


async def test_experiment_webhook_integration(
    testing_datasource_with_user, ppost, pget
):
    """Test creating an experiment with webhook associations and verifying webhook IDs in response."""
    org_id = testing_datasource_with_user.org.id
    datasource_id = testing_datasource_with_user.ds.id

    # Create two webhooks in the organization
    webhook1_response = ppost(
        f"/v1/m/organizations/{org_id}/webhooks",
        json=AddWebhookToOrganizationRequest(
            type="experiment.created",
            name="Test Webhook 1",
            url="https://example.com/webhook1",
        ).model_dump(),
    )
    assert webhook1_response.status_code == 200, webhook1_response.content
    webhook1_id = webhook1_response.json()["id"]

    webhook2_response = ppost(
        f"/v1/m/organizations/{org_id}/webhooks",
        json=AddWebhookToOrganizationRequest(
            type="experiment.created",
            name="Test Webhook 2",
            url="https://example.com/webhook2",
        ).model_dump(),
    )
    assert webhook2_response.status_code == 200, webhook2_response.content
    webhook2_id = webhook2_response.json()["id"]

    # Create an experiment with only the first webhook using proper Pydantic models
    experiment_request = CreateExperimentRequest(
        design_spec=PreassignedFrequentistExperimentSpec(
            participant_type="test_participant_type",
            experiment_type="freq_preassigned",
            experiment_name="Test Experiment with Webhook",
            description="Testing webhook integration",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, 23, 59, 59, tzinfo=UTC),
            arms=[
                Arm(arm_name="control", arm_description="Control group"),
                Arm(arm_name="treatment", arm_description="Treatment group"),
            ],
            metrics=[DesignSpecMetricRequest(field_name="income", metric_pct_change=5)],
            strata=[],
            filters=[],
        ),
        webhooks=[webhook1_id],  # Only include the first webhook
    )

    create_response = ppost(
        f"/v1/m/datasources/{datasource_id}/experiments?chosen_n=100",
        json=experiment_request.model_dump(mode="json"),
    )
    assert create_response.status_code == 200, create_response.content

    # Verify the create response includes the webhook
    created_experiment = create_response.json()
    assert "webhooks" in created_experiment
    assert len(created_experiment["webhooks"]) == 1
    assert created_experiment["webhooks"][0] == webhook1_id

    # Get the experiment ID for further testing
    experiment_id = created_experiment["design_spec"]["experiment_id"]

    # Get the experiment and verify webhook is included
    get_response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}"
    )
    assert get_response.status_code == 200, get_response.content

    retrieved_experiment = get_response.json()
    assert "webhooks" in retrieved_experiment
    assert len(retrieved_experiment["webhooks"]) == 1
    assert retrieved_experiment["webhooks"][0] == webhook1_id

    # Verify the second webhook is not included
    assert webhook2_id not in retrieved_experiment["webhooks"]

    # Test creating an experiment with no webhooks using proper Pydantic models
    experiment_request_no_webhooks = CreateExperimentRequest(
        design_spec=PreassignedFrequentistExperimentSpec(
            experiment_type="freq_preassigned",
            participant_type="test_participant_type",
            experiment_name="Test Experiment without Webhooks",
            description="Testing no webhook integration",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, 23, 59, 59, tzinfo=UTC),
            arms=[
                Arm(arm_name="control", arm_description="Control group"),
                Arm(arm_name="treatment", arm_description="Treatment group"),
            ],
            metrics=[DesignSpecMetricRequest(field_name="income", metric_pct_change=5)],
            strata=[],
            filters=[],
        )
        # No webhooks field - should default to empty list
    )

    create_response_no_webhooks = ppost(
        f"/v1/m/datasources/{datasource_id}/experiments?chosen_n=100",
        json=experiment_request_no_webhooks.model_dump(mode="json"),
    )
    assert create_response_no_webhooks.status_code == 200, (
        create_response_no_webhooks.content
    )

    # Verify no webhooks are associated
    created_experiment_no_webhooks = create_response_no_webhooks.json()
    assert "webhooks" in created_experiment_no_webhooks
    assert len(created_experiment_no_webhooks["webhooks"]) == 0
