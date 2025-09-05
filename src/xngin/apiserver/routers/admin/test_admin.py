import base64
import copy
import json
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import flags
from xngin.apiserver.conftest import delete_seeded_users
from xngin.apiserver.dns import safe_resolve
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.dwh.inspection_types import FieldDescriptor, ParticipantsSchema
from xngin.apiserver.routers.admin.admin_api import user_from_token
from xngin.apiserver.routers.admin.admin_api_converters import (
    CREDENTIALS_UNAVAILABLE_MESSAGE,
)
from xngin.apiserver.routers.admin.admin_api_types import (
    AddWebhookToOrganizationRequest,
    AddWebhookToOrganizationResponse,
    ApiOnlyDsn,
    BqDsn,
    CreateApiKeyResponse,
    CreateDatasourceRequest,
    CreateDatasourceResponse,
    CreateOrganizationRequest,
    CreateOrganizationResponse,
    CreateParticipantsTypeRequest,
    CreateParticipantsTypeResponse,
    CreateSnapshotResponse,
    DatasourceSummary,
    FieldMetadata,
    GcpServiceAccount,
    GetDatasourceResponse,
    GetOrganizationResponse,
    GetSnapshotResponse,
    Hidden,
    InspectDatasourceResponse,
    InspectDatasourceTableResponse,
    ListApiKeysResponse,
    ListDatasourcesResponse,
    ListOrganizationsResponse,
    ListParticipantsTypeResponse,
    ListSnapshotsResponse,
    ListWebhooksResponse,
    PostgresDsn,
    RedshiftDsn,
    RevealedStr,
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
    ArmBandit,
    CMABExperimentSpec,
    CreateExperimentRequest,
    CreateExperimentResponse,
    DataType,
    DesignSpecMetricRequest,
    ExperimentsType,
    Filter,
    FreqExperimentAnalysisResponse,
    GetExperimentAssignmentsResponse,
    GetParticipantAssignmentResponse,
    LikelihoodTypes,
    ListExperimentsResponse,
    MABExperimentSpec,
    PreassignedFrequentistExperimentSpec,
    PriorTypes,
)
from xngin.apiserver.routers.common_enums import ExperimentState, Relation, StopAssignmentReason
from xngin.apiserver.routers.experiments.test_experiments_common import (
    insert_experiment_and_arms,
    make_create_online_bandit_experiment_request,
    make_create_online_experiment_request,
    make_create_preassigned_experiment_request,
    make_createexperimentrequest_json,
    make_insertable_experiment,
)
from xngin.apiserver.settings import NoDwh, ParticipantsDef
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.bootstrap import (
    DEFAULT_DWH_SOURCE_NAME,
    DEFAULT_NO_DWH_SOURCE_NAME,
    DEFAULT_ORGANIZATION_NAME,
)
from xngin.apiserver.testing.assertions import assert_dates_equal
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_PARTICIPANT_DEF
from xngin.cli.main import create_testing_dwh

SAMPLE_GCLOUD_SERVICE_ACCOUNT = {
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
SAMPLE_GCLOUD_SERVICE_ACCOUNT_JSON = json.dumps(SAMPLE_GCLOUD_SERVICE_ACCOUNT)


def find_ds_with_name[DSType: tables.Datasource | DatasourceSummary](datasources: list[DSType], name: str) -> DSType:
    """Helper function to find a datasource with a specific name from an iterable.

    Raises StopIteration if the datasource is not found.
    """
    return next(ds for ds in datasources if ds.name == name)


@pytest.fixture(name="testing_experiment")
async def fixture_testing_experiment(xngin_session: AsyncSession, testing_datasource_with_user):
    """Create an experiment on a test inline schema datasource with proper user permissions."""
    datasource = testing_datasource_with_user.ds
    experiment = await insert_experiment_and_arms(xngin_session, datasource, ExperimentsType.FREQ_PREASSIGNED)
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
    unpriv = await user_from_token(xngin_session, TESTING_TOKENS[UNPRIVILEGED_TOKEN_FOR_TESTING])
    assert not unpriv.is_privileged
    priv = await user_from_token(xngin_session, TESTING_TOKENS[PRIVILEGED_TOKEN_FOR_TESTING])
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

    first_user = await user_from_token(xngin_session, Principal(email="firstuser@example.com", iss="", sub="", hd=""))
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

    first_user = await user_from_token(xngin_session, Principal(email="initial@example.com", iss="", sub="", hd=""))

    # Validate directly from the db that our default org was created with datasources.
    await xngin_session.refresh(first_user, ["organizations"])
    organization = first_user.organizations[0]
    assert organization.name == DEFAULT_ORGANIZATION_NAME
    datasources: list[tables.Datasource] = await organization.awaitable_attrs.datasources
    assert len(datasources) == 2

    # Validate that we added the testing dwh datasource.
    ds = find_ds_with_name(datasources, DEFAULT_DWH_SOURCE_NAME)
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


@pytest.mark.parametrize(
    "dsn",
    [
        PostgresDsn(
            host="127.0.0.1",
            user="postgres",
            port=5499,
            password=RevealedStr(value="postgres"),
            dbname="postgres",
            sslmode="disable",
            search_path=None,
        ),
        RedshiftDsn(
            host="foo.redshift.amazonaws.com",
            user="postgres",
            port=5499,
            password=RevealedStr(value="postgres"),
            dbname="postgres",
            search_path=None,
        ),
        BqDsn(
            project_id="projectid",
            dataset_id="dataset_id",
            credentials=GcpServiceAccount(content=SAMPLE_GCLOUD_SERVICE_ACCOUNT_JSON),
        ),
    ],
    ids=lambda d: type(d),
)
async def test_datasources_hide_credentials(
    disable_safe_resolve_check,
    dsn: PostgresDsn | RedshiftDsn | BqDsn,
    xngin_session,
    ppost,
    pget,
    ppatch,
):
    response = ppost(
        "/v1/m/organizations",
        json=CreateOrganizationRequest(name="test_datasources_hide_credentials").model_dump(),
    )
    assert response.status_code == 200, response.content
    org_id = CreateOrganizationResponse.model_validate(response.json()).id

    response = ppost(
        "/v1/m/datasources",
        content=CreateDatasourceRequest(
            name="test_create_datasource",
            organization_id=org_id,
            dsn=dsn,
        ).model_dump_json(),
    )
    assert response.status_code == 200, response.content
    datasource_id = CreateDatasourceResponse.model_validate(response.json()).id

    response = pget(f"/v1/m/datasources/{datasource_id}")
    assert response.status_code == 200, response.content
    datasource_response = GetDatasourceResponse.model_validate(response.json())
    match datasource_response.dsn:
        case ApiOnlyDsn():
            raise TypeError("unexpected dsn type")
        case PostgresDsn() | RedshiftDsn():
            assert isinstance(datasource_response.dsn.password, Hidden)
        case BqDsn():
            assert isinstance(datasource_response.dsn.credentials, Hidden)

    # Send an update that changes credentials.
    before_revision = (await xngin_session.get(tables.Datasource, datasource_id)).get_config()
    match dsn:
        case PostgresDsn() | RedshiftDsn():
            revised_pg: PostgresDsn | RedshiftDsn = dsn.model_copy(deep=True)
            revised_pg.password = RevealedStr(value="updated")
            update_request = UpdateDatasourceRequest(dsn=revised_pg)
        case BqDsn():
            revised_service_account = copy.deepcopy(SAMPLE_GCLOUD_SERVICE_ACCOUNT)
            revised_service_account["private_key"] = "newprivatekey"
            revised_bq = dsn.model_copy(deep=True)
            revised_bq.credentials = GcpServiceAccount(content=json.dumps(revised_service_account))
            update_request = UpdateDatasourceRequest(dsn=revised_bq)
    response = ppatch(f"/v1/m/datasources/{datasource_id}", content=update_request.model_dump_json())
    assert response.status_code == 204, response.content

    after_revision = (await xngin_session.get(tables.Datasource, datasource_id)).get_config()
    match dsn:
        case PostgresDsn() | RedshiftDsn():
            assert after_revision.dwh.host == before_revision.dwh.host
            assert after_revision.dwh.password != before_revision.dwh.password
            assert after_revision.dwh.password == "updated"
        case BqDsn():
            assert after_revision.dwh.project_id == before_revision.dwh.project_id
            assert after_revision.dwh.credentials != before_revision.dwh.credentials
            assert "newprivatekey" in base64.standard_b64decode(after_revision.dwh.credentials.content_base64).decode()

    # Send an update with hidden credentials and confirm that non-credential data remains the same.
    match dsn:
        case PostgresDsn() | RedshiftDsn():
            revised_pg = dsn.model_copy(deep=True)
            revised_pg.dbname = "newdatabase"
            revised_pg.password = Hidden()
            update_request = UpdateDatasourceRequest(dsn=revised_pg)
        case BqDsn():
            revised_bq = dsn.model_copy(deep=True)
            revised_bq.project_id = "newprojectid"
            revised_bq.credentials = Hidden()
            update_request = UpdateDatasourceRequest(dsn=revised_bq)
    response = ppatch(f"/v1/m/datasources/{datasource_id}", content=update_request.model_dump_json())
    assert response.status_code == 204, response.content

    after_second_revision = (await xngin_session.get(tables.Datasource, datasource_id)).get_config()
    match dsn:
        case PostgresDsn() | RedshiftDsn():
            assert after_second_revision.dwh.dbname == "newdatabase"
            assert after_second_revision.dwh.host == after_revision.dwh.host
            assert after_second_revision.dwh.password == after_revision.dwh.password
            assert after_second_revision.dwh.password == "updated"
        case BqDsn():
            assert after_second_revision.dwh.project_id == "newprojectid"
            assert after_second_revision.dwh.dataset_id == after_revision.dwh.dataset_id
            assert after_second_revision.dwh.credentials == after_revision.dwh.credentials
            assert (
                "newprivatekey"
                in base64.standard_b64decode(after_second_revision.dwh.credentials.content_base64).decode()
            )


@pytest.mark.parametrize(
    "dsn",
    [
        PostgresDsn(
            host="127.0.0.1",
            user="postgres",
            port=5499,
            password=Hidden(),
            dbname="postgres",
            sslmode="disable",
            search_path=None,
        ),
        RedshiftDsn(
            host="foo.redshift.amazonaws.com",
            user="postgres",
            port=5499,
            password=Hidden(),
            dbname="postgres",
            search_path=None,
        ),
        BqDsn(
            project_id="projectid",
            dataset_id="dataset_id",
            credentials=Hidden(),
        ),
    ],
    ids=lambda d: type(d),
)
def test_create_datasource_without_credentials(
    disable_safe_resolve_check, dsn: BqDsn | RedshiftDsn | PostgresDsn, ppost
):
    response = ppost(
        "/v1/m/organizations",
        json=CreateOrganizationRequest(name="test_create_datasource_without_credentials").model_dump(),
    )
    assert response.status_code == 200, response.content
    org_id = CreateOrganizationResponse.model_validate(response.json()).id

    response = ppost(
        "/v1/m/datasources",
        content=CreateDatasourceRequest(
            name="test_create_datasource",
            organization_id=org_id,
            dsn=dsn,
        ).model_dump_json(),
    )
    assert response.status_code == 422, response.content
    assert response.json() == {"message": CREDENTIALS_UNAVAILABLE_MESSAGE}


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
            dsn=PostgresDsn(
                host=safe_resolve.UNSAFE_IP_FOR_TESTING,
                user="postgres",
                port=5499,
                password=RevealedStr(value="postgres"),
                dbname="postgres",
                sslmode="disable",
                search_path=None,
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


def test_remove_member_from_org(xngin_session, pget, ppost, pdelete):
    def create_organization():
        creation_response = ppost(
            "/v1/m/organizations",
            json=CreateOrganizationRequest(name="test_remove_member_from_org").model_dump(),
        )
        assert creation_response.status_code == 200, creation_response.content
        parsed = CreateOrganizationResponse.model_validate(creation_response.json())
        return parsed.id

    org_id = create_organization()

    def list_members():
        org_response = pget(f"/v1/m/organizations/{org_id}")
        assert org_response.status_code == 200, org_response.content
        return {u.email: u.id for u in GetOrganizationResponse.model_validate(org_response.json()).users}

    def add_member(email: str):
        add_response = ppost(
            f"/v1/m/organizations/{org_id}/members",
            json={"email": email},
        )
        assert add_response.status_code == 204, add_response.content
        return add_response

    def remove_member(user_id: str, extra: str | None = None):
        return pdelete(f"/v1/m/organizations/{org_id}/members/{user_id}" + (f"?{extra}" if extra else ""))

    assert list_members().keys() == {PRIVILEGED_EMAIL}
    add_member(UNPRIVILEGED_EMAIL)
    member_list = list_members()
    assert member_list.keys() == {PRIVILEGED_EMAIL, UNPRIVILEGED_EMAIL}

    # 404 when trying to remove self from organization
    response = remove_member(member_list.get(PRIVILEGED_EMAIL))
    assert response.status_code == 404, response.content
    assert list_members().keys() == {PRIVILEGED_EMAIL, UNPRIVILEGED_EMAIL}

    # 204 when remove existing member
    response = remove_member(member_list.get(UNPRIVILEGED_EMAIL))
    assert response.status_code == 204, response.content
    assert list_members().keys() == {PRIVILEGED_EMAIL}

    # 404 when removing non-existent member
    response = remove_member(member_list.get(UNPRIVILEGED_EMAIL))
    assert response.status_code == 404, response.content
    assert list_members().keys() == {PRIVILEGED_EMAIL}

    # 204 when removing non-existent member w/allow-missing
    response = remove_member(member_list.get(UNPRIVILEGED_EMAIL), "allow_missing=true")
    assert response.status_code == 204, response.content
    assert list_members().keys() == {PRIVILEGED_EMAIL}


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


async def test_first_user_has_an_organization_created_at_login_unprivileged(xngin_session, uget):
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
            dsn=PostgresDsn(
                host="127.0.0.1",
                user="postgres",
                port=5499,
                password=RevealedStr(value="postgres"),
                dbname="postgres",
                sslmode="disable",
                search_path=None,
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
    assert no_dwh.driver == "none"
    # Ensure we have a test dwh source
    test_dwh = find_ds_with_name(
        list_ds_response.items,
        new_ds_name,
    )
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
    assert test_dwh.id == datasource_id
    assert test_dwh.driver == "postgresql+psycopg"

    # Update DWH on the datasource
    response = ppatch(
        f"/v1/m/datasources/{datasource_id}",
        content=UpdateDatasourceRequest(
            dsn=BqDsn(
                project_id="123456",
                dataset_id="ds",
                credentials=GcpServiceAccount(content=SAMPLE_GCLOUD_SERVICE_ACCOUNT_JSON),
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
    assert test_dwh.id == datasource_id
    assert test_dwh.driver == "bigquery"


def test_delete_datasource(testing_datasource_with_user, pget, udelete, pdelete):
    """Test deleting a datasource a few different ways."""
    ds_id = testing_datasource_with_user.ds.id
    org_id = testing_datasource_with_user.org.id

    # udelete() authenticates as a user that is not in the same organization as the datasource.
    response = udelete(f"/v1/m/organizations/{org_id}/datasources/{ds_id}")
    assert response.status_code == 403, response.content

    response = pget(f"/v1/m/organizations/{org_id}/datasources")
    assert response.status_code == 200, response.content
    assert ListDatasourcesResponse.model_validate(response.json()).items, response.content  # non-empty list

    # Delete the datasource as a privileged user.
    response = pdelete(f"/v1/m/organizations/{org_id}/datasources/{ds_id}")
    assert response.status_code == 204, response.content

    # Assure the datasource was deleted.
    response = pget(f"/v1/m/organizations/{org_id}/datasources")
    assert response.status_code == 200, response.content
    assert ListDatasourcesResponse.model_validate(response.json()).items == []

    # Delete the datasource a 2nd time returns 404.
    response = pdelete(f"/v1/organizations/{org_id}/m/datasources/{ds_id}")
    assert response.status_code == 404, response.content

    # Delete the datasource a 2nd time returns 204 when ?allow_missing is set.
    response = pdelete(f"/v1/m/organizations/{org_id}/datasources/{ds_id}?allow_missing=true")
    assert response.status_code == 204, response.content

    response = pget(f"/v1/m/organizations/{org_id}/datasources")
    assert response.status_code == 200, response.content
    assert not ListDatasourcesResponse.model_validate(response.json()).items, response.content  # empty list


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

    # Delete the webhook again (404)
    response = pdelete(f"/v1/m/organizations/{org_id}/webhooks/{webhook_id}")
    assert response.status_code == 404, response.content

    # Delete the webhook again (204)
    response = pdelete(f"/v1/m/organizations/{org_id}/webhooks/{webhook_id}?allow_missing=True")
    assert response.status_code == 204, response.content

    # Try to regenerate auth token for a non-existent webhook
    response = ppost(f"/v1/m/organizations/{org_id}/webhooks/{webhook_id}/authtoken")
    assert response.status_code == 404, response.content

    # Try to update a non-existent webhook
    response = ppatch(
        f"/v1/m/organizations/{org_id}/webhooks/{webhook_id}",
        json=UpdateOrganizationWebhookRequest(url="https://should-fail.com/webhook", name="fail").model_dump(),
    )
    assert response.status_code == 404, response.content

    # Try to delete a non-existent webhook
    response = pdelete(f"/v1/m/organizations/{org_id}/webhooks/{webhook_id}")
    assert response.status_code == 404, response.content


def test_participants_lifecycle(testing_datasource_with_user, pget, ppost, ppatch, pdelete):
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
        content=UpdateParticipantsTypeRequest(participant_type="renamedpt").model_dump_json(),
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

    # Delete the renamed participant type again.
    response = pdelete(f"/v1/m/datasources/{ds_id}/participants/renamedpt")
    assert response.status_code == 404, response.content

    # Delete the renamed participant type again w/allow_missing.
    response = pdelete(f"/v1/m/datasources/{ds_id}/participants/renamedpt?allow_missing=1")
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

    # Delete a participant type in a non-existent datasource.
    response = pdelete(
        "/v1/m/datasources/ds-not-exist/participants/test_participant_type",
    )
    assert response.status_code == 403, response.content


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
    assert "no columns marked as unique ID." in response.json()["detail"][0]["msg"], response.content


async def test_lifecycle_with_db(testing_datasource, ppost, pget, pdelete, udelete):
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
            FieldMetadata(field_name="income", data_type=DataType.NUMERIC, description=""),
            FieldMetadata(field_name="is_engaged", data_type=DataType.BOOLEAN, description=""),
            FieldMetadata(field_name="is_onboarded", data_type=DataType.BOOLEAN, description=""),
            FieldMetadata(field_name="is_recruited", data_type=DataType.BOOLEAN, description=""),
            FieldMetadata(
                field_name="is_registered",
                data_type=DataType.BOOLEAN,
                description="",
            ),
            FieldMetadata(field_name="is_retained", data_type=DataType.BOOLEAN, description=""),
            FieldMetadata(
                field_name="last_name",
                data_type=DataType.CHARACTER_VARYING,
                description="",
            ),
            FieldMetadata(field_name="potential_0", data_type=DataType.NUMERIC, description=""),
            FieldMetadata(field_name="potential_1", data_type=DataType.BIGINT, description=""),
            FieldMetadata(field_name="sample_date", data_type=DataType.DATE, description=""),
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
            FieldMetadata(field_name="uuid_filter", data_type=DataType.UUID, description=""),
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
    created_participant_type = CreateParticipantsTypeResponse.model_validate(response.json())
    assert created_participant_type.participant_type == participant_type

    # Create experiment using that participant type.
    response = ppost(
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments",
        params={"chosen_n": 100},
        json=make_createexperimentrequest_json(participant_type),
    )
    assert response.status_code == 200, response.content
    created_experiment = CreateExperimentResponse.model_validate(response.json())
    parsed_experiment_id = created_experiment.experiment_id
    assert parsed_experiment_id is not None
    assert created_experiment.stopped_assignments_at is not None
    assert created_experiment.stopped_assignments_reason == StopAssignmentReason.PREASSIGNED
    parsed_arm_ids = {arm.arm_id for arm in created_experiment.design_spec.arms}
    assert len(parsed_arm_ids) == 2

    # Get that experiment.
    response = pget(f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}")

    assert response.status_code == 200, response.content
    create_experiment_response = CreateExperimentResponse.model_validate(response.json())
    assert create_experiment_response.experiment_id == parsed_experiment_id

    # List org experiments.
    response = pget(f"/v1/m/organizations/{testing_datasource.org.id}/experiments")
    assert response.status_code == 200, response.content
    experiment_list = ListExperimentsResponse.model_validate(response.json())
    assert len(experiment_list.items) == 1, experiment_list
    experiment_config_0 = experiment_list.items[0]
    assert experiment_config_0.experiment_id == parsed_experiment_id

    # Analyze experiment
    response = pget(f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}/analyze")
    assert response.status_code == 200, response.content
    experiment_analysis = FreqExperimentAnalysisResponse.model_validate(response.json())
    assert experiment_analysis.experiment_id == parsed_experiment_id

    # Get assignments for the experiment.
    response = pget(f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}/assignments")
    assert response.status_code == 200, response.content
    assignments = GetExperimentAssignmentsResponse.model_validate(response.json())
    assert assignments.experiment_id == parsed_experiment_id
    assert assignments.sample_size == 100
    assert assignments.balance_check is not None
    assert len(assignments.assignments) == 100
    assert {arm.arm_name for arm in assignments.assignments} == {"control", "treatment"}
    assert {arm.arm_id for arm in assignments.assignments} == parsed_arm_ids

    # Unprivileged user attempts to delete the experiment
    response = udelete(f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}")
    assert response.status_code == 403

    # Delete the experiment.
    response = pdelete(f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}")
    assert response.status_code == 204, response.content

    # Delete the experiment again.
    response = pdelete(f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}")
    assert response.status_code == 404, response.content

    # Delete the experiment again w/allow_missing.
    response = pdelete(
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}?allow_missing=true"
    )
    assert response.status_code == 204, response.content


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
    parsed_experiment_id = created_experiment.experiment_id
    assert parsed_experiment_id is not None
    parsed_arm_ids = {arm.arm_id for arm in created_experiment.design_spec.arms}
    assert len(parsed_arm_ids) == 2

    # Verify basic response
    assert created_experiment.stopped_assignments_at is not None
    assert created_experiment.stopped_assignments_reason == StopAssignmentReason.PREASSIGNED
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
    actual_design_spec.arms[0].arm_id = None
    actual_design_spec.arms[1].arm_id = None
    actual_design_spec.experiment_id = None  # TODO remove in future
    assert actual_design_spec == request_obj.design_spec

    experiment_id = created_experiment.experiment_id
    (arm1_id, arm2_id) = [arm.arm_id for arm in created_experiment.design_spec.arms]

    # Verify database state using the ids in the returned DesignSpec.
    experiment = (
        await xngin_session.scalars(select(tables.Experiment).where(tables.Experiment.id == experiment_id))
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
            select(tables.ArmAssignment).where(tables.ArmAssignment.experiment_id == experiment_id)
        )
    ).all()
    assert len(assignments) == 100, {e.name: getattr(experiment, e.name) for e in tables.Experiment.__table__.columns}

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


def test_create_online_experiment_using_inline_schema_ds(testing_datasource_with_user, use_deterministic_random, ppost):
    datasource_id = testing_datasource_with_user.ds.id
    request_obj = make_create_online_experiment_request()

    response = ppost(
        f"/v1/m/datasources/{datasource_id}/experiments",
        params={"random_state": 42},
        content=request_obj.model_dump_json(),
    )
    assert response.status_code == 200, response.content
    created_experiment = CreateExperimentResponse.model_validate(response.json())
    parsed_experiment_id = created_experiment.experiment_id
    assert parsed_experiment_id is not None
    parsed_arm_ids = {arm.arm_id for arm in created_experiment.design_spec.arms}
    assert len(parsed_arm_ids) == 2

    # Verify basic response
    assert created_experiment.stopped_assignments_at is None
    assert created_experiment.stopped_assignments_reason is None
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
    actual_design_spec.arms[0].arm_id = None
    actual_design_spec.arms[1].arm_id = None
    actual_design_spec.experiment_id = None  # TODO remove in future
    assert actual_design_spec == request_obj.design_spec


@pytest.mark.parametrize(
    "reward_type,prior_type",
    [
        (LikelihoodTypes.BERNOULLI, PriorTypes.BETA),
        (LikelihoodTypes.NORMAL, PriorTypes.NORMAL),
        (LikelihoodTypes.BERNOULLI, PriorTypes.NORMAL),
    ],
)
def test_create_online_mab_experiment_using_inline_schema_ds(
    testing_datasource_with_user,
    ppost,
    reward_type,
    prior_type,
):
    datasource_id = testing_datasource_with_user.ds.id
    request_obj = make_create_online_bandit_experiment_request(reward_type=reward_type, prior_type=prior_type)
    response = ppost(
        f"/v1/m/datasources/{datasource_id}/experiments",
        params={"random_state": 42},
        content=request_obj.model_dump_json(),
    )
    assert response.status_code == 200, response.content
    created_experiment = CreateExperimentResponse.model_validate(response.json())
    parsed_experiment_id = created_experiment.experiment_id
    assert parsed_experiment_id is not None
    parsed_arm_ids = {arm.arm_id for arm in created_experiment.design_spec.arms}
    assert len(parsed_arm_ids) == 2

    # Verify basic response
    assert isinstance(created_experiment.design_spec, MABExperimentSpec)
    assert created_experiment.stopped_assignments_at is None
    assert created_experiment.stopped_assignments_reason is None
    assert created_experiment.state == ExperimentState.ASSIGNED
    assert created_experiment.assign_summary is None
    assert created_experiment.power_analyses is None

    for arm in created_experiment.design_spec.arms:
        assert isinstance(arm, ArmBandit)
        assert arm.arm_id is not None
        if prior_type == PriorTypes.BETA:
            assert arm.alpha is not None
            assert arm.beta is not None
        elif prior_type == PriorTypes.NORMAL:
            assert arm.mu is not None
            assert arm.covariance is not None

    # Check if the representations are equivalent
    # scrub the ids from the config for comparison
    actual_design_spec = created_experiment.design_spec.model_copy(deep=True)
    for arm in actual_design_spec.arms:
        arm.arm_id = None
        # Verify the arm parameters were initialized correctly
        assert arm.alpha == arm.alpha_init
        assert arm.beta == arm.beta_init
        assert arm.mu is None if arm.mu_init is None else [arm.mu_init]
        assert arm.covariance is None if arm.sigma_init is None else [[arm.sigma_init]]
        # Then scrub for comparing the remainder of the spec
        arm.alpha = None
        arm.beta = None
        arm.mu = None
        arm.covariance = None
    actual_design_spec.experiment_id = None  # TODO remove in future
    assert actual_design_spec == request_obj.design_spec


@pytest.mark.parametrize(
    "reward_type,prior_type",
    [
        (LikelihoodTypes.NORMAL, PriorTypes.NORMAL),
        (LikelihoodTypes.BERNOULLI, PriorTypes.NORMAL),
    ],
)
def test_create_online_cmab_experiment_using_inline_schema_ds(
    testing_datasource_with_user,
    ppost,
    reward_type,
    prior_type,
):
    datasource_id = testing_datasource_with_user.ds.id
    request_obj = make_create_online_bandit_experiment_request(
        experiment_type=ExperimentsType.CMAB_ONLINE, reward_type=reward_type, prior_type=prior_type
    )

    response = ppost(
        f"/v1/m/datasources/{datasource_id}/experiments",
        params={"random_state": 42},
        content=request_obj.model_dump_json(),
    )
    assert response.status_code == 200, response.content
    created_experiment = CreateExperimentResponse.model_validate(response.json())
    parsed_experiment_id = created_experiment.experiment_id
    assert parsed_experiment_id is not None
    parsed_arm_ids = {arm.arm_id for arm in created_experiment.design_spec.arms}
    assert len(parsed_arm_ids) == 2

    # Verify basic response
    assert isinstance(created_experiment.design_spec, CMABExperimentSpec)
    assert created_experiment.design_spec.contexts is not None and len(created_experiment.design_spec.contexts) == 2
    assert created_experiment.stopped_assignments_at is None
    assert created_experiment.stopped_assignments_reason is None
    assert created_experiment.experiment_id is not None
    assert created_experiment.state == ExperimentState.ASSIGNED
    assert created_experiment.assign_summary is None
    assert created_experiment.power_analyses is None

    for arm in created_experiment.design_spec.arms:
        assert isinstance(arm, ArmBandit)
        assert arm.arm_id is not None
        assert arm.mu is not None and len(arm.mu) == 2
        assert arm.covariance is not None and np.array(arm.covariance).size == 4

    # Check if the representations are equivalent
    # scrub the ids from the config for comparison
    actual_design_spec = created_experiment.design_spec.model_copy(deep=True)
    for arm in actual_design_spec.arms:
        arm.arm_id = None

        assert arm.alpha == arm.alpha_init
        assert arm.beta == arm.beta_init
        assert arm.mu is None if arm.mu_init is None else [arm.mu_init]
        assert arm.covariance is None if arm.sigma_init is None else [[arm.sigma_init]]

        arm.alpha = None
        arm.beta = None
        arm.mu = None
        arm.covariance = None

    assert actual_design_spec.contexts is not None
    for context in actual_design_spec.contexts:
        context.context_id = None
    actual_design_spec.experiment_id = None  # TODO remove in future
    assert actual_design_spec == request_obj.design_spec


def test_get_experiment_assignment_for_preassigned_participant(testing_experiment, pget):
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id
    assignments = testing_experiment.arm_assignments

    response = pget(f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/unassigned_id")
    assert response.status_code == 200, response.content
    assignment_response = GetParticipantAssignmentResponse.model_validate(response.json())
    assert assignment_response.experiment_id == experiment_id
    assert assignment_response.participant_id == "unassigned_id"
    assert assignment_response.assignment is None

    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/{assignments[0].participant_id}"
    )
    assert response.status_code == 200, response.content
    assignment_response = GetParticipantAssignmentResponse.model_validate(response.json())
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
    assignment_response = GetParticipantAssignmentResponse.model_validate(response.json())
    assert assignment_response.experiment_id == experiment_id
    assert assignment_response.participant_id == "new_id"
    assert assignment_response.assignment is None

    # Create a new participant assignment.
    response = pget(f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/new_id")
    assert response.status_code == 200, response.content
    assignment_response = GetParticipantAssignmentResponse.model_validate(response.json())
    assert assignment_response.experiment_id == experiment_id
    assert assignment_response.participant_id == "new_id"
    assert assignment_response.assignment is not None
    assert str(assignment_response.assignment.arm_id) in {arm.id for arm in test_experiment.arms}

    # Get back the same assignment.
    response = pget(f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/new_id")
    assert response.status_code == 200, response.content
    assignment_response2 = GetParticipantAssignmentResponse.model_validate(response.json())
    assert assignment_response2 == assignment_response

    # Make sure there's only one db entry.
    scalars = await xngin_session.scalars(
        select(tables.ArmAssignment).where(tables.ArmAssignment.experiment_id == experiment_id)
    )
    assignment = scalars.one()
    assert assignment.participant_id == "new_id"
    assert assignment.arm_id == str(assignment_response.assignment.arm_id)


async def test_get_mab_experiment_assignment_for_online_participant(
    xngin_session: AsyncSession, testing_datasource_with_user, pget
):
    test_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource_with_user.ds, ExperimentsType.MAB_ONLINE
    )
    datasource_id = test_experiment.datasource_id
    experiment_id = test_experiment.id

    # Check for an assignment that doesn't exist, but don't create it.
    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/new_id?create_if_none=false"
    )
    assert response.status_code == 200, response.content
    assignment_response = GetParticipantAssignmentResponse.model_validate(response.json())
    assert assignment_response.experiment_id == experiment_id
    assert assignment_response.participant_id == "new_id"
    assert assignment_response.assignment is None

    # Create a new participant assignment.
    response = pget(f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/new_id")
    assert response.status_code == 200, response.content
    assignment_response = GetParticipantAssignmentResponse.model_validate(response.json())
    assert assignment_response.experiment_id == experiment_id
    assert assignment_response.participant_id == "new_id"
    assert assignment_response.assignment is not None
    assert assignment_response.assignment.observed_at is None
    assert assignment_response.assignment.outcome is None
    assert str(assignment_response.assignment.arm_id) in {arm.id for arm in test_experiment.arms}

    # Get back the same assignment.
    response = pget(f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/new_id")
    assert response.status_code == 200, response.content
    assignment_response2 = GetParticipantAssignmentResponse.model_validate(response.json())
    assert assignment_response2 == assignment_response

    # Make sure there's only one db entry.
    scalars = await xngin_session.scalars(select(tables.Draw).where(tables.Draw.experiment_id == experiment_id))
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
    response = pget(f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/new_id")
    assert response.status_code == 200, response.content
    assignment_response = GetParticipantAssignmentResponse.model_validate(response.json())
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

    response = pget(f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/analyze")

    assert response.status_code == 200, response.content
    experiment_analysis = FreqExperimentAnalysisResponse.model_validate(response.json())
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
        assert {arm.arm_id for arm in analysis.arm_analyses} == {arm.id for arm in testing_experiment.arms}
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

    response = pget(f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/analyze")
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

    response = ppost(f"/v1/m/datasources/{datasource.id}/experiments/{experiment.id!s}/{endpoint}")

    # Verify
    assert response.status_code == expected_status
    # If success case, verify state was updated
    if expected_status == 204:
        expected_state = ExperimentState.ABANDONED if endpoint == "abandon" else ExperimentState.COMMITTED
        await xngin_session.refresh(experiment)
        assert experiment.state == expected_state
    # If failure case, verify the error message
    if expected_detail:
        assert response.json()["detail"] == expected_detail


async def test_delete_apikey_not_authorized(pdelete):
    """Checks for a 403 when deleting a resource that doesn't exist.

    This is equivalent to testing that a user does not have access to a datasource.

    Per AIP-135: If the user does not have permission to access the resource, regardless of whether or not it exists,
    the service must error with PERMISSION_DENIED (HTTP 403). Permission must be checked prior to checking if the
    resource exists.
    """
    response = pdelete("/v1/m/datasources/not-a-datasource/apikeys/irrelevant")
    assert response.status_code == 403


async def test_delete_apikey_authorized_and_nonexistent(testing_datasource_with_user, pdelete):
    response = pdelete(f"/v1/m/datasources/{testing_datasource_with_user.ds.id}/apikeys/sample-key-id")
    assert response.status_code == 404


async def test_delete_apikey_authorized_and_nonexistent_allow_missing(testing_datasource_with_user, pdelete):
    response = pdelete(
        f"/v1/m/datasources/{testing_datasource_with_user.ds.id}/apikeys/sample-key-id?allow_missing=true"
    )
    assert response.status_code == 204


async def test_delete_apikey_authorized_and_exists(testing_datasource_with_user, pget, pdelete):
    response = pdelete(
        f"/v1/m/datasources/{testing_datasource_with_user.ds.id}/apikeys/{testing_datasource_with_user.key_id}"
    )
    assert response.status_code == 204


async def test_delete_apikey_authorized_and_exists_allow_missing(testing_datasource_with_user, pget, pdelete):
    response = pdelete(
        f"/v1/m/datasources/{testing_datasource_with_user.ds.id}/apikeys/{testing_datasource_with_user.key_id}?allow_missing=true"
    )
    assert response.status_code == 204


async def test_delete_apikey_authorized_and_exists_idempotency(testing_datasource_with_user, pget, pdelete):
    response = pdelete(
        f"/v1/m/datasources/{testing_datasource_with_user.ds.id}/apikeys/{testing_datasource_with_user.key_id}"
    )
    assert response.status_code == 204

    response = pdelete(
        f"/v1/m/datasources/{testing_datasource_with_user.ds.id}/apikeys/{testing_datasource_with_user.key_id}"
    )
    assert response.status_code == 404

    response = pdelete(
        f"/v1/m/datasources/{testing_datasource_with_user.ds.id}/apikeys/{testing_datasource_with_user.key_id}?allow_missing=True"
    )
    assert response.status_code == 204


async def test_manage_apikeys(testing_datasource_with_user, ppost, pget, pdelete):
    ds = testing_datasource_with_user.ds
    first_key_id = testing_datasource_with_user.key_id

    response = ppost(f"/v1/m/datasources/{ds.id}/apikeys/")
    assert response.status_code == 200
    create_api_key_response = CreateApiKeyResponse.model_validate(response.json())
    assert create_api_key_response.datasource_id == ds.id
    created_key_id = create_api_key_response.id

    response = pget(f"/v1/m/datasources/{ds.id}/apikeys")
    assert response.status_code == 200
    list_api_keys_response = ListApiKeysResponse.model_validate(response.json())
    assert len(list_api_keys_response.items) == 2

    response = pdelete(f"/v1/m/datasources/{ds.id}/apikeys/{created_key_id}")
    assert response.status_code == 204

    response = pget(f"/v1/m/datasources/{ds.id}/apikeys")
    assert response.status_code == 200
    list_api_keys_response = ListApiKeysResponse.model_validate(response.json())
    assert len(list_api_keys_response.items) == 1

    response = pdelete(f"/v1/m/datasources/{ds.id}/apikeys/{first_key_id}")
    assert response.status_code == 204


async def test_experiment_webhook_integration(testing_datasource_with_user, ppost, pget):
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
    get_response = pget(f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}")
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
    assert create_response_no_webhooks.status_code == 200, create_response_no_webhooks.content

    # Verify no webhooks are associated
    created_experiment_no_webhooks = create_response_no_webhooks.json()
    assert "webhooks" in created_experiment_no_webhooks
    assert len(created_experiment_no_webhooks["webhooks"]) == 0


def test_snapshots(pget, ppost, pdelete, uget, ppatch):
    creation_response = ppost(
        "/v1/m/organizations",
        json=CreateOrganizationRequest(name="test_snapshots").model_dump(),
    )
    assert creation_response.status_code == 200, creation_response.content
    create_organization_response = CreateOrganizationResponse.model_validate(creation_response.json())

    parsed = urlparse(flags.XNGIN_DEVDWH_DSN)
    valid_dsn = PostgresDsn(
        type="postgres",
        host=parsed.hostname,
        port=parsed.port,
        user=parsed.username,
        password=RevealedStr(value=parsed.password),
        dbname=parsed.path[1:],
        sslmode="disable",
        search_path=None,
    )
    response = ppost(
        "/v1/m/datasources",
        content=CreateDatasourceRequest(
            name="test_create_datasource",
            organization_id=create_organization_response.id,
            dsn=valid_dsn,
        ).model_dump_json(),
    )
    assert response.status_code == 200, response.content
    create_datasource_response = CreateDatasourceResponse.model_validate(response.json())

    create_participant_type_response = ppost(
        f"/v1/m/datasources/{create_datasource_response.id}/participants",
        content=CreateParticipantsTypeRequest(
            participant_type="test_participant_type",
            schema_def=TESTING_DWH_PARTICIPANT_DEF,
        ).model_dump_json(),
    )
    assert create_participant_type_response.status_code == 200, create_participant_type_response.content

    response = ppost(
        f"/v1/m/datasources/{create_datasource_response.id}/experiments?chosen_n=100",
        json=CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_type=ExperimentsType.FREQ_PREASSIGNED,
                participant_type="test_participant_type",
                experiment_name="test experiment",
                description="test experiment",
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime.now(UTC) + timedelta(days=1),
                arms=[
                    Arm(arm_name="control", arm_description="Control group"),
                    Arm(arm_name="treatment", arm_description="Treatment group"),
                ],
                metrics=[DesignSpecMetricRequest(field_name="income", metric_pct_change=5)],
                strata=[],
                filters=[],
            )
        ).model_dump(mode="json"),
    )
    assert response.status_code == 200, response.content
    experiment_id = CreateExperimentResponse.model_validate_json(response.content).experiment_id

    # Experiments must be in an eligible state to be snapshotted.
    response = ppost(f"/v1/m/datasources/{create_datasource_response.id}/experiments/{experiment_id}/commit")
    assert response.status_code == 204

    # When run via tests, the TestClient that ppost() is built upon will wait for the backend handler to finish
    # all of its background tasks. Therefore this test will not observe the experiment in a "pending" state.
    response = ppost(
        f"/v1/m/organizations/{create_organization_response.id}/datasources/{create_datasource_response.id}"
        f"/experiments/{experiment_id}/snapshots"
    )
    assert response.status_code == 200, response.content
    create_snapshot_response = CreateSnapshotResponse.model_validate(response.json())

    # Force the second snapshot to fail by misconfiguring the Postgres port.
    response = ppatch(
        f"/v1/m/datasources/{create_datasource_response.id}",
        content=UpdateDatasourceRequest(
            dsn=valid_dsn.model_copy(update={"port": valid_dsn.port + 1})
        ).model_dump_json(),
    )
    assert response.status_code == 204, response.content

    response = ppost(
        f"/v1/m/organizations/{create_organization_response.id}/datasources/{create_datasource_response.id}"
        f"/experiments/{experiment_id}/snapshots"
    )
    assert response.status_code == 200, response.content
    create_bad_snapshot_response = CreateSnapshotResponse.model_validate(response.json())

    list_snapshot_response = pget(
        f"/v1/m/organizations/{create_organization_response.id}/datasources/{create_datasource_response.id}"
        f"/experiments/{experiment_id}/snapshots"
    )
    assert list_snapshot_response.status_code == 200, list_snapshot_response.content
    list_snapshot = ListSnapshotsResponse.model_validate(list_snapshot_response.json())
    assert len(list_snapshot.items) == 2, list_snapshot
    assert list_snapshot.items[0].updated_at < list_snapshot.items[1].updated_at, list_snapshot

    success_snapshot, failed_snapshot = list_snapshot.items

    assert success_snapshot.id == create_snapshot_response.id
    assert success_snapshot.experiment_id == experiment_id
    assert success_snapshot.status == "success"
    assert success_snapshot.details is None
    # Verify the snapshot data.
    analysis_response = FreqExperimentAnalysisResponse.model_validate(success_snapshot.data)
    assert analysis_response.experiment_id == experiment_id
    assert analysis_response.num_participants == 100  # chosen_n
    assert analysis_response.num_missing_participants == 0
    assert datetime.now(UTC) - analysis_response.created_at < timedelta(seconds=5)
    assert len(analysis_response.metric_analyses) == 1
    metric_analysis = analysis_response.metric_analyses[0]
    assert metric_analysis.metric_name == "income"
    # Check arm analyses.
    for analysis, arm_name, arm_description, is_baseline in zip(
        metric_analysis.arm_analyses,
        ["control", "treatment"],
        ["Control group", "Treatment group"],
        [True, False],
        strict=False,
    ):
        assert analysis.arm_id is not None
        assert analysis.arm_name == arm_name
        assert analysis.arm_description == arm_description
        assert analysis.estimate is not None
        assert analysis.t_stat is not None
        assert analysis.p_value is not None
        assert analysis.std_error > 0
        assert analysis.num_missing_values == 0
        assert analysis.is_baseline == is_baseline

    assert failed_snapshot.id == create_bad_snapshot_response.id
    assert failed_snapshot.experiment_id == experiment_id
    assert failed_snapshot.status == "failed"
    assert failed_snapshot.data is None
    assert failed_snapshot.details is not None and "OperationalError: " in failed_snapshot.details["message"]

    response = pget(
        f"/v1/m/organizations/{create_organization_response.id}/datasources/{create_datasource_response.id}"
        f"/experiments/{experiment_id}/snapshots/{success_snapshot.id}"
    )
    assert response.status_code == 200, response.content
    get_snapshot_response = GetSnapshotResponse.model_validate(response.json())
    assert get_snapshot_response.snapshot is not None
    assert get_snapshot_response.snapshot.id == success_snapshot.id
    assert get_snapshot_response.snapshot.experiment_id == success_snapshot.experiment_id
    assert get_snapshot_response.snapshot.status == success_snapshot.status
    assert get_snapshot_response.snapshot.data == success_snapshot.data

    # Attempt to read a snapshot as a user that doesn't have access to the snapshot.
    response = uget(
        f"/v1/m/organizations/{create_organization_response.id}/datasources/{create_datasource_response.id}"
        f"/experiments/{experiment_id}/snapshots/{success_snapshot.id}"
    )
    assert response.status_code == 404, response.content

    response = pdelete(
        f"/v1/m/organizations/{create_organization_response.id}/datasources/{create_datasource_response.id}"
        f"/experiments/{experiment_id}/snapshots/{success_snapshot.id}"
    )
    assert response.status_code == 204, response.content

    response = pdelete(
        f"/v1/m/organizations/{create_organization_response.id}/datasources/{create_datasource_response.id}"
        f"/experiments/{experiment_id}/snapshots/{success_snapshot.id}"
    )
    assert response.status_code == 404, response.content

    response = pget(
        f"/v1/m/organizations/{create_organization_response.id}/datasources/{create_datasource_response.id}"
        f"/experiments/{experiment_id}/snapshots/{success_snapshot.id}"
    )
    assert response.status_code == 404, response.content


def test_snapshot_on_ineligible_experiments(testing_datasource_with_user, ppost, pget):
    ds = testing_datasource_with_user.ds
    org = testing_datasource_with_user.org
    # The experiment created below is both too old and not yet committed.
    response = ppost(
        f"/v1/m/datasources/{ds.id}/experiments?chosen_n=10",
        json=CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_type=ExperimentsType.FREQ_PREASSIGNED,
                participant_type="test_participant_type",
                experiment_name="test old experiment",
                description="too old to be snapshotted",
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime.now(UTC) - timedelta(hours=25),
                arms=[
                    Arm(arm_name="control", arm_description="Control group"),
                    Arm(arm_name="treatment", arm_description="Treatment group"),
                ],
                metrics=[DesignSpecMetricRequest(field_name="income", metric_pct_change=5)],
                strata=[],
                filters=[],
            )
        ).model_dump(mode="json"),
    )
    assert response.status_code == 200, response.content
    experiment_id = CreateExperimentResponse.model_validate_json(response.content).design_spec.experiment_id

    # Assert non-committed experiments cannot be snapshotted.
    response = ppost(f"/v1/m/organizations/{org.id}/datasources/{ds.id}/experiments/{experiment_id}/snapshots")
    assert response.status_code == 422
    assert response.json()["message"] == "You can only snapshot committed experiments."

    # So commit the experiment.
    response = ppost(f"/v1/m/datasources/{ds.id}/experiments/{experiment_id}/commit")
    assert response.status_code == 204

    # Assert old experiments cannot be snapshotted.
    response = ppost(f"/v1/m/organizations/{org.id}/datasources/{ds.id}/experiments/{experiment_id}/snapshots")
    assert response.status_code == 422
    assert response.json()["message"] == "You can only snapshot active experiments."

    # But recently ended experiments can be snapshotted within a 1 day buffer.
    response = ppost(
        f"/v1/m/datasources/{ds.id}/experiments?chosen_n=10",
        json=CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_type=ExperimentsType.FREQ_PREASSIGNED,
                participant_type="test_participant_type",
                experiment_name="test just ended experiment",
                description="just ended within a day of now, so can be snapshotted",
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime.now(UTC) - timedelta(hours=23),
                arms=[
                    Arm(arm_name="control", arm_description="Control group"),
                    Arm(arm_name="treatment", arm_description="Treatment group"),
                ],
                metrics=[DesignSpecMetricRequest(field_name="income", metric_pct_change=5)],
                strata=[],
                filters=[],
            )
        ).model_dump(mode="json"),
    )
    assert response.status_code == 200, response.content
    experiment_id = CreateExperimentResponse.model_validate_json(response.content).design_spec.experiment_id
    response = ppost(f"/v1/m/datasources/{ds.id}/experiments/{experiment_id}/commit")
    assert response.status_code == 204
    response = ppost(f"/v1/m/organizations/{org.id}/datasources/{ds.id}/experiments/{experiment_id}/snapshots")
    assert response.status_code == 200
    CreateSnapshotResponse.model_validate(response.json())


def test_snapshot_with_nan(testing_datasource_with_user, ppost, pget):
    """Test that a snapshot with a NaN t-stat/p-value is handled correctly roundtrip."""
    ds = testing_datasource_with_user.ds
    org = testing_datasource_with_user.org
    response = ppost(
        f"/v1/m/datasources/{ds.id}/experiments?chosen_n=10",
        json=CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_type=ExperimentsType.FREQ_PREASSIGNED,
                participant_type="test_participant_type",
                experiment_name="test experiment",
                description="test experiment",
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime.now(UTC) + timedelta(days=1),
                arms=[
                    Arm(arm_name="control", arm_description="Control group"),
                    Arm(arm_name="treatment", arm_description="Treatment group"),
                ],
                metrics=[DesignSpecMetricRequest(field_name="is_engaged", metric_pct_change=5)],
                strata=[],
                # Force no variation in the primary metric => t-stat will be NaN
                filters=[Filter(field_name="is_engaged", relation=Relation.INCLUDES, value=[False])],
            )
        ).model_dump(mode="json"),
    )
    assert response.status_code == 200, response.content
    experiment_id = CreateExperimentResponse.model_validate_json(response.content).experiment_id

    response = ppost(f"/v1/m/datasources/{ds.id}/experiments/{experiment_id}/commit")
    assert response.status_code == 204

    # Take a snapshot.
    response = ppost(f"/v1/m/organizations/{org.id}/datasources/{ds.id}/experiments/{experiment_id}/snapshots")
    assert response.status_code == 200, response.content
    create_snapshot_response = CreateSnapshotResponse.model_validate(response.json())

    # Verify the snapshot.
    snapshot_id = create_snapshot_response.id
    response = pget(
        f"/v1/m/organizations/{org.id}/datasources/{ds.id}/experiments/{experiment_id}/snapshots/{snapshot_id}"
    )
    assert response.status_code == 200, response.content
    snapshot = GetSnapshotResponse.model_validate(response.json()).snapshot
    assert snapshot is not None
    assert snapshot.id == snapshot_id
    assert snapshot.status == "success"
    assert snapshot.experiment_id == experiment_id
    assert snapshot.data is not None
    analysis_response = FreqExperimentAnalysisResponse.model_validate(snapshot.data)
    assert analysis_response.experiment_id == experiment_id
    assert analysis_response.num_participants == 10
    assert analysis_response.num_missing_participants == 0
    assert datetime.now(UTC) - analysis_response.created_at < timedelta(seconds=5)
    assert len(analysis_response.metric_analyses) == 1
    metric_analysis = analysis_response.metric_analyses[0]
    assert metric_analysis.metric_name == "is_engaged"
    for analysis, arm_name, is_baseline in zip(
        metric_analysis.arm_analyses, ["control", "treatment"], [True, False], strict=True
    ):
        assert analysis.arm_id is not None
        assert analysis.arm_name == arm_name
        assert analysis.estimate == 0
        assert analysis.t_stat is None
        assert analysis.p_value is None
        assert analysis.std_error == 0
        assert analysis.num_missing_values == 0
        assert analysis.is_baseline == is_baseline
