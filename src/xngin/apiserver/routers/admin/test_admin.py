import base64
import collections
import copy
import csv
import io
import json
import math
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from typing import Protocol
from urllib.parse import urlparse

import numpy as np
import pytest
from deepdiff import DeepDiff
from pydantic import HttpUrl, TypeAdapter
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import flags
from xngin.apiserver.conftest import delete_seeded_users, expect_status_code
from xngin.apiserver.dns import safe_resolve
from xngin.apiserver.dwh.inspection_types import FieldDescriptor, ParticipantsSchema
from xngin.apiserver.dwh.inspections import ColumnDeleted, Drift, FieldChangedType
from xngin.apiserver.routers.admin.admin_api_converters import CREDENTIALS_UNAVAILABLE_MESSAGE
from xngin.apiserver.routers.admin.admin_api_types import (
    AddMemberToOrganizationRequest,
    AddWebhookToOrganizationRequest,
    ApiOnlyDsn,
    BqDsn,
    CreateDatasourceRequest,
    CreateOrganizationRequest,
    CreateParticipantsTypeRequest,
    DeleteExperimentDataRequest,
    FieldMetadata,
    GcpServiceAccount,
    GetExperimentForUiResponse,
    Hidden,
    InspectDatasourceTableResponse,
    PostgresDsn,
    RedshiftDsn,
    RevealedStr,
    SnapshotStatus,
    TableDeleted,
    UpdateArmRequest,
    UpdateDatasourceRequest,
    UpdateExperimentRequest,
    UpdateOrganizationWebhookRequest,
    UpdateParticipantsTypeRequest,
)
from xngin.apiserver.routers.admin.admin_common import DEFAULT_NO_DWH_SOURCE_NAME
from xngin.apiserver.routers.auth.auth_dependencies import (
    PRIVILEGED_EMAIL,
    PRIVILEGED_TOKEN_FOR_TESTING,
    UNPRIVILEGED_EMAIL,
)
from xngin.apiserver.routers.common_api_types import (
    Arm,
    ArmBandit,
    BanditExperimentAnalysisResponse,
    CMABContextInputRequest,
    CMABExperimentSpec,
    CreateExperimentRequest,
    CreateExperimentResponse,
    DataType,
    DesignSpecMetricRequest,
    ExperimentConfig,
    ExperimentsType,
    Filter,
    FreqExperimentAnalysisResponse,
    GetExperimentResponse,
    LikelihoodTypes,
    MABExperimentSpec,
    OnlineFrequentistExperimentSpec,
    PowerRequest,
    PreassignedFrequentistExperimentSpec,
    PriorTypes,
    Stratum,
)
from xngin.apiserver.routers.common_enums import (
    ExperimentState,
    Relation,
    StopAssignmentReason,
    UpdateTypeBeta,
    UpdateTypeNormal,
)
from xngin.apiserver.routers.experiments.experiments_common import fetch_fields_or_raise
from xngin.apiserver.routers.experiments.test_experiments_common import (
    insert_experiment_and_arms,
    make_create_freq_online_experiment_request,
    make_create_online_bandit_experiment_request,
    make_create_preassigned_experiment_request,
    make_createexperimentrequest_json,
    make_insertable_experiment,
)
from xngin.apiserver.settings import ParticipantsDef, RemoteDatabaseConfig
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.experiments_api_client import ExperimentsAPIClient
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_PARTICIPANT_DEF
from xngin.stats.bandit_sampling import update_arm

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


class HasName(Protocol):
    name: str


def find_ds_with_name[DSType: HasName](datasources: list[DSType], name: str) -> DSType:
    """Helper function to find a datasource with a specific name from an iterable.

    Raises StopIteration if the datasource is not found.
    """
    return next(ds for ds in datasources if ds.name == name)


async def make_freq_online_experiment(
    datasource_id: str,
    aclient: AdminAPIClient,
    end_date: datetime | None = None,
) -> GetExperimentForUiResponse:
    """Create a frequentist online experiment using our API (rather than a fixture)."""
    end_date = end_date or datetime.now(UTC) + timedelta(days=1)
    experiment_id = aclient.create_experiment(
        datasource_id=datasource_id,
        body=CreateExperimentRequest(
            design_spec=OnlineFrequentistExperimentSpec(
                experiment_type=ExperimentsType.FREQ_ONLINE,
                experiment_name="test experiment",
                description="test experiment",
                table_name=TESTING_DWH_PARTICIPANT_DEF.table_name,
                primary_key="id",
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=end_date,
                arms=[Arm(arm_name="C", arm_description="C"), Arm(arm_name="T", arm_description="T")],
                metrics=[DesignSpecMetricRequest(field_name="is_engaged", metric_pct_change=0.1)],
                strata=[],
                filters=[],
            ),
        ),
        random_state=42,
    ).data.experiment_id

    # Experiments must be in an eligible state to be snapshotted.
    aclient.commit_experiment(datasource_id=datasource_id, experiment_id=experiment_id)

    data = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=experiment_id).data
    assert isinstance(data, GetExperimentForUiResponse)
    return data


@pytest.fixture(name="testing_experiment")
async def fixture_testing_experiment(xngin_session: AsyncSession, testing_datasource) -> tables.Experiment:
    """Create a preassigned experiment directly in our app db on the datasource with proper user permissions."""
    datasource = testing_datasource.ds

    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="test experiment",
        description="test experiment",
        table_name=TESTING_DWH_PARTICIPANT_DEF.table_name,
        primary_key="id",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC) + timedelta(days=1),
        arms=[Arm(arm_name="C", arm_description="C"), Arm(arm_name="T", arm_description="T")],
        metrics=[DesignSpecMetricRequest(field_name="is_engaged", metric_pct_change=0.1)],
        strata=[],
        filters=[],
    )
    field_type_map = await fetch_fields_or_raise(datasource, design_spec)

    experiment_converter = ExperimentStorageConverter.init_from_components(
        datasource_id=datasource.id,
        organization_id=datasource.organization_id,
        experiment_type=design_spec.experiment_type,
        design_spec=design_spec,
        state=ExperimentState.COMMITTED,
        stopped_assignments_at=datetime.now(UTC),
        stopped_assignments_reason=StopAssignmentReason.PREASSIGNED,
        table_name=design_spec.table_name,
        unique_id_name=design_spec.primary_key,
        field_type_map=field_type_map,
    )
    experiment = experiment_converter.get_experiment()
    xngin_session.add(experiment)
    await xngin_session.commit()

    # Add fake assignments for each arm for real participant ids in our test data.
    arm_ids = [arm.id for arm in experiment.arms]
    arm_pop = [0] * len(arm_ids)
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
        arm_pop[i % 2] += 1
    for j, arm_id in enumerate(arm_ids):
        xngin_session.add(tables.ArmStats(arm_id=arm_id, population=arm_pop[j]))
    await xngin_session.commit()
    await xngin_session.refresh(experiment, ["arm_assignments"])
    return experiment


@pytest.fixture(name="testing_bandit_experiment")
async def fixture_testing_bandit_experiment(request, xngin_session: AsyncSession, testing_datasource):
    """Create an experiment on a test inline schema datasource with proper user permissions."""
    experiment_type, prior_type, reward_type, num_participants = request.param
    datasource = testing_datasource.ds
    experiment = await insert_experiment_and_arms(
        xngin_session, datasource, experiment_type, prior_type=prior_type, reward_type=reward_type
    )
    contexts = await experiment.awaitable_attrs.contexts
    # Add fake assignments for each arm for real participant ids in our test data.
    arm_ids = [arm.id for arm in experiment.arms]
    arm_map = {arm.id: arm for arm in experiment.arms}
    arm_id_counts = collections.Counter[str]()
    rng = np.random.default_rng(66)

    for i in range(num_participants):
        arm_id = arm_ids[i % 2]  # Alternate between the two arms
        arm_id_counts.update([arm_id])

        outcome = rng.binomial(n=1, p=0.5)  # Randomly generate outcome
        # NB: this step hijacks the algorithm to generate realistic arm parameters
        # It's not meant to be a test of the bandit algorithm
        current_params = update_arm(experiment=experiment, arm_to_update=arm_map[arm_id], outcomes=[outcome])
        match current_params:
            case UpdateTypeNormal():
                arm_map[arm_id].mu = current_params.mu
                arm_map[arm_id].covariance = current_params.covariance
            case UpdateTypeBeta():
                arm_map[arm_id].alpha = current_params.alpha
                arm_map[arm_id].beta = current_params.beta

        assignment = tables.Draw(
            experiment_id=experiment.id,
            participant_id=str(i),
            participant_type=experiment.participant_type,
            arm_id=arm_id,
            outcome=outcome,
            observed_at=datetime.now(tz=UTC),
            current_mu=current_params[0] if prior_type == PriorTypes.NORMAL else None,
            current_covariance=current_params[1] if prior_type == PriorTypes.NORMAL else None,
            current_alpha=current_params[0] if prior_type == PriorTypes.BETA else None,
            current_beta=current_params[1] if prior_type == PriorTypes.BETA else None,
            context_vals=[1.0] * len(contexts) if contexts is not None else None,
        )
        xngin_session.add(assignment)
        xngin_session.add(arm_map[arm_id])

    # Maintain ArmStats table for newly assigned arms.
    # When this test setup process is replaced by API calls,
    # this can be removed.
    for arm_id, count in arm_id_counts.items():
        stmt = (
            pg_insert(tables.ArmStats)
            .values(arm_id=arm_id, population=count)
            .on_conflict_do_update(
                index_elements=[tables.ArmStats.arm_id],
                set_={"population": tables.ArmStats.population + count},
            )
        )
        await xngin_session.execute(stmt)
    await xngin_session.commit()
    await xngin_session.refresh(experiment, ["draws", "arms"])
    return experiment


@pytest.mark.skipif(
    flags.AIRPLANE_MODE,
    reason="This test will fail in airplane mode because airplane mode treats all Admin API calls as authenticated.",
)
def test_list_orgs_unauthenticated(client):
    response = client.get("/v1/m/organizations")
    assert response.status_code == 403, response.content


def test_list_orgs_privileged(aclient: AdminAPIClient):
    response = aclient.list_organizations()
    assert response.data.items == []


def test_create_and_get_organization(aclient: AdminAPIClient):
    """Test basic organization creation."""
    # Create an organization
    org_name = "New Organization"
    create_response = aclient.create_organizations(body=CreateOrganizationRequest(name=org_name)).data

    # Fetch the organization
    org_response = aclient.get_organization(organization_id=create_response.id).data
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
    assert aclient.inspect_datasource(datasource_id=nodwh_summary.id).data.tables == []

    # Inspecting a specific table for NoDwh should fail.
    with expect_status_code(400, text="Only remote datasources may be inspected."):
        aclient.inspect_table_in_datasource(datasource_id=nodwh_summary.id, table_name="notable")


@pytest.mark.skipif(
    flags.AIRPLANE_MODE,
    reason="This test will fail in airplane mode because airplane mode treats all Admin API calls as authenticated.",
)
def test_list_orgs_unprivileged(aclient_unpriv: AdminAPIClient):
    response = aclient_unpriv.list_organizations()
    assert response.data.items == []


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
    disable_safe_resolve_check, dsn: PostgresDsn | RedshiftDsn | BqDsn, xngin_session, aclient: AdminAPIClient
):
    org_id = aclient.create_organizations(
        body=CreateOrganizationRequest(name="test_datasources_hide_credentials")
    ).data.id

    datasource_id = aclient.create_datasource(
        body=CreateDatasourceRequest(name="test_create_datasource", organization_id=org_id, dsn=dsn)
    ).data.id

    datasource_response = aclient.get_datasource(datasource_id=datasource_id).data
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
    aclient.update_datasource(datasource_id=datasource_id, body=update_request)

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
    aclient.update_datasource(datasource_id=datasource_id, body=update_request)

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
    disable_safe_resolve_check, dsn: BqDsn | RedshiftDsn | PostgresDsn, aclient: AdminAPIClient
):
    org_id = aclient.create_organizations(
        body=CreateOrganizationRequest(name="test_create_datasource_without_credentials")
    ).data.id

    with expect_status_code(422, detail_eq=CREDENTIALS_UNAVAILABLE_MESSAGE):
        aclient.create_datasource(
            body=CreateDatasourceRequest(
                name="test_create_datasource",
                organization_id=org_id,
                dsn=dsn,
            )
        )


def test_create_datasource_invalid_dns(testing_datasource, aclient: AdminAPIClient):
    """Tests that we reject insecure hostnames with a 400."""
    aclient.add_member_to_organization(
        organization_id=testing_datasource.org.id, body=AddMemberToOrganizationRequest(email=PRIVILEGED_EMAIL)
    )

    with expect_status_code(400, text="DNS resolution failed"):
        aclient.create_datasource(
            body=CreateDatasourceRequest(
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
            )
        )


def test_create_datasource_with_connectivity_check_connection_failure(
    disable_safe_resolve_check, aclient: AdminAPIClient
):
    org_id = aclient.create_organizations(
        body=CreateOrganizationRequest(name="test_create_datasource_with_preflight")
    ).data.id

    with expect_status_code(502, message_contains="password authentication failed"):
        aclient.create_datasource(
            body=CreateDatasourceRequest(
                name="test_create_datasource",
                organization_id=org_id,
                dsn=PostgresDsn(
                    host="127.0.0.1",
                    user="postgres",
                    port=5499,
                    password=RevealedStr(value="wrong-password"),
                    dbname="postgres",
                    sslmode="disable",
                    search_path=None,
                ),
            ),
            connectivity_check=True,
        )

    list_datasources = aclient.list_organization_datasources(organization_id=org_id).data
    assert len(list_datasources.items) == 1


def test_create_datasource_with_connectivity_check_disabled_by_default(
    disable_safe_resolve_check, aclient: AdminAPIClient
):
    org_id = aclient.create_organizations(
        body=CreateOrganizationRequest(name="test_create_datasource_connectivity_check_default")
    ).data.id

    aclient.create_datasource(
        body=CreateDatasourceRequest(
            name="test_create_datasource",
            organization_id=org_id,
            dsn=PostgresDsn(
                host="127.0.0.1",
                user="postgres",
                port=5499,
                password=RevealedStr(value="wrong-password"),
                dbname="postgres",
                sslmode="disable",
                search_path=None,
            ),
        )
    )


def test_create_datasource_with_connectivity_check_can_be_enabled(disable_safe_resolve_check, aclient: AdminAPIClient):
    org_id = aclient.create_organizations(
        body=CreateOrganizationRequest(name="test_create_datasource_connectivity_check_enabled")
    ).data.id

    with expect_status_code(502, message_contains="password authentication failed"):
        aclient.create_datasource(
            body=CreateDatasourceRequest(
                name="test_create_datasource",
                organization_id=org_id,
                dsn=PostgresDsn(
                    host="127.0.0.1",
                    user="postgres",
                    port=5499,
                    password=RevealedStr(value="wrong-password"),
                    dbname="postgres",
                    sslmode="disable",
                    search_path=None,
                ),
            ),
            connectivity_check=True,
        )


def test_add_member_to_org(testing_datasource, aclient: AdminAPIClient):
    """Test adding a user to an org."""
    # Add privileged user to existing organization
    aclient.add_member_to_organization(
        organization_id=testing_datasource.org.id, body=AddMemberToOrganizationRequest(email=PRIVILEGED_EMAIL)
    )

    # Add unprivileged user to existing organization
    aclient.add_member_to_organization(
        organization_id=testing_datasource.org.id, body=AddMemberToOrganizationRequest(email=UNPRIVILEGED_EMAIL)
    )

    # Adding a user to an existing organization (again)
    aclient.add_member_to_organization(
        organization_id=testing_datasource.org.id, body=AddMemberToOrganizationRequest(email=UNPRIVILEGED_EMAIL)
    )

    # Adding a new user to an existing organization
    aclient.add_member_to_organization(
        organization_id=testing_datasource.org.id, body=AddMemberToOrganizationRequest(email="newuser@example.com")
    )

    # Confirm all users added.
    org_response = aclient.get_organization(organization_id=testing_datasource.org.id)
    member_list = {u.email for u in org_response.data.users}
    assert member_list == {UNPRIVILEGED_EMAIL, PRIVILEGED_EMAIL, "newuser@example.com"}, member_list


def test_remove_member_from_org(aclient: AdminAPIClient):
    def create_organization():
        return aclient.create_organizations(body=CreateOrganizationRequest(name="test_remove_member_from_org")).data.id

    org_id = create_organization()

    def list_members():
        org_response = aclient.get_organization(organization_id=org_id)
        return {u.email: u.id for u in org_response.data.users}

    def add_member(email: str):
        return aclient.add_member_to_organization(
            organization_id=org_id, body=AddMemberToOrganizationRequest(email=email)
        ).response

    def remove_member(
        user_id: str,
        *,
        allow_missing: bool = False,
        expected_status_code: int | None = None,
    ):
        if expected_status_code is not None:
            with expect_status_code(expected_status_code):
                aclient.remove_member_from_organization(
                    organization_id=org_id,
                    user_id=user_id,
                    allow_missing=allow_missing,
                )
            return None
        return aclient.remove_member_from_organization(
            organization_id=org_id,
            user_id=user_id,
            allow_missing=allow_missing,
        ).response

    assert list_members().keys() == {PRIVILEGED_EMAIL}
    add_member(UNPRIVILEGED_EMAIL)
    member_list = list_members()
    assert member_list.keys() == {PRIVILEGED_EMAIL, UNPRIVILEGED_EMAIL}

    # 404 when trying to remove self from organization
    remove_member(member_list.get(PRIVILEGED_EMAIL), expected_status_code=404)
    assert list_members().keys() == {PRIVILEGED_EMAIL, UNPRIVILEGED_EMAIL}

    # 204 when remove existing member
    remove_member(member_list.get(UNPRIVILEGED_EMAIL))
    assert list_members().keys() == {PRIVILEGED_EMAIL}

    # 404 when removing non-existent member
    remove_member(member_list.get(UNPRIVILEGED_EMAIL), expected_status_code=404)
    assert list_members().keys() == {PRIVILEGED_EMAIL}

    # 204 when removing non-existent member w/allow-missing
    remove_member(member_list.get(UNPRIVILEGED_EMAIL), allow_missing=True)
    assert list_members().keys() == {PRIVILEGED_EMAIL}


def test_list_orgs(testing_datasource, aclient: AdminAPIClient):
    """Test listing the orgs the user is a member of."""
    response = aclient.list_organizations().data
    # User was added to the test fixture org already, so no extra org was created.
    assert len(response.items) == 1
    assert response.items[0].id == testing_datasource.org.id
    assert response.items[0].name == "test organization"


async def test_first_user_has_an_organization_created_at_login(xngin_session, aclient: AdminAPIClient):
    """Test listing the orgs by the first user of the system using aclient."""
    await delete_seeded_users(xngin_session)

    response = aclient.list_organizations().data
    assert len(response.items) == 1, response
    assert response.items[0].name == "My Organization"


async def test_first_user_has_an_organization_created_at_login_unprivileged(
    xngin_session, aclient_unpriv: AdminAPIClient
):
    """Test listing the orgs by the first user of the system using aclient_unpriv."""
    await delete_seeded_users(xngin_session)

    list_organizations_response = aclient_unpriv.list_organizations().data
    assert len(list_organizations_response.items) == 1, list_organizations_response
    org_id = list_organizations_response.items[0].id

    list_experiments_response = aclient_unpriv.list_organization_experiments(organization_id=org_id).data
    assert {exp.design_spec.experiment_type for exp in list_experiments_response.items} == {
        "freq_preassigned",
        "freq_online",
        "mab_online",
        "cmab_online",
    }


def test_datasource_lifecycle(aclient: AdminAPIClient):
    """Test creating, listing, updating a datasource."""
    # The user does not initially have any organizations.
    assert not aclient.list_organizations().data.items

    # Create an organization.
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="test_datasource_lifecycle")).data.id

    # Create datasource
    new_ds_name = "test remote ds"
    datasource_id = aclient.create_datasource(
        body=CreateDatasourceRequest(
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
        )
    ).data.id

    # List datasources
    list_ds_response = aclient.list_organization_datasources(organization_id=org_id).data

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
    aclient.update_datasource(
        datasource_id=datasource_id,
        body=UpdateDatasourceRequest(
            name=updated_ds_name,
        ),
    )

    # List datasources to confirm update
    list_ds_response = aclient.list_organization_datasources(organization_id=org_id).data
    test_dwh = find_ds_with_name(list_ds_response.items, updated_ds_name)
    # Ensure driver didn't change, just name
    assert test_dwh.id == datasource_id
    assert test_dwh.driver == "postgresql+psycopg"

    # Update DWH on the datasource
    aclient.update_datasource(
        datasource_id=datasource_id,
        body=UpdateDatasourceRequest(
            dsn=BqDsn(
                project_id="123456",
                dataset_id="ds",
                credentials=GcpServiceAccount(content=SAMPLE_GCLOUD_SERVICE_ACCOUNT_JSON),
            )
        ),
    )

    # List datasources to confirm update
    list_ds_response = aclient.list_organization_datasources(organization_id=org_id).data
    test_dwh = find_ds_with_name(list_ds_response.items, updated_ds_name)
    # Ensure driver changed, name didn't
    assert test_dwh.id == datasource_id
    assert test_dwh.driver == "bigquery"


async def test_list_datasources_ordered_by_experiment_count(
    xngin_session: AsyncSession, testing_datasource, aclient: AdminAPIClient
):
    """Datasources should be ordered by number of experiments (desc), then by name (asc)."""
    org = testing_datasource.org
    ds_a = testing_datasource.ds
    pt_config = ds_a.get_config().participants
    dwh = ds_a.get_config().dwh

    # Create two additional fake datasources in the same org.
    common_dwh_config = RemoteDatabaseConfig(participants=pt_config, type="remote", dwh=dwh)
    ds_b = tables.Datasource(id=tables.datasource_id_factory(), name="AAA datasource", organization=org)
    ds_b.set_config(common_dwh_config)
    ds_c = tables.Datasource(id=tables.datasource_id_factory(), name="ZZZ datasource", organization=org)
    ds_c.set_config(common_dwh_config)
    xngin_session.add_all([ds_b, ds_c])
    await xngin_session.commit()

    # Add experiments: ds_b gets 3, ds_a gets 1, ds_c gets 0.
    for ds, count in [(ds_b, 3), (ds_a, 1)]:
        for _ in range(count):
            await insert_experiment_and_arms(xngin_session, ds)

    items = aclient.list_organization_datasources(organization_id=org.id).data.items
    assert len(items) == 4

    # ds_b (3 experiments) first, ds_a (1 experiment) second.
    # The zero-experiment datasources are then ordered by name: the default NoDWH datasource
    # before ds_c.
    assert items[0].id == ds_b.id
    assert items[1].id == ds_a.id
    assert items[2].name == DEFAULT_NO_DWH_SOURCE_NAME
    assert items[3].id == ds_c.id


def test_datasource_errors(aclient: AdminAPIClient):
    """Test creating a datasource with various error conditions."""
    # Create an organization.
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="test_datasource_errors")).data.id

    # Test DB does not exist Error (404) - Postgres
    ds_id = aclient.create_datasource(
        body=CreateDatasourceRequest(
            organization_id=org_id,
            name="test invalid credentials",
            dsn=PostgresDsn(
                host="127.0.0.1",
                user="postgres",
                port=5499,
                password=RevealedStr(value="postgres"),
                dbname="nonexistent_db",
                sslmode="disable",
                search_path=None,
            ),
        )
    ).data.id

    with expect_status_code(404, message_contains='database "nonexistent_db" does not exist'):
        aclient.inspect_datasource(datasource_id=ds_id)

    # Test connection Error (502) - RedShift with wrong port
    ds_id = aclient.create_datasource(
        body=CreateDatasourceRequest(
            organization_id=org_id,
            name="test invalid credentials",
            dsn=RedshiftDsn(
                host="127.0.0.1",
                user="redshift",
                port=9999,  # Invalid port
                password=RevealedStr(value="redshift"),
                dbname="redshift",
                search_path=None,
            ),
        )
    ).data.id

    with expect_status_code(502, message_contains="CONNECTION ERROR"):
        aclient.inspect_datasource(datasource_id=ds_id)

    # Test connection Error (502) - PostgreSQL with wrong port
    ds_id = aclient.create_datasource(
        body=CreateDatasourceRequest(
            organization_id=org_id,
            name="test invalid credentials",
            dsn=PostgresDsn(
                host="127.0.0.1",
                user="postgres",
                port=9999,  # Invalid port
                password=RevealedStr(value="postgres"),
                dbname="postgres",
                sslmode="disable",
                search_path=None,
            ),
        )
    ).data.id

    with expect_status_code(502, message_contains="CONNECTION ERROR"):
        aclient.inspect_datasource(datasource_id=ds_id)

    # Test credential Error (502) - BigQuery with invalid service account
    gcloud_invalid = copy.deepcopy(SAMPLE_GCLOUD_SERVICE_ACCOUNT)
    ds_id = aclient.create_datasource(
        body=CreateDatasourceRequest(
            organization_id=org_id,
            name="test invalid credentials",
            dsn=BqDsn(
                project_id="some-project",
                dataset_id="ds",
                credentials=GcpServiceAccount(content=json.dumps(gcloud_invalid)),
            ),
        )
    ).data.id

    # Inspect datasource should return 502 for credential errors
    with expect_status_code(502, message_contains="CONNECTION ERROR"):
        aclient.inspect_datasource(datasource_id=ds_id)


def test_delete_datasource(testing_datasource, aclient: AdminAPIClient, aclient_unpriv: AdminAPIClient):
    """Test deleting a datasource a few different ways."""
    ds_id = testing_datasource.ds.id
    org_id = testing_datasource.org.id

    # aclient_unpriv authenticates as a user that is not in the same organization as the datasource.
    with expect_status_code(403):
        aclient_unpriv.delete_datasource(organization_id=org_id, datasource_id=ds_id)

    list_datasources1 = aclient.list_organization_datasources(organization_id=org_id).data
    assert len(list_datasources1.items) == 2, list_datasources1  # non-empty list

    # Delete the datasource as a privileged user.
    aclient.delete_datasource(organization_id=org_id, datasource_id=ds_id)

    # Assure the datasource was deleted.
    list_datasources2 = aclient.list_organization_datasources(organization_id=org_id).data
    assert len(list_datasources2.items) == 1
    assert list_datasources2.items[0].name == DEFAULT_NO_DWH_SOURCE_NAME

    # Delete the datasource a 2nd time returns 404.
    with expect_status_code(404):
        aclient.delete_datasource(organization_id=org_id, datasource_id=ds_id)

    # Delete the datasource a 2nd time returns 204 when ?allow_missing is set.
    aclient.delete_datasource(organization_id=org_id, datasource_id=ds_id, allow_missing=True)

    list_datasources3 = aclient.list_organization_datasources(organization_id=org_id).data
    assert len(list_datasources3.items) == 1, list_datasources3
    assert list_datasources3.items[0].name == DEFAULT_NO_DWH_SOURCE_NAME


async def test_webhook_lifecycle(aclient: AdminAPIClient):
    """Test creating, updating, and deleting a webhook."""
    # Create an organization.
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="test_webhook_lifecycle")).data.id

    # Create a webhook
    webhook_data = aclient.add_webhook_to_organization(
        organization_id=org_id,
        body=AddWebhookToOrganizationRequest(
            type="experiment.created",
            url="https://example.com/webhook",
            name="test webhook",
        ),
    ).data
    assert webhook_data.name == "test webhook"
    assert webhook_data.type == "experiment.created"
    assert webhook_data.url == "https://example.com/webhook"
    assert webhook_data.auth_token is not None
    webhook_id = webhook_data.id
    original_auth_token = webhook_data.auth_token

    # List webhooks to verify creation
    webhooks = aclient.list_organization_webhooks(organization_id=org_id).data.items
    assert len(webhooks) == 1
    assert webhooks[0].id == webhook_id
    assert webhooks[0].url == "https://example.com/webhook"
    assert webhooks[0].auth_token == original_auth_token

    # Regenerate the auth token
    aclient.regenerate_webhook_auth_token(organization_id=org_id, webhook_id=webhook_id)

    # List webhooks to verify auth token was changed
    webhooks = aclient.list_organization_webhooks(organization_id=org_id).data.items
    assert len(webhooks) == 1
    assert webhooks[0].auth_token != original_auth_token
    assert webhooks[0].auth_token is not None

    # Update the webhook URL
    new_url = "https://updated-example.com/webhook"
    new_name = "new name"
    aclient.update_organization_webhook(
        organization_id=org_id, webhook_id=webhook_id, body=UpdateOrganizationWebhookRequest(url=new_url, name=new_name)
    )

    # List webhooks to verify update
    webhooks = aclient.list_organization_webhooks(organization_id=org_id).data.items
    assert len(webhooks) == 1
    assert webhooks[0].url == new_url
    assert webhooks[0].name == new_name

    # Delete the webhook
    aclient.delete_webhook_from_organization(organization_id=org_id, webhook_id=webhook_id)

    # List webhooks to verify deletion
    webhooks = aclient.list_organization_webhooks(organization_id=org_id).data.items
    assert len(webhooks) == 0

    # Delete the webhook again (404)
    with expect_status_code(404):
        aclient.delete_webhook_from_organization(organization_id=org_id, webhook_id=webhook_id)

    # Delete the webhook again (204)
    aclient.delete_webhook_from_organization(organization_id=org_id, webhook_id=webhook_id, allow_missing=True)

    # Try to regenerate auth token for a non-existent webhook
    with expect_status_code(404):
        aclient.regenerate_webhook_auth_token(organization_id=org_id, webhook_id=webhook_id)

    # Try to update a non-existent webhook
    with expect_status_code(404):
        aclient.update_organization_webhook(
            organization_id=org_id,
            webhook_id=webhook_id,
            body=UpdateOrganizationWebhookRequest(url="https://should-fail.com/webhook", name="fail"),
        )

    # Try to delete a non-existent webhook
    with expect_status_code(404):
        aclient.delete_webhook_from_organization(organization_id=org_id, webhook_id=webhook_id)


def test_participants_lifecycle(testing_datasource, aclient: AdminAPIClient):
    """Test getting, creating, listing, updating, and deleting a participant type."""
    ds_id = testing_datasource.ds.id

    # Get participants
    parsed = aclient.get_participant_type(datasource_id=ds_id, participant_id="test_participant_type").data.current
    assert parsed.type == "schema"
    assert parsed.participant_type == "test_participant_type"
    assert parsed.table_name == "dwh"

    # Create participant
    create_pt_response = aclient.create_participant_type(
        datasource_id=ds_id,
        body=CreateParticipantsTypeRequest(
            participant_type="newpt",
            schema_def=ParticipantsSchema(
                table_name="dwh",
                fields=[
                    FieldDescriptor(
                        field_name="id",
                        data_type=DataType.BIGINT,
                        description="test",
                        is_unique_id=True,
                        is_strata=False,
                        is_filter=False,
                        is_metric=False,
                    )
                ],
            ),
        ),
    ).data
    assert create_pt_response.participant_type == "newpt"

    # List participants
    list_pt_response = aclient.list_participant_types(datasource_id=ds_id).data
    assert len(list_pt_response.items) == 2, list_pt_response

    # Update participant
    update_pt_response = aclient.update_participant_type(
        datasource_id=ds_id, participant_id="newpt", body=UpdateParticipantsTypeRequest(participant_type="renamedpt")
    ).data
    assert update_pt_response.participant_type == "renamedpt"

    # List participants (again)
    list_pt_response = aclient.list_participant_types(datasource_id=ds_id).data
    assert len(list_pt_response.items) == 2, list_pt_response

    # Get the named participant type
    participants_def = aclient.get_participant_type(datasource_id=ds_id, participant_id="renamedpt").data.current
    assert participants_def.participant_type == "renamedpt"

    # Delete the renamed participant type.
    aclient.delete_participant(datasource_id=ds_id, participant_id="renamedpt")

    # Delete the renamed participant type again.
    with expect_status_code(404):
        aclient.delete_participant(datasource_id=ds_id, participant_id="renamedpt")

    # Delete the renamed participant type again w/allow_missing.
    aclient.delete_participant(datasource_id=ds_id, participant_id="renamedpt", allow_missing=True)

    # Get the named participant type after it has been deleted
    with expect_status_code(404):
        aclient.get_participant_type(datasource_id=ds_id, participant_id="renamedpt")

    # Delete the testing participant type.
    aclient.delete_participant(datasource_id=ds_id, participant_id="test_participant_type")

    # Delete the testing participant type a 2nd time.
    with expect_status_code(404):
        aclient.delete_participant(datasource_id=ds_id, participant_id="test_participant_type")

    # Delete a participant type in a non-existent datasource.
    with expect_status_code(403):
        aclient.delete_participant(datasource_id="ds-not-exist", participant_id="test_participant_type")


def test_create_participants_type_without_unique_id(testing_datasource, aclient: AdminAPIClient):
    response = aclient.create_participant_type(
        datasource_id=testing_datasource.datasource_id,
        body=CreateParticipantsTypeRequest.model_construct(
            participant_type="newpt",
            schema_def=ParticipantsSchema.model_construct(
                table_name="dwh",
                fields=[
                    FieldDescriptor(
                        field_name="newf",
                        data_type=DataType.INTEGER,
                        description="test",
                        is_unique_id=False,  # previously used to require a field be set to true
                        is_strata=False,
                        is_filter=False,
                        is_metric=False,
                    )
                ],
            ),
        ),
    )
    assert response.data.participant_type == "newpt"
    assert response.data.schema_def.fields[0].is_unique_id is False


def test_get_participants_type_with_schema_drift(testing_datasource, aclient: AdminAPIClient):
    """Test schema drift detection when a column is missing from the table and a type changed."""
    ds_id = testing_datasource.ds.id
    # Initial schema: simulate a type change and a missing column
    schema = ParticipantsSchema(
        table_name="dwh",
        fields=[
            FieldDescriptor(
                field_name="id",
                data_type=DataType.INTEGER,
                description="simulating integer -> bigint",
                is_unique_id=True,
            ),
            FieldDescriptor(
                field_name="is_engaged",
                data_type=DataType.BOOLEAN,
                description="ok",
                is_filter=True,
            ),
            FieldDescriptor(
                field_name="missing_col",
                data_type=DataType.CHARACTER_VARYING,
                description="simulates a deleted column",
                is_metric=True,
            ),
        ],
    )

    # Create participant type with the initial schema
    aclient.create_participant_type(
        datasource_id=ds_id, body=CreateParticipantsTypeRequest(participant_type="pt", schema_def=schema)
    )

    # Get the participant type to fetch drift info
    get_response = aclient.get_participant_type(datasource_id=ds_id, participant_id="pt").data

    # First verify the drift is as expected.
    assert get_response.drift == Drift(
        schema_diff=[
            FieldChangedType(table_name="dwh", column_name="id", old_type=DataType.INTEGER, new_type=DataType.BIGINT),
            ColumnDeleted(table_name="dwh", column_name="missing_col"),
        ]
    )

    # Verify that the current (last known) schema is the dehydrated minimal schema.
    current = get_response.current.fields
    assert current == schema.fields

    # Verify the full proposed schema has the expected field changes.
    proposed = get_response.proposed.fields
    assert len(proposed) > len(current)
    id_field = next((f for f in proposed if f.field_name == "id"), None)
    assert id_field == schema.fields[0].model_copy(update={"data_type": DataType.BIGINT})
    is_engaged_field = next((f for f in proposed if f.field_name == "is_engaged"), None)
    assert is_engaged_field == schema.fields[1]
    missing_col_field = next((f for f in proposed if f.field_name == schema.fields[2].field_name), None)
    assert missing_col_field is None


def test_get_participants_type_bad_table(testing_datasource, aclient: AdminAPIClient):
    ds_id = testing_datasource.ds.id
    schema = ParticipantsSchema(
        table_name="deleted_dwh",
        fields=[
            FieldDescriptor(
                field_name="newf",
                data_type=DataType.INTEGER,
                description="test",
                is_unique_id=True,
            )
        ],
    )
    aclient.create_participant_type(
        datasource_id=ds_id, body=CreateParticipantsTypeRequest(participant_type="newpt", schema_def=schema)
    )
    # Now verify that the underlying table looks like it was deleted.
    get_response = aclient.get_participant_type(datasource_id=ds_id, participant_id="newpt").data
    assert get_response.drift == Drift(schema_diff=[TableDeleted(table_name=schema.table_name)])
    # And that the old known state is still returned as well.
    current_def = get_response.current
    assert current_def.participant_type == "newpt"
    assert current_def.table_name == schema.table_name
    assert current_def.fields == schema.fields
    assert get_response.proposed == current_def


async def test_lifecycle_with_db(testing_datasource, aclient: AdminAPIClient, aclient_unpriv: AdminAPIClient):
    """Exercises the admin API methods that require an external database."""
    # Add the privileged user to the organization.
    aclient.add_member_to_organization(
        organization_id=testing_datasource.org.id, body=AddMemberToOrganizationRequest(email=PRIVILEGED_EMAIL)
    )

    # Inspect the datasource.
    datasource_inspection = aclient.inspect_datasource(datasource_id=testing_datasource.ds.id).data
    assert "dwh" in datasource_inspection.tables, datasource_inspection

    # Inspect one table in the datasource.
    table_inspection = aclient.inspect_table_in_datasource(
        datasource_id=testing_datasource.ds.id, table_name="dwh"
    ).data
    assert table_inspection == InspectDatasourceTableResponse(
        # Note: create_inspect_table_response_from_table() doesn't explicitly check for uniqueness.
        primary_key_fields=["id"],
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
            FieldMetadata(field_name="is_onboarded_onetime", data_type=DataType.BOOLEAN, description=""),
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
    created_participant_type = aclient.create_participant_type(
        datasource_id=testing_datasource.ds.id,
        body=CreateParticipantsTypeRequest(
            participant_type=participant_type,
            schema_def=ParticipantsSchema(
                table_name="dwh",
                fields=[
                    FieldDescriptor(
                        field_name="id",
                        data_type=DataType.BIGINT,
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
                    FieldDescriptor(
                        field_name="is_engaged",
                        data_type=DataType.BOOLEAN,
                        description="test",
                        is_unique_id=False,
                        is_strata=False,
                        is_filter=True,
                        is_metric=True,
                    ),
                ],
            ),
        ),
    ).data
    assert created_participant_type.participant_type == participant_type

    # Create experiment using that participant type.
    create_exp_dict = make_createexperimentrequest_json()
    create_exp_request = TypeAdapter(CreateExperimentRequest).validate_python(create_exp_dict)
    create_exp_request.design_spec.design_url = HttpUrl("https://example.com/design")
    assert isinstance(create_exp_request.design_spec, PreassignedFrequentistExperimentSpec)
    create_exp_request.design_spec.filters = [
        Filter(field_name="id", relation=Relation.EXCLUDES, value=[str((2 << 52) + 1), None]),
        Filter(field_name="is_engaged", relation=Relation.INCLUDES, value=[True, False]),
    ]
    created_experiment = aclient.create_experiment(
        datasource_id=testing_datasource.ds.id, body=create_exp_request, desired_n=100
    ).data
    parsed_experiment_id = created_experiment.experiment_id
    assert parsed_experiment_id is not None
    assert created_experiment.design_spec.design_url == HttpUrl("https://example.com/design")
    assert created_experiment.stopped_assignments_at is not None
    assert created_experiment.stopped_assignments_reason == StopAssignmentReason.PREASSIGNED
    parsed_arm_ids = [arm.arm_id for arm in created_experiment.design_spec.arms]
    assert len(parsed_arm_ids) == 2
    # Confirm that the deprecated participant type is set to the empty string for new experiments
    assert created_experiment.participant_type_deprecated == ""

    # Commit the new experiment.
    aclient.commit_experiment(datasource_id=testing_datasource.ds.id, experiment_id=parsed_experiment_id)

    # Verify it committed.
    assert (
        aclient.get_experiment_for_ui(
            datasource_id=testing_datasource.ds.id, experiment_id=parsed_experiment_id
        ).data.config.state
        == ExperimentState.COMMITTED
    )

    # Attempting to abandon a committed experiment should fail
    with expect_status_code(409):
        aclient.abandon_experiment(datasource_id=testing_datasource.ds.id, experiment_id=parsed_experiment_id)

    # Verify it is still committed.
    get_exp = aclient.get_experiment_for_ui(
        datasource_id=testing_datasource.ds.id, experiment_id=parsed_experiment_id
    ).data
    assert get_exp.config.state == ExperimentState.COMMITTED
    assert get_exp.config.participant_type_deprecated == ""
    assert get_exp.participant_type is not None
    assert get_exp.participant_type.participant_type == ""

    # Update the experiment.
    aclient.update_experiment(
        datasource_id=testing_datasource.ds.id,
        experiment_id=parsed_experiment_id,
        body=UpdateExperimentRequest(name="updated"),
    )

    # Update an arm.
    updated_arm_id = parsed_arm_ids[0]
    assert updated_arm_id is not None
    aclient.update_arm(
        datasource_id=testing_datasource.ds.id,
        experiment_id=parsed_experiment_id,
        arm_id=updated_arm_id,
        body=UpdateArmRequest(name="updated arm"),
    )

    # Get that experiment.
    get_experiment_response = aclient.get_experiment_for_ui(
        datasource_id=testing_datasource.ds.id, experiment_id=parsed_experiment_id
    ).data
    assert get_experiment_response.config.experiment_id == parsed_experiment_id
    assert get_experiment_response.config.design_spec.experiment_name == "updated"
    arm = next((arm for arm in get_experiment_response.config.design_spec.arms if arm.arm_id == updated_arm_id), None)
    assert arm is not None
    assert arm.arm_name == "updated arm"

    # List org experiments.
    experiment_list = aclient.list_organization_experiments(organization_id=testing_datasource.org.id).data
    assert len(experiment_list.items) == 1, experiment_list
    experiment_config_0 = experiment_list.items[0]
    assert experiment_config_0.experiment_id == parsed_experiment_id

    # Analyze experiment
    experiment_analysis = aclient.analyze_experiment(
        datasource_id=testing_datasource.ds.id, experiment_id=parsed_experiment_id
    ).data
    assert experiment_analysis.experiment_id == parsed_experiment_id

    # Get assignments for the experiment as CSV.
    # Use aclient.client directly because generated client does not correctly handle CSV response types.
    assignments_csv_response = aclient.client.get(
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}/assignments/csv"
    )
    assert assignments_csv_response.status_code == 200, assignments_csv_response.content
    assert assignments_csv_response.headers["content-type"].startswith("text/csv")

    csv_reader = csv.reader(io.StringIO(assignments_csv_response.text))
    csv_header = next(csv_reader)
    assert csv_header == ["participant_id", "arm_id", "arm_name", "created_at", "gender"]

    assignments_csv_rows = [dict(zip(csv_header, row, strict=True)) for row in csv_reader]
    assert len(assignments_csv_rows) == 100
    for row in assignments_csv_rows:
        datetime.fromisoformat(row["created_at"])
    assert {row["arm_name"] for row in assignments_csv_rows} == {"updated arm", "treatment"}
    assert {row["arm_id"] for row in assignments_csv_rows} == set(parsed_arm_ids)

    # Unprivileged user attempts to delete the experiment
    with expect_status_code(403):
        aclient_unpriv.delete_experiment(datasource_id=testing_datasource.ds.id, experiment_id=parsed_experiment_id)

    # Delete the experiment.
    aclient.delete_experiment(datasource_id=testing_datasource.ds.id, experiment_id=parsed_experiment_id)

    # Delete the experiment again.
    with expect_status_code(404):
        aclient.delete_experiment(datasource_id=testing_datasource.ds.id, experiment_id=parsed_experiment_id)

    # Delete the experiment again w/allow_missing.
    aclient.delete_experiment(
        datasource_id=testing_datasource.ds.id, experiment_id=parsed_experiment_id, allow_missing=True
    )


async def test_abandon_experiment(testing_datasource, aclient: AdminAPIClient):
    datasource_id = testing_datasource.ds.id
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="test experiment",
        description="test experiment",
        table_name="dwh",
        primary_key="id",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC) + timedelta(days=1),
        arms=[Arm(arm_name="C", arm_description="C"), Arm(arm_name="T", arm_description="T")],
        metrics=[DesignSpecMetricRequest(field_name="is_engaged", metric_pct_change=0.1)],
        strata=[],
        filters=[],
    )
    parsed_response = aclient.create_experiment(
        datasource_id=datasource_id,
        body=CreateExperimentRequest(design_spec=design_spec),
        desired_n=1,
    ).data
    assert parsed_response.state == ExperimentState.ASSIGNED
    parsed_experiment_id = parsed_response.experiment_id

    aclient.abandon_experiment(datasource_id=datasource_id, experiment_id=parsed_experiment_id)

    response = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=parsed_experiment_id)
    assert response.data.config.state == ExperimentState.ABANDONED


async def test_power_check_with_unbalanced_arms(testing_datasource, aclient: AdminAPIClient):
    """Test power check endpoint with balanced vs unbalanced arms."""
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="test power check",
        description="test power check with unbalanced arms",
        table_name="dwh",
        primary_key="id",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC) + timedelta(days=1),
        arms=[
            Arm(arm_name="control", arm_description="Control group"),
            Arm(arm_name="treatment", arm_description="Treatment group"),
        ],
        metrics=[DesignSpecMetricRequest(field_name="current_income", metric_pct_change=0.1)],
        strata=[],
        filters=[],
    )

    # Call the power check endpoint
    power_response = aclient.power_check(
        datasource_id=testing_datasource.ds.id,
        body=PowerRequest(design_spec=design_spec, table_name="dwh", primary_key="id"),
    ).data
    assert len(power_response.analyses) == 1
    metric_analysis = power_response.analyses[0]
    assert metric_analysis.metric_spec.field_name == "current_income"
    assert metric_analysis.target_n == 474
    assert metric_analysis.sufficient_n is True

    # Now check with unbalanced arms
    design_spec.arms[0].arm_weight = 20.0
    design_spec.arms[1].arm_weight = 80.0
    power_response2 = aclient.power_check(
        datasource_id=testing_datasource.ds.id,
        body=PowerRequest(design_spec=design_spec, table_name="dwh", primary_key="id"),
    ).data
    assert len(power_response2.analyses) == 1
    metric_analysis2 = power_response2.analyses[0]
    assert metric_analysis2.metric_spec.field_name == "current_income"
    assert metric_analysis2.target_n is not None
    assert metric_analysis2.target_n > metric_analysis.target_n  # Unbalanced design requires more participants

    # And again with three arms
    design_spec.arms = [*design_spec.arms, Arm(arm_name="arm3", arm_description="Arm 3")]
    design_spec.arms[0].arm_weight = 10
    design_spec.arms[1].arm_weight = 50
    design_spec.arms[2].arm_weight = 40
    power_response3 = aclient.power_check(
        datasource_id=testing_datasource.ds.id,
        body=PowerRequest(design_spec=design_spec, table_name="dwh", primary_key="id"),
    ).data
    assert len(power_response3.analyses) == 1
    metric_analysis3 = power_response3.analyses[0]
    assert metric_analysis3.metric_spec.field_name == "current_income"
    assert metric_analysis3.target_n is not None
    # Min ratio is still 4:1 (the smallest treatment arm) as in the previous case, but the control
    # is now only 10% of the total instead of 20%, so we need more participants than before to
    # ensure that comparison with the smaller arm still has sufficient power.
    assert metric_analysis3.target_n == math.ceil(metric_analysis2.target_n * 0.2 / 0.10)


async def test_power_check_validations(testing_datasource, aclient: AdminAPIClient):
    """Test power check validations."""
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="test power check with synthesized schema",
        description="test power check using table_name and primary_key",
        table_name="dwh",
        primary_key="id",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC) + timedelta(days=1),
        arms=[
            Arm(arm_name="control", arm_description="Control group"),
            Arm(arm_name="treatment", arm_description="Treatment group"),
        ],
        metrics=[DesignSpecMetricRequest(field_name="current_income", metric_pct_change=0.1)],
        strata=[],
        filters=[],
    )

    # First check a valid power check
    datasource_id = testing_datasource.ds.id
    power_response = aclient.power_check(
        datasource_id=datasource_id,
        body=PowerRequest(design_spec=design_spec, table_name="dwh", primary_key="id"),
    ).data
    assert len(power_response.analyses) == 1
    assert power_response.analyses[0].metric_spec.field_name == "current_income"
    assert power_response.analyses[0].target_n is not None

    # Now check various failure scenarios
    with expect_status_code(404, message_contains="The table '' does not exist."):
        aclient.power_check(
            datasource_id=datasource_id, body=PowerRequest(design_spec=design_spec, table_name="", primary_key="")
        )

    with expect_status_code(422, detail_contains="columns that do not exist in the table: no_such_primary_key"):
        aclient.power_check(
            datasource_id=datasource_id,
            body=PowerRequest(design_spec=design_spec, table_name="dwh", primary_key="no_such_primary_key"),
        )

    with expect_status_code(
        422, detail_contains="columns that do not exist in the table: bad_filter, bad_metric, bad_stratum"
    ):
        bad_design_spec = design_spec.model_copy(deep=True)
        bad_design_spec.metrics = [DesignSpecMetricRequest(field_name="bad_metric", metric_pct_change=0.1)]
        bad_design_spec.strata = [Stratum(field_name="bad_stratum")]
        bad_design_spec.filters = [Filter(field_name="bad_filter", relation=Relation.INCLUDES, value=["value"])]
        aclient.power_check(
            datasource_id=datasource_id,
            body=PowerRequest(design_spec=bad_design_spec, table_name="dwh", primary_key="id"),
        )

    with expect_status_code(422, detail_contains="Invalid metric field(s): (gender). Only boolean or numeric"):
        bad_design_spec = design_spec.model_copy(deep=True)
        bad_design_spec.metrics = [DesignSpecMetricRequest(field_name="gender", metric_pct_change=0.1)]
        aclient.power_check(
            datasource_id=datasource_id,
            body=PowerRequest(design_spec=bad_design_spec, table_name="dwh", primary_key="id"),
        )


async def test_create_experiment_with_invalid_design_url(testing_datasource, aclient: AdminAPIClient):
    datasource_id = testing_datasource.ds.id
    # Work with the raw json to construct a bad request
    request = make_createexperimentrequest_json()
    request["design_spec"]["design_url"] = "example.com/"

    with expect_status_code(422, detail_contains="Input should be a valid URL, relative URL without a base"):
        aclient.create_experiment(datasource_id=datasource_id, body=request, desired_n=1)

    # Now check that a too long URL is rejected.
    request["design_spec"]["design_url"] = "http://example.com/" + "a" * 500
    with expect_status_code(422, detail_contains="URL should have at most 500 characters"):
        aclient.create_experiment(datasource_id=datasource_id, body=request, desired_n=1)

    # And we need a host.
    request["design_spec"]["design_url"] = "https://"
    with expect_status_code(422, detail_contains="Input should be a valid URL, empty host"):
        aclient.create_experiment(datasource_id=datasource_id, body=request, desired_n=1)


async def test_create_and_get_freq_preassigned_experiment(
    testing_datasource,
    use_deterministic_random,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    datasource_id = testing_datasource.ds.id
    request_obj = make_create_preassigned_experiment_request()

    created_experiment = aclient.create_experiment(
        datasource_id=datasource_id, body=request_obj, desired_n=100, random_state=42
    ).data
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
    assert actual_design_spec == request_obj.design_spec

    experiment_id = created_experiment.experiment_id
    (arm1_id, arm2_id) = [arm.arm_id for arm in created_experiment.design_spec.arms]

    # Now get the experiment using the admin API and verify config matches the created experiment.
    admin_experiment = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=experiment_id).data
    assert admin_experiment.participant_type is not None
    assert admin_experiment.participant_type.participant_type == ""
    assert admin_experiment.participant_type.hidden is True
    ui_experiment = admin_experiment.config
    diff = DeepDiff(
        created_experiment,
        ui_experiment,
        ignore_type_in_groups=[(CreateExperimentResponse, ExperimentConfig)],
    )
    assert not diff, f"Objects differ:\n{diff.pretty()}"

    # Check getting the experiment from the integration API is consistent with the created experiment.
    integration_experiment = eclient.get_experiment(api_key=testing_datasource.key, experiment_id=experiment_id).data
    diff = DeepDiff(
        created_experiment,
        integration_experiment,
        ignore_type_in_groups=[(CreateExperimentResponse, GetExperimentResponse)],
    )
    assert not diff, f"Objects differ:\n{diff.pretty()}"

    # Verify assignments were created
    actual_assignments = eclient.get_experiment_assignments(
        api_key=testing_datasource.key, experiment_id=experiment_id
    ).data
    assert len(actual_assignments.assignments) == 100

    # Check one assignment to see if it looks roughly right
    sample_assignment = actual_assignments.assignments[0]
    assert sample_assignment.arm_id in {arm1_id, arm2_id}
    assert sample_assignment.strata is not None and len(sample_assignment.strata) == 2
    for stratum in sample_assignment.strata:
        assert stratum.field_name in {"is_onboarded", "gender"}

    # Check for approximate balance in arm assignment
    num_control = sum(1 for a in actual_assignments.assignments if a.arm_id == arm1_id)
    num_treat = sum(1 for a in actual_assignments.assignments if a.arm_id == arm2_id)
    assert abs(num_control - num_treat) <= 5  # Allow some wiggle room


async def test_create_freq_preassigned_experiment_fields_use_roundtrip(
    testing_datasource,
    aclient: AdminAPIClient,
):
    datasource_id = testing_datasource.ds.id
    experiment_request = CreateExperimentRequest(
        design_spec=PreassignedFrequentistExperimentSpec(
            experiment_type="freq_preassigned",
            experiment_name="Test Experiment with Filters",
            description="Testing filters roundtrip",
            table_name="dwh",
            primary_key="id",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, 23, 59, 59, tzinfo=UTC),
            arms=[
                Arm(arm_name="control", arm_description="Control group"),
                Arm(arm_name="treatment", arm_description="Treatment group"),
            ],
            metrics=[
                DesignSpecMetricRequest(field_name="current_income", metric_pct_change=5),
                DesignSpecMetricRequest(field_name="is_engaged", metric_target=0.5),
            ],
            strata=[Stratum(field_name="ethnicity"), Stratum(field_name="baseline_income")],
            filters=[
                Filter(field_name="gender", relation=Relation.INCLUDES, value=["Male"]),
                Filter(field_name="current_income", relation=Relation.BETWEEN, value=[0.0, 100000.0]),
                Filter(field_name="is_engaged", relation=Relation.INCLUDES, value=[True, None]),
                Filter(field_name="id", relation=Relation.EXCLUDES, value=[9007199254740993, None]),
                Filter(field_name="sample_date", relation=Relation.BETWEEN, value=["2024-01-01", "2026-01-01"]),
                Filter(
                    field_name="uuid_filter", relation=Relation.EXCLUDES, value=["123e4567-e89b-12d3-a456-426614174000"]
                ),
            ],
        ),
        webhooks=[],
    )

    created_experiment = aclient.create_experiment(
        datasource_id=datasource_id, body=experiment_request, desired_n=100, random_state=42
    ).data
    experiment_id = created_experiment.experiment_id

    # Verify basic response
    spec = created_experiment.design_spec
    assert isinstance(spec, PreassignedFrequentistExperimentSpec)

    assert len(spec.metrics) == 2
    assert spec.metrics[0].field_name == "current_income"
    assert spec.metrics[0].metric_pct_change == 5
    assert spec.metrics[1].field_name == "is_engaged"
    assert spec.metrics[1].metric_target == 0.5

    assert len(spec.strata) == 2
    assert spec.strata[0].field_name == "ethnicity"
    assert spec.strata[1].field_name == "baseline_income"

    assert len(spec.filters) == 6
    a_filter = next(f for f in spec.filters if f.field_name == "gender")
    assert a_filter.field_name == "gender"
    assert a_filter.relation == Relation.INCLUDES
    assert a_filter.value == ["Male"]
    a_filter = next(f for f in spec.filters if f.field_name == "current_income")
    assert a_filter.field_name == "current_income"
    assert a_filter.relation == Relation.BETWEEN
    assert a_filter.value == [0.0, 100000.0]
    a_filter = next(f for f in spec.filters if f.field_name == "is_engaged")
    assert a_filter.field_name == "is_engaged"
    assert a_filter.relation == Relation.INCLUDES
    assert a_filter.value == [True, None]
    a_filter = next(f for f in spec.filters if f.field_name == "id")
    assert a_filter.field_name == "id"
    assert a_filter.relation == Relation.EXCLUDES
    assert a_filter.value == ["9007199254740993", None]
    a_filter = next(f for f in spec.filters if f.field_name == "sample_date")
    assert a_filter.field_name == "sample_date"
    assert a_filter.relation == Relation.BETWEEN
    assert a_filter.value == ["2024-01-01", "2026-01-01"]
    a_filter = next(f for f in spec.filters if f.field_name == "uuid_filter")
    assert a_filter.field_name == "uuid_filter"
    assert a_filter.relation == Relation.EXCLUDES
    assert a_filter.value == ["123e4567-e89b-12d3-a456-426614174000"]

    # Check getting the experiment from the UI API is consistent with the created experiment.
    experiment_for_ui = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=experiment_id).data
    diff = DeepDiff(
        created_experiment,
        experiment_for_ui.config,
        ignore_type_in_groups=[(CreateExperimentResponse, ExperimentConfig)],
    )
    assert not diff, f"Objects differ:\n{diff.pretty()}"

    # Get assignments for the experiment as CSV to verify the spec's strata fields are included.
    csv_response = aclient.client.get(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/csv",
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )
    assert csv_response.status_code == HTTPStatus.OK
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert (
        csv_response.headers["content-disposition"]
        == f'attachment; filename="experiment_{experiment_id}_assignments.csv"'
    )

    csv_lines = csv_response.text.strip().splitlines()
    assert len(csv_lines) == 101
    assert csv_lines[0] == "participant_id,arm_id,arm_name,created_at,baseline_income,ethnicity"
    for line in csv_lines[1:]:
        # CSV lines should have values for all fields
        assert len([value for value in line.split(",") if value != ""]) == 6


def test_preassigned_experiment_assign_summary_matches_get(testing_datasource, aclient: AdminAPIClient):
    """The assign_summary from create_experiment must match the persisted experiment summary."""
    datasource_id = testing_datasource.ds.id
    request_obj = make_create_preassigned_experiment_request()

    created = aclient.create_experiment(
        datasource_id=datasource_id, body=request_obj, desired_n=100, random_state=42
    ).data
    create_summary = created.assign_summary
    assert create_summary is not None

    aclient.commit_experiment(datasource_id=datasource_id, experiment_id=created.experiment_id)

    get_summary = aclient.get_experiment_for_ui(
        datasource_id=datasource_id, experiment_id=created.experiment_id
    ).data.config.assign_summary
    assert get_summary is not None

    assert create_summary.sample_size == get_summary.sample_size
    assert create_summary.balance_check == get_summary.balance_check
    assert create_summary.arm_sizes is not None
    assert get_summary.arm_sizes is not None
    assert len(create_summary.arm_sizes) == len(get_summary.arm_sizes)
    for create_arm, get_arm in zip(create_summary.arm_sizes, get_summary.arm_sizes, strict=True):
        assert create_arm.arm.arm_id == get_arm.arm.arm_id
        assert create_arm.arm.arm_name == get_arm.arm.arm_name
        assert create_arm.size == get_arm.size


def test_create_and_get_freq_online_experiment(testing_datasource, aclient: AdminAPIClient):
    datasource_id = testing_datasource.ds.id
    request_obj = make_create_freq_online_experiment_request()

    created_experiment = aclient.create_experiment(datasource_id=datasource_id, body=request_obj, random_state=42).data
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
    assert actual_design_spec == request_obj.design_spec

    # Now get the experiment using the admin API and verify config matches the created experiment.
    fetched_resp = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=parsed_experiment_id).data
    assert fetched_resp.participant_type is not None
    assert fetched_resp.participant_type.participant_type == ""
    assert fetched_resp.participant_type.hidden is True
    ui_experiment = fetched_resp.config
    diff = DeepDiff(
        created_experiment,
        ui_experiment,
        ignore_type_in_groups=[(CreateExperimentResponse, ExperimentConfig)],
    )
    assert not diff, f"Objects differ:\n{diff.pretty()}"


@pytest.mark.parametrize(
    "reward_type,prior_type",
    [
        (LikelihoodTypes.BERNOULLI, PriorTypes.BETA),
        (LikelihoodTypes.NORMAL, PriorTypes.NORMAL),
        (LikelihoodTypes.BERNOULLI, PriorTypes.NORMAL),
    ],
)
def test_create_and_get_online_mab_experiment(testing_datasource, aclient: AdminAPIClient, reward_type, prior_type):
    datasource_id = testing_datasource.ds.id
    request_obj = make_create_online_bandit_experiment_request(reward_type=reward_type, prior_type=prior_type)
    created_experiment = aclient.create_experiment(datasource_id=datasource_id, body=request_obj, random_state=42).data
    parsed_experiment_id = created_experiment.experiment_id
    assert parsed_experiment_id is not None
    parsed_arm_ids = {arm.arm_id for arm in created_experiment.design_spec.arms}
    assert len(parsed_arm_ids) == 2

    # Verify basic response
    assert isinstance(created_experiment.design_spec, MABExperimentSpec)
    assert created_experiment.stopped_assignments_at is None
    assert created_experiment.stopped_assignments_reason is None
    assert created_experiment.state == ExperimentState.ASSIGNED
    assert created_experiment.power_analyses is None

    # Verify assign summary
    assign_summary = created_experiment.assign_summary
    assert assign_summary is not None
    assert assign_summary.balance_check is None
    assert assign_summary.sample_size == 0
    assert assign_summary.arm_sizes is not None
    assert all(a.size == 0 for a in assign_summary.arm_sizes)

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
    assert actual_design_spec == request_obj.design_spec

    # Now get the experiment using the admin API and verify config matches the created experiment.
    fetched_resp = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=parsed_experiment_id).data
    assert fetched_resp.participant_type is None
    ui_experiment = fetched_resp.config
    diff = DeepDiff(
        created_experiment,
        ui_experiment,
        ignore_type_in_groups=[(CreateExperimentResponse, ExperimentConfig)],
    )
    assert not diff, f"Objects differ:\n{diff.pretty()}"


@pytest.mark.parametrize(
    "reward_type,prior_type",
    [
        (LikelihoodTypes.NORMAL, PriorTypes.NORMAL),
        (LikelihoodTypes.BERNOULLI, PriorTypes.NORMAL),
    ],
)
def test_create_online_cmab_experiment(testing_datasource, aclient: AdminAPIClient, reward_type, prior_type):
    datasource_id = testing_datasource.ds.id
    request_obj = make_create_online_bandit_experiment_request(
        experiment_type=ExperimentsType.CMAB_ONLINE, reward_type=reward_type, prior_type=prior_type
    )

    created_experiment = aclient.create_experiment(datasource_id=datasource_id, body=request_obj, random_state=42).data
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
    assert created_experiment.power_analyses is None

    # Verify assign summary
    assign_summary = created_experiment.assign_summary
    assert assign_summary is not None
    assert assign_summary.balance_check is None
    assert assign_summary.sample_size == 0
    assert assign_summary.arm_sizes is not None
    assert all(a.size == 0 for a in assign_summary.arm_sizes)

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
    assert actual_design_spec == request_obj.design_spec


@pytest.mark.parametrize(
    "experiment_type,reward_type,prior_type",
    [
        (ExperimentsType.MAB_ONLINE, LikelihoodTypes.NORMAL, PriorTypes.NORMAL),
        (ExperimentsType.MAB_ONLINE, LikelihoodTypes.BERNOULLI, PriorTypes.BETA),
        (ExperimentsType.CMAB_ONLINE, LikelihoodTypes.NORMAL, PriorTypes.NORMAL),
    ],
)
def test_create_online_mab_and_cmab_experiment_with_arm_weights(
    testing_datasource, aclient: AdminAPIClient, experiment_type, reward_type, prior_type
):
    datasource_id = testing_datasource.ds.id
    request_obj = make_create_online_bandit_experiment_request(
        experiment_type=experiment_type, reward_type=reward_type, prior_type=prior_type
    )
    # Replace mu, sigma, alpha and beta and add arm_weights instead
    arm_weights = [25.0, 75.0]
    for i, arm in enumerate(request_obj.design_spec.arms):
        assert isinstance(arm, ArmBandit)
        arm.mu_init = None
        arm.sigma_init = None
        arm.alpha_init = None
        arm.beta_init = None
        arm.arm_weight = arm_weights[i]

    created_experiment = aclient.create_experiment(datasource_id=datasource_id, body=request_obj, random_state=42).data
    parsed_experiment_id = created_experiment.experiment_id
    assert parsed_experiment_id is not None
    for arm in created_experiment.design_spec.arms:
        assert isinstance(arm, ArmBandit)
        if prior_type == PriorTypes.BETA:
            assert arm.alpha_init is not None
            assert arm.alpha is not None
            assert arm.beta_init is not None
            assert arm.beta is not None
        elif prior_type == PriorTypes.NORMAL:
            assert arm.mu_init is not None
            assert arm.mu is not None
            assert arm.sigma_init is not None
            assert arm.covariance is not None


async def test_update_experiment_invalid_impact(testing_experiment, aclient: AdminAPIClient):
    """Test updating an experiment's metadata."""
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id
    now = datetime.now(UTC)
    request = UpdateExperimentRequest.model_construct(
        name="updated name",
        description="updated desc",
        design_url="https://example.com/updated",
        start_date=now,
        end_date=now + timedelta(days=1),
        impact="invalid impact",  # type: ignore[arg-type]
        decision="new decision",
    )
    with expect_status_code(422):
        aclient.update_experiment(datasource_id=datasource_id, experiment_id=experiment_id, body=request)


async def test_update_experiment(testing_experiment, aclient: AdminAPIClient):
    """Test updating an experiment's metadata."""
    organization_id = testing_experiment.datasource.organization_id
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id
    now = datetime.now(UTC)
    request = UpdateExperimentRequest(
        name="updated name",
        description="updated desc",
        design_url="https://example.com/updated",
        start_date=now,
        end_date=now + timedelta(days=1),
        impact="high",
        decision="new decision",
    )
    aclient.update_experiment(datasource_id=datasource_id, experiment_id=experiment_id, body=request)

    updated_response = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=experiment_id)
    experiment = updated_response.data
    design_spec = experiment.config.design_spec
    assert design_spec.experiment_name == "updated name"
    assert design_spec.description == "updated desc"
    assert design_spec.design_url == HttpUrl("https://example.com/updated")
    assert design_spec.start_date == now
    assert design_spec.end_date == now + timedelta(days=1)
    assert experiment.config.impact == "high"
    assert experiment.config.decision == "new decision"

    list_experiments_response = aclient.list_organization_experiments(organization_id=organization_id)
    listing = list_experiments_response.data
    listed = next(i for i in listing.items if i.experiment_id == experiment.config.experiment_id)
    assert listed.impact == "high"
    assert listed.decision == "new decision"


@pytest.mark.parametrize("url", ["http", "http:", "http://", "http:///", "https:///", "postgres://"])
async def test_update_experiment_url_invalid(testing_experiment, aclient: AdminAPIClient, url):
    with expect_status_code(422) as status_match:
        aclient.update_experiment(
            datasource_id=testing_experiment.datasource_id,
            experiment_id=testing_experiment.id,
            body=UpdateExperimentRequest.model_construct(design_url=url),
        )
    response = status_match.http_response()
    message = response.json()["detail"][0]["msg"]
    assert message.startswith(("Input should be a valid URL", "URL scheme should be")), response.content


@pytest.mark.parametrize(
    ("url", "expected_url"),
    [
        ("http://example.com", "http://example.com/"),
        ("https://example.com", "https://example.com/"),
        ("https://drive.google.com/...?q=1", "https://drive.google.com/...?q=1"),
    ],
)
async def test_update_experiment_url_valid(testing_experiment, aclient: AdminAPIClient, url, expected_url):
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id
    aclient.update_experiment(
        datasource_id=datasource_id, experiment_id=experiment_id, body=UpdateExperimentRequest(design_url=url)
    )
    parsed_response = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=experiment_id).data
    assert parsed_response.config.design_spec.design_url is not None
    assert parsed_response.config.design_spec.design_url.encoded_string() == expected_url


async def test_update_experiment_url_null_when_empty(testing_experiment, aclient: AdminAPIClient):
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id

    aclient.update_experiment(
        datasource_id=datasource_id,
        experiment_id=experiment_id,
        body=UpdateExperimentRequest(design_url="https://example.com/"),
    )
    parsed_response = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=experiment_id).data
    assert parsed_response.config.design_spec.design_url is not None
    assert parsed_response.config.design_spec.design_url.encoded_string() == "https://example.com/"

    aclient.update_experiment(
        datasource_id=datasource_id, experiment_id=experiment_id, body=UpdateExperimentRequest(design_url="")
    )
    parsed_response = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=experiment_id).data
    assert parsed_response.config.design_spec.design_url is None, parsed_response


async def test_update_experiment_invalid(xngin_session, testing_experiment, aclient: AdminAPIClient):
    """Test experiment update validation checks."""
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id

    request = UpdateExperimentRequest(start_date=testing_experiment.end_date + timedelta(days=1))
    with expect_status_code(422, detail_eq="New start date must be before end date."):
        aclient.update_experiment(datasource_id=datasource_id, experiment_id=experiment_id, body=request)

    request = UpdateExperimentRequest(end_date=testing_experiment.start_date - timedelta(days=1))
    with expect_status_code(422, detail_eq="New end date must be after start date."):
        aclient.update_experiment(datasource_id=datasource_id, experiment_id=experiment_id, body=request)

    # Lastly check invalid experiment state
    testing_experiment.state = ExperimentState.ASSIGNED
    await xngin_session.commit()

    with expect_status_code(422, detail_eq="Experiment must have been committed to be updated."):
        aclient.update_experiment(
            datasource_id=datasource_id, experiment_id=experiment_id, body=UpdateExperimentRequest(name="updated")
        )


async def test_update_arm(testing_experiment, aclient: AdminAPIClient):
    """Test updating an arm's metadata."""
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id
    arm_id = testing_experiment.arms[0].id
    request = UpdateArmRequest(name="updated name", description="updated desc")
    aclient.update_arm(datasource_id=datasource_id, experiment_id=experiment_id, arm_id=arm_id, body=request)

    updated_response = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=experiment_id)
    design_spec = updated_response.data.config.design_spec
    arm = next((arm for arm in design_spec.arms if arm.arm_id == arm_id), None)
    assert arm is not None
    assert arm.arm_name == "updated name"
    assert arm.arm_description == "updated desc"


async def test_update_arm_invalid(xngin_session, testing_experiment, aclient: AdminAPIClient):
    """Test arm update validation checks."""
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id

    # check invalid arm id
    with expect_status_code(404, text="Arm not found."):
        aclient.update_arm(
            datasource_id=datasource_id,
            experiment_id=experiment_id,
            arm_id="invalid_id",
            body=UpdateArmRequest(name="updated"),
        )

    # check invalid experiment state
    testing_experiment.state = ExperimentState.ASSIGNED
    await xngin_session.commit()

    with expect_status_code(422, detail_eq="Experiment must have been committed to update arms."):
        aclient.update_arm(
            datasource_id=datasource_id,
            experiment_id=experiment_id,
            arm_id=testing_experiment.arms[0].id,
            body=UpdateArmRequest(name="updated"),
        )


def test_freq_experiments_analyze(testing_experiment, aclient: AdminAPIClient):
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id

    experiment_analysis = aclient.analyze_experiment(datasource_id=datasource_id, experiment_id=experiment_id).data
    assert isinstance(experiment_analysis, FreqExperimentAnalysisResponse)
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
    for metric_analysis in experiment_analysis.metric_analyses:
        # Verify arm_ids match the database model.
        assert {arm.arm_id for arm in metric_analysis.arm_analyses} == {arm.id for arm in testing_experiment.arms}
        # id=0 doesn't exist in our test data, so we'll have 1 missing value across all arms.
        assert sum([arm.num_missing_values for arm in metric_analysis.arm_analyses]) == 1
        for analysis in metric_analysis.arm_analyses:
            assert analysis.ci_lower is not None
            assert analysis.ci_upper is not None
            assert analysis.ci_lower < analysis.estimate < analysis.ci_upper
            assert analysis.mean_ci_lower is not None
            assert analysis.mean_ci_upper is not None
            assert analysis.mean_ci_lower < analysis.mean_ci_upper


@pytest.mark.parametrize(
    "testing_bandit_experiment",
    [
        (ExperimentsType.MAB_ONLINE, PriorTypes.BETA, LikelihoodTypes.BERNOULLI, 10),
        (ExperimentsType.MAB_ONLINE, PriorTypes.NORMAL, LikelihoodTypes.NORMAL, 10),
        (ExperimentsType.MAB_ONLINE, PriorTypes.NORMAL, LikelihoodTypes.BERNOULLI, 10),
    ],
    indirect=True,
)
def test_mab_experiments_analyze(testing_bandit_experiment, aclient: AdminAPIClient):
    datasource_id = testing_bandit_experiment.datasource_id
    experiment_id = testing_bandit_experiment.id

    experiment_analysis = aclient.analyze_experiment(datasource_id=datasource_id, experiment_id=experiment_id).data
    assert isinstance(experiment_analysis, BanditExperimentAnalysisResponse)
    assert experiment_analysis.experiment_id == experiment_id
    assert len(experiment_analysis.arm_analyses) == len(testing_bandit_experiment.arms)
    assert experiment_analysis.n_outcomes == 10
    assert datetime.now(UTC) - experiment_analysis.created_at < timedelta(seconds=5)
    analyses = experiment_analysis.arm_analyses
    assert {arm.arm_id for arm in analyses} == {arm.id for arm in testing_bandit_experiment.arms}
    for analysis in analyses:
        assert analysis.prior_pred_mean is not None
        assert analysis.prior_pred_stdev is not None
        assert analysis.post_pred_mean is not None
        assert analysis.post_pred_stdev is not None


@pytest.mark.parametrize(
    "testing_bandit_experiment",
    [
        (ExperimentsType.CMAB_ONLINE, PriorTypes.NORMAL, LikelihoodTypes.NORMAL, 10),
        (ExperimentsType.CMAB_ONLINE, PriorTypes.NORMAL, LikelihoodTypes.BERNOULLI, 10),
    ],
    indirect=True,
)
def test_cmab_experiments_analyze(testing_bandit_experiment, aclient: AdminAPIClient):
    datasource_id = testing_bandit_experiment.datasource_id
    experiment_id = testing_bandit_experiment.id

    experiment_analysis = aclient.analyze_cmab_experiment(
        datasource_id=datasource_id,
        experiment_id=experiment_id,
        body=(
            CMABContextInputRequest.model_validate({
                "context_inputs": [
                    {"context_id": context.id, "context_value": 1.0}
                    for context in sorted(testing_bandit_experiment.contexts, key=lambda c: c.id)
                ]
            })
        ),
    ).data
    assert isinstance(experiment_analysis, BanditExperimentAnalysisResponse)
    assert experiment_analysis.experiment_id == experiment_id
    assert len(experiment_analysis.arm_analyses) == len(testing_bandit_experiment.arms)
    assert experiment_analysis.n_outcomes == 10
    assert datetime.now(UTC) - experiment_analysis.created_at < timedelta(seconds=5)
    analyses = experiment_analysis.arm_analyses
    assert {arm.arm_id for arm in analyses} == {arm.id for arm in testing_bandit_experiment.arms}
    for analysis in analyses:
        assert analysis.prior_pred_mean is not None
        assert analysis.prior_pred_stdev is not None
        assert analysis.post_pred_mean is not None
        assert analysis.post_pred_stdev is not None


async def test_analyze_experiment_with_no_participants(testing_datasource, aclient: AdminAPIClient):
    datasource_id = testing_datasource.ds.id
    experiment_id = (await make_freq_online_experiment(datasource_id, aclient)).config.experiment_id

    with expect_status_code(422, detail_eq="No participants found for experiment."):
        aclient.analyze_experiment(datasource_id=datasource_id, experiment_id=experiment_id)


async def test_analyze_experiment_whose_assignments_have_no_dwh_data(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    datasource_id = testing_datasource.ds.id
    experiment_id = (await make_freq_online_experiment(datasource_id, aclient)).config.experiment_id

    eclient.get_assignment(api_key=testing_datasource.key, experiment_id=experiment_id, participant_id="0")

    with expect_status_code(
        422,
        detail_contains="Check that ids used in assignment are usable with your unique identifier (id)",
    ):
        aclient.analyze_experiment(datasource_id=datasource_id, experiment_id=experiment_id)


async def test_analyze_experiment_with_no_assignments_in_one_arm_yet(
    xngin_session, testing_datasource, aclient: AdminAPIClient
):
    datasource_id = testing_datasource.ds.id
    experiment_id = (await make_freq_online_experiment(datasource_id, aclient)).config.experiment_id

    # Setup: create artificial assignments directly in db to deterministically allocate them all to
    # one arm. Multiple are used for stable analysis calcs.
    arms = (await xngin_session.scalars(select(tables.Arm).where(tables.Arm.experiment_id == experiment_id))).all()
    assigned_arm_id = arms[0].id
    arm_assignments = []
    expected_num_assignments = 3
    for i in range(1, 1 + expected_num_assignments):
        arm_assignments.append(
            tables.ArmAssignment(
                experiment_id=experiment_id,
                participant_type="",
                participant_id=f"{i}",
                arm_id=assigned_arm_id,
                strata=[],
            )
        )
    xngin_session.add_all(arm_assignments)
    xngin_session.add(tables.ArmStats(arm_id=assigned_arm_id, population=expected_num_assignments))
    await xngin_session.commit()

    # Test analysis when one arm has no assignments still has the expected ArmAnalysis for each.
    analysis_response = aclient.analyze_experiment(datasource_id=datasource_id, experiment_id=experiment_id).data
    assert isinstance(analysis_response, FreqExperimentAnalysisResponse)
    assert analysis_response.experiment_id == experiment_id
    assert analysis_response.num_participants == expected_num_assignments
    assert analysis_response.num_missing_participants == 0
    assert len(analysis_response.metric_analyses) == 1
    metric_analysis = analysis_response.metric_analyses[0]
    assert metric_analysis.metric_name == "is_engaged"
    assert len(metric_analysis.arm_analyses) == 2
    for analysis in metric_analysis.arm_analyses:
        if analysis.arm_id == assigned_arm_id:
            assert analysis.estimate is not None
            assert analysis.p_value is not None
            assert analysis.t_stat is not None
            assert analysis.std_error is not None
            assert analysis.num_missing_values == 0
        else:
            assert analysis.estimate == 0
            assert analysis.p_value is None
            assert analysis.t_stat is None
            assert analysis.std_error is None
            assert analysis.num_missing_values == -1


@pytest.mark.parametrize(
    "testing_bandit_experiment",
    [
        (ExperimentsType.MAB_ONLINE, PriorTypes.BETA, LikelihoodTypes.BERNOULLI, 0),
    ],
    indirect=True,
)
def test_mab_experiments_analyze_with_no_participants(testing_bandit_experiment, aclient: AdminAPIClient):
    datasource_id = testing_bandit_experiment.datasource_id
    experiment_id = testing_bandit_experiment.id

    experiment_analysis = aclient.analyze_experiment(datasource_id=datasource_id, experiment_id=experiment_id).data
    assert isinstance(experiment_analysis, BanditExperimentAnalysisResponse)
    assert experiment_analysis.experiment_id == experiment_id
    assert len(experiment_analysis.arm_analyses) == len(testing_bandit_experiment.arms)
    assert experiment_analysis.n_outcomes == 0
    assert datetime.now(UTC) - experiment_analysis.created_at < timedelta(seconds=5)
    analyses = experiment_analysis.arm_analyses
    assert {arm.arm_id for arm in analyses} == {arm.id for arm in testing_bandit_experiment.arms}
    for analysis in analyses:
        assert analysis.prior_pred_mean is not None
        assert analysis.prior_pred_stdev is not None
        assert analysis.post_pred_mean == analysis.prior_pred_mean
        assert analysis.post_pred_stdev == analysis.prior_pred_stdev


@pytest.mark.parametrize(
    "endpoint,initial_state,expected_status,expected_detail",
    [
        ("commit", ExperimentState.ASSIGNED, 204, None),  # Success case
        ("commit", ExperimentState.COMMITTED, 204, None),  # No-op
        ("commit", ExperimentState.DESIGNING, 409, "Invalid state: designing"),
        ("commit", ExperimentState.ABORTED, 409, "Invalid state: aborted"),
        ("abandon", ExperimentState.DESIGNING, 204, None),  # Success case
        ("abandon", ExperimentState.ASSIGNED, 204, None),  # Success case
        ("abandon", ExperimentState.ABANDONED, 204, None),  # No-op
        ("abandon", ExperimentState.COMMITTED, 409, "Invalid state: committed"),
    ],
)
async def test_admin_experiment_state_setting(
    xngin_session: AsyncSession,
    testing_datasource,
    endpoint,
    initial_state,
    expected_status,
    expected_detail,
    aclient: AdminAPIClient,
):
    # Initialize our state with an existing experiment who's state we want to modify.
    datasource = testing_datasource.ds
    experiment, _ = await make_insertable_experiment(datasource, initial_state)
    xngin_session.add(experiment)
    await xngin_session.commit()

    if expected_detail:
        expect_kwargs = {"text": expected_detail}
    else:
        expect_kwargs = {}

    if endpoint == "commit":
        if expected_status == 204:
            response = aclient.commit_experiment(datasource_id=datasource.id, experiment_id=str(experiment.id)).response
        else:
            with expect_status_code(expected_status, **expect_kwargs):
                aclient.commit_experiment(datasource_id=datasource.id, experiment_id=str(experiment.id))
    elif expected_status == 204:
        response = aclient.abandon_experiment(datasource_id=datasource.id, experiment_id=str(experiment.id)).response
    else:
        with expect_status_code(expected_status, **expect_kwargs):
            aclient.abandon_experiment(datasource_id=datasource.id, experiment_id=str(experiment.id))

    # Verify
    if expected_status == 204:
        assert response.status_code == expected_status
        expected_state = ExperimentState.ABANDONED if endpoint == "abandon" else ExperimentState.COMMITTED
        await xngin_session.refresh(experiment)
        assert experiment.state == expected_state


async def test_delete_apikey_not_authorized(aclient: AdminAPIClient):
    """Checks for a 403 when deleting a resource that doesn't exist.

    This is equivalent to testing that a user does not have access to a datasource.

    Per AIP-135: If the user does not have permission to access the resource, regardless of whether or not it exists,
    the service must error with PERMISSION_DENIED (HTTP 403). Permission must be checked prior to checking if the
    resource exists.
    """
    with expect_status_code(403):
        aclient.delete_api_key(datasource_id="not-a-datasource", api_key_id="irrelevant")


async def test_delete_apikey_authorized_and_nonexistent(testing_datasource, aclient: AdminAPIClient):
    with expect_status_code(404):
        aclient.delete_api_key(datasource_id=testing_datasource.ds.id, api_key_id="sample-key-id")


async def test_delete_apikey_authorized_and_nonexistent_allow_missing(testing_datasource, aclient: AdminAPIClient):
    aclient.delete_api_key(datasource_id=testing_datasource.ds.id, api_key_id="sample-key-id", allow_missing=True)


async def test_delete_apikey_authorized_and_exists(testing_datasource, aclient: AdminAPIClient):
    aclient.delete_api_key(datasource_id=testing_datasource.ds.id, api_key_id=testing_datasource.key_id)


async def test_delete_apikey_authorized_and_exists_allow_missing(testing_datasource, aclient: AdminAPIClient):
    aclient.delete_api_key(
        datasource_id=testing_datasource.ds.id,
        api_key_id=testing_datasource.key_id,
        allow_missing=True,
    )


async def test_delete_apikey_authorized_and_exists_idempotency(testing_datasource, aclient: AdminAPIClient):
    aclient.delete_api_key(datasource_id=testing_datasource.ds.id, api_key_id=testing_datasource.key_id)

    with expect_status_code(404):
        aclient.delete_api_key(datasource_id=testing_datasource.ds.id, api_key_id=testing_datasource.key_id)

    aclient.delete_api_key(
        datasource_id=testing_datasource.ds.id,
        api_key_id=testing_datasource.key_id,
        allow_missing=True,
    )


async def test_manage_apikeys(testing_datasource, aclient: AdminAPIClient):
    ds = testing_datasource.ds
    first_key_id = testing_datasource.key_id

    create_api_key_response = aclient.create_api_key(datasource_id=ds.id).data
    assert create_api_key_response.datasource_id == ds.id
    created_key_id = create_api_key_response.id

    list_api_keys_response = aclient.list_api_keys(datasource_id=ds.id).data
    assert len(list_api_keys_response.items) == 2

    aclient.delete_api_key(datasource_id=ds.id, api_key_id=created_key_id)

    list_api_keys_response = aclient.list_api_keys(datasource_id=ds.id).data
    assert len(list_api_keys_response.items) == 1

    aclient.delete_api_key(datasource_id=ds.id, api_key_id=first_key_id)


async def test_experiment_webhook_integration(testing_datasource, aclient: AdminAPIClient):
    """Test creating an experiment with webhook associations and verifying webhook IDs in response."""
    org_id = testing_datasource.org.id
    datasource_id = testing_datasource.ds.id

    # Create two webhooks in the organization
    webhook1_response = aclient.add_webhook_to_organization(
        organization_id=org_id,
        body=AddWebhookToOrganizationRequest(
            type="experiment.created",
            name="Test Webhook 1",
            url="https://example.com/webhook1",
        ),
    ).data
    webhook1_id = webhook1_response.id

    aclient.add_webhook_to_organization(
        organization_id=org_id,
        body=AddWebhookToOrganizationRequest(
            type="experiment.created",
            name="Test Webhook 2",
            url="https://example.com/webhook2",
        ),
    )

    # Create an experiment with only the first webhook using proper Pydantic models
    experiment_request = CreateExperimentRequest(
        design_spec=PreassignedFrequentistExperimentSpec(
            experiment_type=ExperimentsType.FREQ_PREASSIGNED,
            experiment_name="Test Experiment with Webhook",
            description="Testing webhook integration",
            table_name="dwh",
            primary_key="id",
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

    create_response = aclient.create_experiment(
        datasource_id=datasource_id, body=experiment_request, desired_n=100
    ).data

    # Verify the create response includes the webhook
    assert len(create_response.webhooks) == 1
    assert create_response.webhooks[0] == webhook1_id

    # Get the experiment ID for further testing
    experiment_id = create_response.experiment_id

    # Get the experiment and verify webhook is included
    get_response = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=experiment_id)
    experiment = get_response.data
    assert len(experiment.config.webhooks) == 1
    assert experiment.config.webhooks[0] == webhook1_id

    # Test creating an experiment with no webhooks using proper Pydantic models
    experiment_request_no_webhooks = CreateExperimentRequest(
        design_spec=PreassignedFrequentistExperimentSpec(
            experiment_type=ExperimentsType.FREQ_PREASSIGNED,
            experiment_name="Test Experiment without Webhooks",
            description="Testing no webhook integration",
            table_name="dwh",
            primary_key="id",
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
        # No webhooks field - should default to empty list
    )

    create_response_no_webhooks = aclient.create_experiment(
        datasource_id=datasource_id, body=experiment_request_no_webhooks, desired_n=100
    ).data

    # Verify no webhooks are associated
    assert len(create_response_no_webhooks.webhooks) == 0


def test_snapshots(aclient: AdminAPIClient, aclient_unpriv: AdminAPIClient):
    creation_response = aclient.create_organizations(body=CreateOrganizationRequest(name="test_snapshots"))
    create_organization_response = creation_response.data

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
    create_datasource_response = aclient.create_datasource(
        body=CreateDatasourceRequest(
            name="test_create_datasource",
            organization_id=create_organization_response.id,
            dsn=valid_dsn,
        )
    ).data

    aclient.create_participant_type(
        datasource_id=create_datasource_response.id,
        body=CreateParticipantsTypeRequest(
            participant_type="test_participant_type",
            schema_def=TESTING_DWH_PARTICIPANT_DEF,
        ),
    )

    experiment_id = aclient.create_experiment(
        datasource_id=create_datasource_response.id,
        body=CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_type=ExperimentsType.FREQ_PREASSIGNED,
                table_name="dwh",
                primary_key="id",
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
            ),
        ),
        desired_n=100,
    ).data.experiment_id

    # Experiments must be in an eligible state to be snapshotted.
    aclient.commit_experiment(datasource_id=create_datasource_response.id, experiment_id=experiment_id)

    # In tests, the underlying TestClient waits for the backend handler before returning.
    # to finish all of its background tasks. Therefore this test will not observe the experiment in a "pending" state.
    create_snapshot_response = aclient.create_snapshot(
        organization_id=create_organization_response.id,
        datasource_id=create_datasource_response.id,
        experiment_id=experiment_id,
    ).data

    # Force the second snapshot to fail by misconfiguring the Postgres port.
    aclient.update_datasource(
        datasource_id=create_datasource_response.id,
        body=UpdateDatasourceRequest(dsn=valid_dsn.model_copy(update={"port": valid_dsn.port + 1})),
    )

    create_bad_snapshot_response = aclient.create_snapshot(
        organization_id=create_organization_response.id,
        datasource_id=create_datasource_response.id,
        experiment_id=experiment_id,
    ).data

    # get the snapshot we just created and verify it is failed
    get_snapshot_response = aclient.get_snapshot(
        organization_id=create_organization_response.id,
        datasource_id=create_datasource_response.id,
        experiment_id=experiment_id,
        snapshot_id=create_bad_snapshot_response.id,
    ).data
    assert get_snapshot_response.snapshot.status == SnapshotStatus.FAILED
    assert get_snapshot_response.snapshot.data is None

    list_snapshot_response = aclient.list_snapshots(
        organization_id=create_organization_response.id,
        datasource_id=create_datasource_response.id,
        experiment_id=experiment_id,
    )
    list_snapshot = list_snapshot_response.data
    assert len(list_snapshot.items) == 2, list_snapshot
    assert list_snapshot.items[0].updated_at > list_snapshot.items[1].updated_at, list_snapshot

    failed_snapshot, success_snapshot = list_snapshot.items
    assert list_snapshot.latest_failure == failed_snapshot.updated_at

    assert success_snapshot.id == create_snapshot_response.id
    assert success_snapshot.experiment_id == experiment_id
    assert success_snapshot.status == "success"
    assert success_snapshot.details is None
    # Verify the snapshot data.
    analysis_response = FreqExperimentAnalysisResponse.model_validate(success_snapshot.data)
    assert analysis_response.experiment_id == experiment_id
    assert analysis_response.num_participants == 100  # desired_n
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
        assert analysis.std_error is not None
        assert analysis.num_missing_values == 0
        assert analysis.is_baseline == is_baseline

    assert failed_snapshot.id == create_bad_snapshot_response.id
    assert failed_snapshot.experiment_id == experiment_id
    assert failed_snapshot.status == "failed"
    assert failed_snapshot.data is None
    assert failed_snapshot.details is not None and "OperationalError: " in failed_snapshot.details["message"]

    get_snapshot_response = aclient.get_snapshot(
        organization_id=create_organization_response.id,
        datasource_id=create_datasource_response.id,
        experiment_id=experiment_id,
        snapshot_id=success_snapshot.id,
    ).data
    assert get_snapshot_response.snapshot is not None
    assert get_snapshot_response.snapshot.id == success_snapshot.id
    assert get_snapshot_response.snapshot.experiment_id == success_snapshot.experiment_id
    assert get_snapshot_response.snapshot.status == success_snapshot.status
    assert get_snapshot_response.snapshot.data == success_snapshot.data

    # list snapshots with empty string in status_ param
    with expect_status_code(422):
        aclient.list_snapshots(
            organization_id=create_organization_response.id,
            datasource_id=create_datasource_response.id,
            experiment_id=experiment_id,
            status_=[""],  # type: ignore
        )

    # list snapshots filtered for running
    list_snapshot_response = aclient.list_snapshots(
        organization_id=create_organization_response.id,
        datasource_id=create_datasource_response.id,
        experiment_id=experiment_id,
        status_=[SnapshotStatus.RUNNING],
    )
    list_snapshot = list_snapshot_response.data
    assert len(list_snapshot.items) == 0, list_snapshot
    assert list_snapshot.latest_failure == failed_snapshot.updated_at, list_snapshot

    # list snapshots restricted to success
    list_snapshot_response = aclient.list_snapshots(
        organization_id=create_organization_response.id,
        datasource_id=create_datasource_response.id,
        experiment_id=experiment_id,
        status_=[SnapshotStatus.SUCCESS],
    )
    list_snapshot = list_snapshot_response.data
    assert len(list_snapshot.items) == 1, list_snapshot
    assert list_snapshot.latest_failure == failed_snapshot.updated_at, list_snapshot

    # list snapshots restricted to failed
    list_snapshot_response = aclient.list_snapshots(
        organization_id=create_organization_response.id,
        datasource_id=create_datasource_response.id,
        experiment_id=experiment_id,
        status_=[SnapshotStatus.FAILED],
    )
    list_snapshot = list_snapshot_response.data
    assert len(list_snapshot.items) == 1, list_snapshot
    assert list_snapshot.latest_failure == list_snapshot.items[0].updated_at, list_snapshot
    assert list_snapshot.latest_failure == failed_snapshot.updated_at, list_snapshot

    # Attempt to read a snapshot as a user that doesn't have access to the snapshot.
    with expect_status_code(404):
        aclient_unpriv.get_snapshot(
            organization_id=create_organization_response.id,
            datasource_id=create_datasource_response.id,
            experiment_id=experiment_id,
            snapshot_id=success_snapshot.id,
        )

    with expect_status_code(204):
        aclient.delete_snapshot(
            _organization_id=create_organization_response.id,
            datasource_id=create_datasource_response.id,
            experiment_id=experiment_id,
            snapshot_id=success_snapshot.id,
        )

    with expect_status_code(404):
        aclient.delete_snapshot(
            _organization_id=create_organization_response.id,
            datasource_id=create_datasource_response.id,
            experiment_id=experiment_id,
            snapshot_id=success_snapshot.id,
        )

    with expect_status_code(404):
        aclient.get_snapshot(
            organization_id=create_organization_response.id,
            datasource_id=create_datasource_response.id,
            experiment_id=experiment_id,
            snapshot_id=success_snapshot.id,
        )


def test_snapshot_on_ineligible_experiments(testing_datasource, aclient: AdminAPIClient):
    ds = testing_datasource.ds
    org = testing_datasource.org
    # The experiment created below is both too old and not yet committed.
    experiment_id = aclient.create_experiment(
        datasource_id=ds.id,
        body=CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_type=ExperimentsType.FREQ_PREASSIGNED,
                table_name="dwh",
                primary_key="id",
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
            ),
        ),
        desired_n=20,
    ).data.experiment_id

    # Assert non-committed experiments cannot be snapshotted.
    with expect_status_code(422, detail_eq="You can only snapshot committed experiments."):
        aclient.create_snapshot(organization_id=org.id, datasource_id=ds.id, experiment_id=experiment_id)

    # So commit the experiment.
    aclient.commit_experiment(datasource_id=ds.id, experiment_id=experiment_id)

    # Assert old experiments cannot be snapshotted.
    with expect_status_code(422, detail_eq="You can only snapshot active experiments."):
        aclient.create_snapshot(organization_id=org.id, datasource_id=ds.id, experiment_id=experiment_id)

    # But recently ended experiments can be snapshotted within a 1 day buffer.
    experiment_id = aclient.create_experiment(
        datasource_id=ds.id,
        body=CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_type=ExperimentsType.FREQ_PREASSIGNED,
                table_name="dwh",
                primary_key="id",
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
            ),
        ),
        desired_n=20,
    ).data.experiment_id
    aclient.commit_experiment(datasource_id=ds.id, experiment_id=experiment_id)
    aclient.create_snapshot(organization_id=org.id, datasource_id=ds.id, experiment_id=experiment_id)


def test_snapshot_with_nan(testing_datasource, aclient: AdminAPIClient):
    """Test that a snapshot with a NaN t-stat/p-value is handled correctly roundtrip."""
    ds = testing_datasource.ds
    org = testing_datasource.org
    experiment_id = aclient.create_experiment(
        datasource_id=ds.id,
        body=CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_type=ExperimentsType.FREQ_PREASSIGNED,
                table_name="dwh",
                primary_key="id",
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
            ),
        ),
        desired_n=10,
    ).data.experiment_id

    aclient.commit_experiment(datasource_id=ds.id, experiment_id=experiment_id)

    # Take a snapshot.
    create_snapshot_response = aclient.create_snapshot(
        organization_id=org.id, datasource_id=ds.id, experiment_id=experiment_id
    ).data

    # Verify the snapshot.
    snapshot_id = create_snapshot_response.id
    snapshot = aclient.get_snapshot(
        organization_id=org.id, datasource_id=ds.id, experiment_id=experiment_id, snapshot_id=snapshot_id
    ).data.snapshot
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


async def test_logout_updates_last_logout_timestamp(xngin_session: AsyncSession, aclient: AdminAPIClient):
    """Test that logout endpoint updates user's last_logout timestamp and returns 204."""
    user = await xngin_session.scalar(select(tables.User).where(tables.User.email == PRIVILEGED_EMAIL))
    assert user is not None

    initial_last_logout = user.last_logout
    response = aclient.logout().response
    assert response.content == b""
    await xngin_session.refresh(user)
    assert user.last_logout > initial_last_logout
    assert datetime.now(UTC) - user.last_logout < timedelta(seconds=60)


async def test_delete_experiment_data_not_authorized(client):
    """Test that deleting experiment data without authorization returns 401."""
    response = client.request(
        "DELETE",
        "/v1/m/datasources/not-a-datasource/experiments/not-an-experiment/data",
        json=DeleteExperimentDataRequest(snapshots=True).model_dump(),
        headers={"Authorization": "Bearer fake-token"},
    )
    assert response.status_code == 401


async def test_delete_experiment_data_experiment_not_found(testing_datasource, aclient: AdminAPIClient):
    """Test that deleting data for a non-existent experiment returns 404."""
    ds_id = testing_datasource.ds.id
    with expect_status_code(404):
        aclient.delete_experiment_data(
            datasource_id=ds_id, experiment_id="not-an-experiment", body=DeleteExperimentDataRequest(snapshots=True)
        )


async def test_delete_experiment_data_assignments(
    xngin_session: AsyncSession, testing_experiment: tables.Experiment, testing_datasource, client
):
    """Test deleting arm assignments for an experiment."""
    ds_id = testing_datasource.ds.id
    experiment_id = testing_experiment.id

    # Verify assignments exist before deletion
    assignments_before = await xngin_session.scalars(
        select(tables.ArmAssignment).where(tables.ArmAssignment.experiment_id == experiment_id)
    )
    assert len(list(assignments_before)) > 0

    # Delete assignments
    client.request(
        "DELETE",
        f"/v1/m/datasources/{ds_id}/experiments/{experiment_id}/data",
        json=DeleteExperimentDataRequest(assignments=True).model_dump(),
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )

    # Verify assignments are deleted
    xngin_session.expire_all()
    assignments_after = await xngin_session.scalars(
        select(tables.ArmAssignment).where(tables.ArmAssignment.experiment_id == experiment_id)
    )
    assert len(list(assignments_after)) == 0

    # Verify arm_stats rows are deleted
    arm_stats_after = (
        await xngin_session.scalars(
            select(tables.ArmStats).where(
                tables.ArmStats.arm_id.in_(select(tables.Arm.id).where(tables.Arm.experiment_id == experiment_id))
            )
        )
    ).all()
    assert len(arm_stats_after) == 0


async def test_delete_experiment_cascades_arm_stats(
    xngin_session: AsyncSession,
    testing_experiment: tables.Experiment,
    testing_datasource,
    aclient: AdminAPIClient,
):
    """Test that deleting an entire experiment cascade-deletes its arm_stats rows."""
    ds_id = testing_datasource.ds.id
    experiment_id = testing_experiment.id
    arm_ids = [arm.id for arm in testing_experiment.arms]

    # Verify arm_stats exist before deletion
    arm_stats_before = (
        await xngin_session.scalars(select(tables.ArmStats).where(tables.ArmStats.arm_id.in_(arm_ids)))
    ).all()
    assert len(arm_stats_before) > 0

    aclient.delete_experiment(datasource_id=ds_id, experiment_id=experiment_id)

    # Verify arm_stats rows were cascade-deleted
    xngin_session.expire_all()
    arm_stats_after = (
        await xngin_session.scalars(select(tables.ArmStats).where(tables.ArmStats.arm_id.in_(arm_ids)))
    ).all()
    assert len(arm_stats_after) == 0


@pytest.mark.parametrize(
    "testing_bandit_experiment",
    [(ExperimentsType.MAB_ONLINE, PriorTypes.BETA, LikelihoodTypes.BERNOULLI, 10)],
    indirect=True,
)
async def test_delete_experiment_data_draws(
    testing_bandit_experiment: tables.Experiment,
    testing_datasource,
    aclient: AdminAPIClient,
):
    """Test deleting draws for a bandit experiment."""
    ds_id = testing_datasource.ds.id
    experiment_id = testing_bandit_experiment.id

    # Verify draws exist before deletion
    experiment = aclient.get_experiment_for_ui(datasource_id=ds_id, experiment_id=experiment_id).data
    assert experiment.config is not None
    assert experiment.config.assign_summary is not None
    assert experiment.config.assign_summary.arm_sizes is not None
    assert len(experiment.config.assign_summary.arm_sizes) > 0
    assert all(a.size > 0 for a in experiment.config.assign_summary.arm_sizes)

    # Delete draws
    aclient.delete_experiment_data(
        datasource_id=ds_id, experiment_id=experiment_id, body=DeleteExperimentDataRequest(assignments=True)
    )

    # Verify zero arm sizes after data deletion
    experiment = aclient.get_experiment_for_ui(datasource_id=ds_id, experiment_id=experiment_id).data
    assert experiment.config is not None
    assert experiment.config.assign_summary is not None
    assert experiment.config.assign_summary.arm_sizes is not None
    assert len(experiment.config.assign_summary.arm_sizes) > 0
    assert all(a.size == 0 for a in experiment.config.assign_summary.arm_sizes)


async def test_delete_experiment_data_snapshots(
    xngin_session: AsyncSession, testing_experiment: tables.Experiment, testing_datasource, client
):
    """Test deleting snapshots for an experiment."""
    ds_id = testing_datasource.ds.id
    experiment_id = testing_experiment.id

    # Create a snapshot directly in the database
    snapshot = tables.Snapshot(experiment_id=experiment_id)
    xngin_session.add(snapshot)
    await xngin_session.commit()

    # Verify snapshot exists before deletion
    snapshots_before = await xngin_session.scalars(
        select(tables.Snapshot).where(tables.Snapshot.experiment_id == experiment_id)
    )
    assert len(list(snapshots_before)) > 0

    # Delete snapshots
    client.request(
        "DELETE",
        f"/v1/m/datasources/{ds_id}/experiments/{experiment_id}/data",
        json=DeleteExperimentDataRequest(snapshots=True).model_dump(),
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )

    # Verify snapshots are deleted
    xngin_session.expire_all()
    snapshots_after = await xngin_session.scalars(
        select(tables.Snapshot).where(tables.Snapshot.experiment_id == experiment_id)
    )
    assert len(list(snapshots_after)) == 0


async def test_delete_experiment_data_multiple(
    xngin_session: AsyncSession, testing_experiment: tables.Experiment, testing_datasource, client
):
    """Test deleting multiple data types at once."""
    ds_id = testing_datasource.ds.id
    experiment_id = testing_experiment.id

    # Create a snapshot
    snapshot = tables.Snapshot(experiment_id=experiment_id)
    xngin_session.add(snapshot)
    await xngin_session.commit()

    # Verify data exists before deletion
    assignments_before = list(
        await xngin_session.scalars(
            select(tables.ArmAssignment).where(tables.ArmAssignment.experiment_id == experiment_id)
        )
    )
    snapshots_before = list(
        await xngin_session.scalars(select(tables.Snapshot).where(tables.Snapshot.experiment_id == experiment_id))
    )
    assert len(assignments_before) > 0
    assert len(snapshots_before) > 0

    # Delete both assignments and snapshots
    client.request(
        "DELETE",
        f"/v1/m/datasources/{ds_id}/experiments/{experiment_id}/data",
        json=DeleteExperimentDataRequest(assignments=True, snapshots=True).model_dump(),
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )

    # Verify both are deleted
    xngin_session.expire_all()
    assignments_after = list(
        await xngin_session.scalars(
            select(tables.ArmAssignment).where(tables.ArmAssignment.experiment_id == experiment_id)
        )
    )
    snapshots_after = list(
        await xngin_session.scalars(select(tables.Snapshot).where(tables.Snapshot.experiment_id == experiment_id))
    )
    assert len(assignments_after) == 0
    assert len(snapshots_after) == 0


async def test_delete_experiment_data_none_specified(
    xngin_session: AsyncSession,
    testing_experiment: tables.Experiment,
    testing_datasource,
    aclient: AdminAPIClient,
):
    """Test that specifying no data types deletes nothing."""
    ds_id = testing_datasource.ds.id
    experiment_id = testing_experiment.id

    # Create a snapshot to verify that it is not deleted
    snapshot = tables.Snapshot(experiment_id=experiment_id)
    xngin_session.add(snapshot)
    await xngin_session.commit()

    # Count assignments before
    assignments_before = list(
        await xngin_session.scalars(
            select(tables.ArmAssignment).where(tables.ArmAssignment.experiment_id == experiment_id)
        )
    )
    count_before = len(assignments_before)
    assert count_before > 0

    # Delete with no flags set
    aclient.delete_experiment_data(datasource_id=ds_id, experiment_id=experiment_id, body=DeleteExperimentDataRequest())

    # Verify nothing was deleted
    xngin_session.expire_all()
    assignments_after = list(
        await xngin_session.scalars(
            select(tables.ArmAssignment).where(tables.ArmAssignment.experiment_id == experiment_id)
        )
    )
    assert len(assignments_after) == count_before
    snapshots_after = list(
        await xngin_session.scalars(select(tables.Snapshot).where(tables.Snapshot.experiment_id == experiment_id))
    )
    assert len(snapshots_after) == 1


async def test_list_participant_types_excludes_hidden(
    xngin_session: AsyncSession, testing_datasource, aclient: AdminAPIClient
):
    """Test that list_participant_types excludes hidden participant types."""
    ds_id = testing_datasource.ds.id
    ds = testing_datasource.ds

    # Add a hidden participant type directly
    config = ds.get_config()
    config.participants.append(
        ParticipantsDef(
            type="schema",
            participant_type="hidden_pt",
            table_name="some_table",
            fields=[FieldDescriptor(field_name="id", data_type=DataType.BIGINT, is_unique_id=True)],
            hidden=True,
        )
    )
    ds.set_config(config)
    await xngin_session.commit()

    # List participants - should not include hidden one
    list_response = aclient.list_participant_types(datasource_id=ds_id).data
    participant_names = [p.participant_type for p in list_response.items]
    assert "hidden_pt" not in participant_names
    assert "test_participant_type" in participant_names


async def test_create_experiment_with_table_name_and_primary_key(
    xngin_session: AsyncSession, testing_datasource, aclient: AdminAPIClient
):
    """Test creating an experiment with table_name and primary_key instead of participant_type."""
    ds_id = testing_datasource.ds.id

    request_json = make_createexperimentrequest_json(experiment_type=ExperimentsType.FREQ_ONLINE)
    initial_participant_count = len(testing_datasource.ds.get_config().participants)
    experiment_request = CreateExperimentRequest.model_validate(request_json)

    created = aclient.create_experiment(
        datasource_id=ds_id,
        body=experiment_request,
        random_state=42,
    ).data

    assert created.participant_type_deprecated == ""

    # Verify no participant type was persisted to datasource config.
    ds = await xngin_session.get_one(tables.Datasource, ds_id)
    await xngin_session.refresh(ds)
    config = ds.get_config()
    assert len(config.participants) == initial_participant_count

    # Verify datasource_table is set to the requested table name
    experiment = await xngin_session.get_one(tables.Experiment, created.experiment_id)
    assert experiment.datasource_table == "dwh"


def test_create_experiment_freq_design_spec_requires_primary_key(testing_datasource, aclient: AdminAPIClient):
    ds_id = testing_datasource.ds.id
    request_json = make_createexperimentrequest_json(experiment_type=ExperimentsType.FREQ_ONLINE)
    del request_json["design_spec"]["primary_key"]
    with expect_status_code(422, text="primary_key", detail_eq="Field required"):
        aclient.create_experiment(datasource_id=ds_id, body=request_json)


def test_create_experiment_freq_design_spec_requires_table_name(testing_datasource, aclient: AdminAPIClient):
    ds_id = testing_datasource.ds.id
    request_json = make_createexperimentrequest_json(experiment_type=ExperimentsType.FREQ_ONLINE)
    del request_json["design_spec"]["table_name"]
    with expect_status_code(422, text="table_name", detail_eq="Field required"):
        aclient.create_experiment(datasource_id=ds_id, body=request_json)


async def test_create_preassigned_experiment_with_table_name_and_primary_key(
    xngin_session: AsyncSession, testing_datasource, aclient: AdminAPIClient
):
    ds_id = testing_datasource.ds.id
    request_json = make_createexperimentrequest_json(experiment_type=ExperimentsType.FREQ_PREASSIGNED)
    experiment_request = CreateExperimentRequest.model_validate(request_json)

    created = aclient.create_experiment(
        datasource_id=ds_id,
        body=experiment_request,
        desired_n=100,
        random_state=42,
    ).data

    assert created.participant_type_deprecated == ""
    assert created.assign_summary is not None
    assert created.assign_summary.sample_size == 100

    # Verify datasource_table is set to the requested table name
    experiment = await xngin_session.get_one(tables.Experiment, created.experiment_id)
    assert experiment.datasource_table == "dwh"


def test_list_snapshots_pagination(aclient: AdminAPIClient):
    """Test cursor-based pagination of the list_snapshots endpoint."""
    org = aclient.create_organizations(body=CreateOrganizationRequest(name="test_snapshots_pagination")).data

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
    ds = aclient.create_datasource(
        body=CreateDatasourceRequest(
            name="test_snapshots_pagination_ds",
            organization_id=org.id,
            dsn=valid_dsn,
        )
    ).data

    aclient.create_participant_type(
        datasource_id=ds.id,
        body=CreateParticipantsTypeRequest(
            participant_type="test_participant_type",
            schema_def=TESTING_DWH_PARTICIPANT_DEF,
        ),
    )

    experiment_id = aclient.create_experiment(
        datasource_id=ds.id,
        body=CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_type=ExperimentsType.FREQ_PREASSIGNED,
                table_name="dwh",
                primary_key="id",
                experiment_name="pagination test",
                description="pagination test",
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime.now(UTC) + timedelta(days=1),
                arms=[
                    Arm(arm_name="control", arm_description="Control"),
                    Arm(arm_name="treatment", arm_description="Treatment"),
                ],
                metrics=[DesignSpecMetricRequest(field_name="income", metric_pct_change=5)],
                strata=[],
                filters=[],
            ),
        ),
        desired_n=100,
    ).data.experiment_id

    aclient.commit_experiment(datasource_id=ds.id, experiment_id=experiment_id)

    # Create two snapshots
    aclient.create_snapshot(organization_id=org.id, datasource_id=ds.id, experiment_id=experiment_id)

    # Force second snapshot to fail
    aclient.update_datasource(
        datasource_id=ds.id, body=UpdateDatasourceRequest(dsn=valid_dsn.model_copy(update={"port": valid_dsn.port + 1}))
    )
    aclient.create_snapshot(organization_id=org.id, datasource_id=ds.id, experiment_id=experiment_id)

    # Page through with page_size=1
    page1 = aclient.list_snapshots(
        organization_id=org.id, datasource_id=ds.id, experiment_id=experiment_id, page_size=1
    ).data
    assert len(page1.items) == 1
    assert page1.next_page_token != ""
    assert page1.latest_failure is not None

    page2 = aclient.list_snapshots(
        organization_id=org.id,
        datasource_id=ds.id,
        experiment_id=experiment_id,
        page_size=1,
        page_token=page1.next_page_token,
    ).data
    assert len(page2.items) == 1
    assert page2.next_page_token == ""
    assert page2.latest_failure is not None

    # No overlap between pages
    assert page1.items[0].id != page2.items[0].id
    # Descending order maintained
    assert page1.items[0].updated_at >= page2.items[0].updated_at

    # Without pagination params, all results returned on one page
    all_items = aclient.list_snapshots(organization_id=org.id, datasource_id=ds.id, experiment_id=experiment_id).data
    assert len(all_items.items) == 2
    assert all_items.next_page_token == ""

    # Skip works from the beginning of the result set.
    skipped_page = aclient.list_snapshots(
        organization_id=org.id, datasource_id=ds.id, experiment_id=experiment_id, page_size=1, skip=1
    ).data
    assert len(skipped_page.items) == 1
    assert skipped_page.items[0].id == all_items.items[1].id

    # Invalid page_token returns 400
    with expect_status_code(400, text="Invalid page_token."):
        aclient.list_snapshots(
            organization_id=org.id, datasource_id=ds.id, experiment_id=experiment_id, page_token="bogus"
        )


def test_list_organization_events_pagination(testing_datasource, aclient: AdminAPIClient):
    """Test cursor-based pagination of the list_organization_events endpoint."""
    org_id = testing_datasource.org.id
    ds_id = testing_datasource.ds.id

    # Create a webhook
    webhook_id = aclient.add_webhook_to_organization(
        organization_id=org_id,
        body=AddWebhookToOrganizationRequest(
            type="experiment.created",
            url="https://example.com/webhook",
            name="test webhook",
        ),
    ).data.id

    webhooks = aclient.list_organization_webhooks(organization_id=org_id).data.items
    assert len(webhooks) == 1

    # Creating experiments generates events. Create a few to ensure we have enough.
    for i in range(3):
        experiment = aclient.create_experiment(
            datasource_id=ds_id,
            body=CreateExperimentRequest(
                webhooks=[webhook_id],
                design_spec=PreassignedFrequentistExperimentSpec(
                    experiment_type=ExperimentsType.FREQ_PREASSIGNED,
                    table_name="dwh",
                    primary_key="id",
                    experiment_name=f"pagination event test {i}",
                    description=f"pagination event test {i}",
                    start_date=datetime(2024, 1, 1, tzinfo=UTC),
                    end_date=datetime.now(UTC) + timedelta(days=1),
                    arms=[
                        Arm(arm_name="control", arm_description="Control"),
                        Arm(arm_name="treatment", arm_description="Treatment"),
                    ],
                    metrics=[DesignSpecMetricRequest(field_name="income", metric_pct_change=5)],
                    strata=[],
                    filters=[],
                ),
            ),
            desired_n=100,
        ).data
        aclient.commit_experiment(datasource_id=ds_id, experiment_id=experiment.experiment_id)

    # Get all events first to know how many we have
    all_events = aclient.list_organization_events(organization_id=org_id).data
    total = len(all_events.items)
    assert total >= 3

    # Page through with page_size=1
    page1 = aclient.list_organization_events(organization_id=org_id, page_size=1).data
    assert len(page1.items) == 1
    assert page1.next_page_token != ""

    page2 = aclient.list_organization_events(organization_id=org_id, page_size=1, page_token=page1.next_page_token).data
    assert len(page2.items) == 1

    # No overlap
    assert page1.items[0].id != page2.items[0].id
    # Descending order
    assert page1.items[0].created_at >= page2.items[0].created_at

    # Skip works from the beginning of the result set.
    first_page = aclient.list_organization_events(organization_id=org_id, page_size=3).data
    assert len(first_page.items) >= 3
    skipped_page = aclient.list_organization_events(organization_id=org_id, page_size=2, skip=1).data
    assert [item.id for item in skipped_page.items] == [item.id for item in first_page.items[1:3]]

    # Collect all pages
    all_ids: set[str] = set()
    token: str | None = None
    pages = 0
    while True:
        page = aclient.list_organization_events(organization_id=org_id, page_size=2, page_token=token).data
        for item in page.items:
            assert item.id not in all_ids, f"Duplicate event {item.id}"
            all_ids.add(item.id)
        pages += 1
        if not page.next_page_token:
            break
        token = page.next_page_token
    assert len(all_ids) == total
    assert pages > 1

    # Invalid page_token returns 400
    with expect_status_code(400, text="Invalid page_token."):
        aclient.list_organization_events(organization_id=org_id, page_token="bogus")


async def test_list_organization_events_pagination_with_same_timestamp_is_id_desc(
    testing_datasource, xngin_session: AsyncSession, aclient: AdminAPIClient
):
    """Events with identical created_at must paginate deterministically by id (descending)."""
    org_id = testing_datasource.org.id
    ds_id = testing_datasource.ds.id

    # Create events by creating and committing experiments.
    for i in range(2):
        experiment = aclient.create_experiment(
            datasource_id=ds_id,
            body=CreateExperimentRequest(
                design_spec=PreassignedFrequentistExperimentSpec(
                    experiment_type=ExperimentsType.FREQ_PREASSIGNED,
                    table_name="dwh",
                    primary_key="id",
                    experiment_name=f"same-ts event test {i}",
                    description=f"same-ts event test {i}",
                    start_date=datetime(2024, 1, 1, tzinfo=UTC),
                    end_date=datetime.now(UTC) + timedelta(days=1),
                    arms=[
                        Arm(arm_name="control", arm_description="Control"),
                        Arm(arm_name="treatment", arm_description="Treatment"),
                    ],
                    metrics=[DesignSpecMetricRequest(field_name="income", metric_pct_change=5)],
                    strata=[],
                    filters=[],
                ),
            ),
            desired_n=100,
        ).data
        aclient.commit_experiment(datasource_id=ds_id, experiment_id=experiment.experiment_id)

    all_events = aclient.list_organization_events(organization_id=org_id).data.items
    assert len(all_events) >= 2

    tied_event_ids = [all_events[0].id, all_events[1].id]
    tied_created_at = datetime(2099, 1, 1, tzinfo=UTC)
    await xngin_session.execute(
        update(tables.Event).where(tables.Event.id.in_(tied_event_ids)).values(created_at=tied_created_at)
    )
    await xngin_session.commit()

    expected_tied_order = list(
        await xngin_session.scalars(
            select(tables.Event.id).where(tables.Event.id.in_(tied_event_ids)).order_by(tables.Event.id.desc())
        )
    )
    assert len(expected_tied_order) == 2

    page1 = aclient.list_organization_events(organization_id=org_id, page_size=1).data
    assert len(page1.items) == 1
    assert page1.next_page_token != ""
    assert page1.items[0].id == expected_tied_order[0]
    assert page1.items[0].created_at == tied_created_at

    page2 = aclient.list_organization_events(organization_id=org_id, page_size=1, page_token=page1.next_page_token).data
    assert len(page2.items) == 1
    assert page2.items[0].id == expected_tied_order[1]
    assert page2.items[0].id != page1.items[0].id
    assert page2.items[0].created_at == tied_created_at
