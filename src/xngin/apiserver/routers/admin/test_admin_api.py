import copy
import csv
import io
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from typing import Protocol
from urllib.parse import urlparse

import numpy as np
import pytest
from deepdiff import DeepDiff
from pydantic import HttpUrl

from xngin.apiserver import flags
from xngin.apiserver.conftest import convert_dwh_to_create_api_dsn, expect_status_code
from xngin.apiserver.dns import safe_resolve
from xngin.apiserver.dwh.inspection_types import FieldDescriptor, ParticipantsSchema
from xngin.apiserver.dwh.inspections import ColumnDeleted, Drift, FieldChangedType
from xngin.apiserver.routers.admin.admin_api_converters import CREDENTIALS_UNAVAILABLE_MESSAGE
from xngin.apiserver.routers.admin.admin_api_types import (
    AddMemberToOrganizationRequest,
    AddWebhookToOrganizationRequest,
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
    ContextInput,
    CreateExperimentRequest,
    CreateExperimentResponse,
    DataType,
    DesignSpecMetricRequest,
    ExperimentConfig,
    ExperimentsType,
    Filter,
    FreqExperimentAnalysisResponse,
    GetExperimentResponse,
    GetParticipantAssignmentResponse,
    LikelihoodTypes,
    MABExperimentSpec,
    OnlineFrequentistExperimentSpec,
    PowerRequest,
    PreassignedFrequentistExperimentSpec,
    PriorTypes,
    Stratum,
    UpdateBanditArmOutcomeRequest,
)
from xngin.apiserver.routers.common_enums import (
    ExperimentState,
    MetricPowerAnalysisMessageType,
    Relation,
    StopAssignmentReason,
)
from xngin.apiserver.routers.experiments.test_experiments_common import (
    make_create_freq_online_experiment_request,
    make_create_online_bandit_experiment_request,
    make_create_preassigned_experiment_request,
    make_createexperimentrequest_json,
)
from xngin.apiserver.settings import Dsn
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.experiments_api_client import ExperimentsAPIClient
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_PARTICIPANT_DEF
from xngin.apiserver.testing.wide_dwh_def import WIDE_DWH_PARTICIPANT_DEF

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


@dataclass(slots=True)
class TestExperiment:
    config: ExperimentConfig
    organization_id: str
    api_key: str

    @property
    def id(self) -> str:
        return self.config.experiment_id

    @property
    def datasource_id(self) -> str:
        return self.config.datasource_id

    @property
    def arms(self) -> Sequence[Arm]:
        return self.config.design_spec.arms

    @property
    def start_date(self) -> datetime:
        return self.config.design_spec.start_date

    @property
    def end_date(self) -> datetime:
        return self.config.design_spec.end_date


def arm_ids(arms: Sequence[Arm]) -> set[str]:
    ids = {arm.arm_id for arm in arms}
    assert None not in ids
    return {arm_id for arm_id in ids if arm_id is not None}


def find_ds_with_name[DSType: HasName](datasources: list[DSType], name: str) -> DSType:
    """Helper function to find a datasource with a specific name from an iterable.

    Raises StopIteration if the datasource is not found.
    """
    return next(ds for ds in datasources if ds.name == name)


def create_org_with_default_datasource(aclient: AdminAPIClient, org_name: str) -> tuple[str, str]:
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name=org_name)).data.id
    datasource_id = aclient.create_datasource(
        body=CreateDatasourceRequest(
            organization_id=org_id,
            name=org_name,
            dsn=convert_dwh_to_create_api_dsn(Dsn.from_url(flags.XNGIN_DEVDWH_DSN)),
        )
    ).data.id
    return org_id, datasource_id


async def make_freq_online_experiment(
    aclient: AdminAPIClient, datasource_id: str, end_date: datetime | None = None
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


def normalize_bandit_analysis(response: BanditExperimentAnalysisResponse) -> BanditExperimentAnalysisResponse:
    return response.model_copy(update={"created_at": datetime(2000, 1, 1, tzinfo=UTC)})


async def make_bandit_online_experiment(
    aclient: AdminAPIClient,
    datasource_id: str,
    *,
    experiment_type: ExperimentsType = ExperimentsType.MAB_ONLINE,
    prior_type: PriorTypes = PriorTypes.BETA,
    reward_type: LikelihoodTypes = LikelihoodTypes.BERNOULLI,
) -> GetExperimentForUiResponse:
    request_obj = make_create_online_bandit_experiment_request(
        experiment_type=experiment_type,
        reward_type=reward_type,
        prior_type=prior_type,
    )
    experiment_id = aclient.create_experiment(datasource_id=datasource_id, body=request_obj, random_state=42).data
    aclient.commit_experiment(datasource_id=datasource_id, experiment_id=experiment_id.experiment_id)
    data = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=experiment_id.experiment_id).data
    assert isinstance(data, GetExperimentForUiResponse)
    return data


async def test_get_experiment_for_ui_includes_sample_calls_for_mab(testing_datasource, aclient: AdminAPIClient):
    """A MAB experiment's UI response carries example calls with a type-correct outcome."""
    experiment = await make_bandit_online_experiment(
        aclient,
        testing_datasource.datasource_id,
        prior_type=PriorTypes.NORMAL,
        reward_type=LikelihoodTypes.NORMAL,
    )
    assert experiment.sample_calls is not None
    assert [c.label for c in experiment.sample_calls.calls] == ["Get assignment", "Report outcome"]
    outcome_call = experiment.sample_calls.calls[1]
    assert outcome_call.method == "POST"
    assert outcome_call.path.endswith("/outcome")
    assert experiment.config.experiment_id in outcome_call.path
    assert outcome_call.body == {"outcome": 1.5}  # NORMAL reward => real-valued example


async def test_get_experiment_for_ui_sample_calls_none_for_frequentist(
    testing_datasource, aclient: AdminAPIClient, testing_experiment: TestExperiment
):
    """Frequentist experiments don't push outcomes to us, so they get no sample calls."""
    response = aclient.get_experiment_for_ui(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=testing_experiment.config.experiment_id,  # NB this is a frequentist experiment fixture
    ).data
    assert response.sample_calls is None


def make_cmab_context_inputs(
    experiment: GetExperimentForUiResponse,
    values: list[float],
) -> list[ContextInput]:
    design_spec = experiment.config.design_spec
    assert isinstance(design_spec, CMABExperimentSpec)
    assert design_spec.contexts is not None
    sorted_contexts = sorted(design_spec.contexts, key=lambda c: c.context_id or "")
    assert len(sorted_contexts) == len(values)
    return [
        ContextInput(context_id=context.context_id or "", context_value=value)
        for context, value in zip(sorted_contexts, values, strict=True)
    ]


@pytest.fixture(name="testing_experiment")
async def fixture_testing_experiment(testing_datasource, aclient: AdminAPIClient) -> TestExperiment:
    """Create a committed preassigned experiment through the Admin API."""
    datasource_id = testing_datasource.datasource_id
    created = aclient.create_experiment(
        datasource_id=datasource_id,
        body=CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
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
                desired_n=10,
            ),
        ),
        random_state=42,
    ).data
    aclient.commit_experiment(datasource_id=datasource_id, experiment_id=created.experiment_id)
    config = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=created.experiment_id).data.config
    return TestExperiment(
        config=config,
        organization_id=testing_datasource.organization_id,
        api_key=testing_datasource.key,
    )


@pytest.fixture(name="testing_bandit_experiment")
async def fixture_testing_bandit_experiment(
    request,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
) -> TestExperiment:
    """Create a committed bandit experiment and optional outcomes through public APIs."""
    experiment_type, prior_type, reward_type, num_participants = request.param
    experiment = await make_bandit_online_experiment(
        aclient,
        testing_datasource.datasource_id,
        experiment_type=experiment_type,
        prior_type=prior_type,
        reward_type=reward_type,
    )

    for i in range(num_participants):
        participant_id = str(i)
        if experiment_type == ExperimentsType.CMAB_ONLINE:
            design_spec = experiment.config.design_spec
            assert isinstance(design_spec, CMABExperimentSpec)
            assert design_spec.contexts is not None
            eclient.get_assignment_cmab(
                api_key=testing_datasource.key,
                experiment_id=experiment.config.experiment_id,
                participant_id=participant_id,
                body=CMABContextInputRequest(
                    context_inputs=make_cmab_context_inputs(experiment, [1.0] * len(design_spec.contexts))
                ),
                random_state=i,
            )
        else:
            eclient.get_assignment(
                api_key=testing_datasource.key,
                experiment_id=experiment.config.experiment_id,
                participant_id=participant_id,
            )
        eclient.update_bandit_arm_with_participant_outcome(
            api_key=testing_datasource.key,
            experiment_id=experiment.config.experiment_id,
            participant_id=participant_id,
            body=UpdateBanditArmOutcomeRequest(outcome=float(i % 2)),
        )

    config = aclient.get_experiment_for_ui(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=experiment.config.experiment_id,
    ).data.config
    return TestExperiment(
        config=config,
        organization_id=testing_datasource.organization_id,
        api_key=testing_datasource.key,
    )


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

    # Inspect the default NoDwh datasource; should expose no table names.
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
def test_create_datasource_without_credentials(dsn: BqDsn | RedshiftDsn | PostgresDsn, aclient: AdminAPIClient):
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
        organization_id=testing_datasource.organization_id, body=AddMemberToOrganizationRequest(email=PRIVILEGED_EMAIL)
    )

    with expect_status_code(400, text="DNS resolution failed"):
        aclient.create_datasource(
            body=CreateDatasourceRequest(
                organization_id=testing_datasource.organization_id,
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


def test_create_datasource_with_connectivity_check_connection_failure(aclient: AdminAPIClient):
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


def test_create_datasource_with_connectivity_check_disabled_by_default(aclient: AdminAPIClient):
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


def test_create_datasource_with_connectivity_check_can_be_enabled(aclient: AdminAPIClient):
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
        organization_id=testing_datasource.organization_id, body=AddMemberToOrganizationRequest(email=PRIVILEGED_EMAIL)
    )

    # Add unprivileged user to existing organization
    aclient.add_member_to_organization(
        organization_id=testing_datasource.organization_id,
        body=AddMemberToOrganizationRequest(email=UNPRIVILEGED_EMAIL),
    )

    # Adding a user to an existing organization (again)
    aclient.add_member_to_organization(
        organization_id=testing_datasource.organization_id,
        body=AddMemberToOrganizationRequest(email=UNPRIVILEGED_EMAIL),
    )

    # Adding a new user to an existing organization
    aclient.add_member_to_organization(
        organization_id=testing_datasource.organization_id,
        body=AddMemberToOrganizationRequest(email="newuser@example.com"),
    )

    # Confirm all users added.
    org_response = aclient.get_organization(organization_id=testing_datasource.organization_id)
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

    # 403 when trying to remove self from organization
    remove_member(member_list.get(PRIVILEGED_EMAIL), expected_status_code=403)
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
    assert response.items[0].id == testing_datasource.organization_id
    assert response.items[0].name == "test organization"


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


def test_list_datasources_ordered_by_experiment_count(testing_datasource, aclient: AdminAPIClient):
    """Datasources should be ordered by number of experiments (desc), then by name (asc)."""
    org_id = testing_datasource.organization_id
    ds_a_id = testing_datasource.datasource_id
    dsn = convert_dwh_to_create_api_dsn(Dsn.from_url(flags.XNGIN_DEVDWH_DSN))
    ds_b_id = aclient.create_datasource(
        body=CreateDatasourceRequest(organization_id=org_id, name="AAA datasource", dsn=dsn)
    ).data.id
    ds_c_id = aclient.create_datasource(
        body=CreateDatasourceRequest(organization_id=org_id, name="ZZZ datasource", dsn=dsn)
    ).data.id

    # Add experiments: ds_b gets 3, ds_a gets 1, ds_c gets 0.
    for ds_id, count in [(ds_b_id, 3), (ds_a_id, 1)]:
        for _ in range(count):
            aclient.create_experiment(
                datasource_id=ds_id,
                body=make_create_freq_online_experiment_request(),
                random_state=42,
            )

    items = aclient.list_organization_datasources(organization_id=org_id).data.items
    assert len(items) == 4

    # ds_b (3 experiments) first, ds_a (1 experiment) second.
    # The zero-experiment datasources are then ordered by name: the default NoDWH datasource
    # before ds_c.
    assert items[0].id == ds_b_id
    assert items[1].id == ds_a_id
    assert items[2].name == DEFAULT_NO_DWH_SOURCE_NAME
    assert items[3].id == ds_c_id


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
    ds_id = testing_datasource.datasource_id
    org_id = testing_datasource.organization_id

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


def test_delete_datasource_scopes_resource_to_organization(
    testing_datasource,
    aclient: AdminAPIClient,
    aclient_unpriv: AdminAPIClient,
):
    caller_org_id, caller_datasource_id = create_org_with_default_datasource(
        aclient_unpriv, "datasource-delete-caller-org"
    )

    with expect_status_code(404):
        aclient_unpriv.delete_datasource(
            organization_id=caller_org_id,
            datasource_id=testing_datasource.datasource_id,
        )

    response = aclient.get_datasource(datasource_id=testing_datasource.datasource_id).data
    assert response.id == testing_datasource.datasource_id

    # aclient is privileged but still needs to be a member of caller_org_id
    with expect_status_code(403):
        aclient.delete_datasource(
            organization_id=caller_org_id,
            datasource_id=caller_datasource_id,
        )

    # aclient can access datasource_id but not via caller_org_id
    with expect_status_code(403):
        aclient.delete_datasource(
            organization_id=caller_org_id,
            datasource_id=testing_datasource.datasource_id,
        )


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
    new_url = "https://example.com/updated_webhook"
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
            body=UpdateOrganizationWebhookRequest.model_construct(url="https://example.com/should-fail", name="fail"),
        )

    # Try to delete a non-existent webhook
    with expect_status_code(404):
        aclient.delete_webhook_from_organization(organization_id=org_id, webhook_id=webhook_id)


def test_delete_webhook_scopes_resource_to_organization(aclient: AdminAPIClient, aclient_unpriv: AdminAPIClient):
    victim_org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="webhook-delete-victim")).data.id
    webhook = aclient.add_webhook_to_organization(
        organization_id=victim_org_id,
        body=AddWebhookToOrganizationRequest(
            type="experiment.created",
            url="https://example.com/webhook",
            name="victim webhook",
        ),
    ).data
    caller_org_id, _caller_datasource_id = create_org_with_default_datasource(
        aclient_unpriv, "webhook-delete-caller-org"
    )

    with expect_status_code(404):
        aclient_unpriv.delete_webhook_from_organization(
            organization_id=caller_org_id,
            webhook_id=webhook.id,
        )

    webhook_ids = {item.id for item in aclient.list_organization_webhooks(organization_id=victim_org_id).data.items}
    assert webhook.id in webhook_ids


def test_create_webhook_rejects_ssrf_url(aclient: AdminAPIClient):
    """Creating a webhook with an internal/unsafe hostname is rejected."""
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="add_webhook_rejects_ssrf")).data.id
    with expect_status_code(422, detail_contains="Failed to resolve hostname from webhook URL"):
        aclient.add_webhook_to_organization(
            organization_id=org_id,
            body=AddWebhookToOrganizationRequest.model_construct(
                type="experiment.created",
                url=f"http://{safe_resolve.UNSAFE_IP_FOR_TESTING}/steal-creds",
                name="ssrf attempt",
            ),
        )


def test_update_webhook_rejects_ssrf_url(aclient: AdminAPIClient):
    """Updating a webhook to point at an internal/unsafe hostname is rejected."""
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="update_webhook_rejects_ssrf")).data.id
    webhook_id = aclient.add_webhook_to_organization(
        organization_id=org_id,
        body=AddWebhookToOrganizationRequest(
            type="experiment.created",
            url="http://8.8.8.8/webhook",
            name="safe webhook",
        ),
    ).data.id
    with expect_status_code(422, detail_contains="Failed to resolve hostname from webhook URL"):
        aclient.update_organization_webhook(
            organization_id=org_id,
            webhook_id=webhook_id,
            body=UpdateOrganizationWebhookRequest.model_construct(
                url=f"http://{safe_resolve.UNSAFE_IP_FOR_TESTING}/steal-creds",
                name="ssrf attempt",
            ),
        )


def test_participants_lifecycle(testing_datasource, aclient: AdminAPIClient):
    """Test getting, creating, listing, updating, and deleting a participant type."""
    ds_id = testing_datasource.datasource_id

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
    ds_id = testing_datasource.datasource_id
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
    ds_id = testing_datasource.datasource_id
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
        organization_id=testing_datasource.organization_id, body=AddMemberToOrganizationRequest(email=PRIVILEGED_EMAIL)
    )

    # Inspect the datasource.
    datasource_inspection = aclient.inspect_datasource(datasource_id=testing_datasource.datasource_id).data
    assert "dwh" in datasource_inspection.tables, datasource_inspection

    # Inspect one table in the datasource.
    table_inspection = aclient.inspect_table_in_datasource(
        datasource_id=testing_datasource.datasource_id, table_name="dwh"
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
        datasource_id=testing_datasource.datasource_id,
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
    create_exp_dict = make_createexperimentrequest_json(desired_n=100)
    create_exp_request = CreateExperimentRequest.model_validate(create_exp_dict)
    create_exp_request.design_spec.design_url = HttpUrl("https://example.com/design")
    assert isinstance(create_exp_request.design_spec, PreassignedFrequentistExperimentSpec)
    create_exp_request.design_spec.filters = [
        Filter(field_name="id", relation=Relation.EXCLUDES, value=[str((2 << 52) + 1), None]),
        Filter(field_name="is_engaged", relation=Relation.INCLUDES, value=[True, False]),
    ]
    created_experiment = aclient.create_experiment(
        datasource_id=testing_datasource.datasource_id, body=create_exp_request
    ).data
    parsed_experiment_id = created_experiment.experiment_id
    assert parsed_experiment_id is not None
    assert created_experiment.design_spec.design_url == HttpUrl("https://example.com/design")
    assert created_experiment.stopped_assignments_at is not None
    assert created_experiment.stopped_assignments_reason == StopAssignmentReason.PREASSIGNED
    parsed_arm_ids = [arm.arm_id for arm in created_experiment.design_spec.arms]
    assert len(parsed_arm_ids) == 2

    # Commit the new experiment.
    aclient.commit_experiment(datasource_id=testing_datasource.datasource_id, experiment_id=parsed_experiment_id)

    # Verify it committed.
    assert (
        aclient.get_experiment_for_ui(
            datasource_id=testing_datasource.datasource_id, experiment_id=parsed_experiment_id
        ).data.config.state
        == ExperimentState.COMMITTED
    )

    # Attempting to abandon a committed experiment should fail
    with expect_status_code(409):
        aclient.abandon_experiment(datasource_id=testing_datasource.datasource_id, experiment_id=parsed_experiment_id)

    # Verify it is still committed.
    get_exp = aclient.get_experiment_for_ui(
        datasource_id=testing_datasource.datasource_id, experiment_id=parsed_experiment_id
    ).data
    assert get_exp.config.state == ExperimentState.COMMITTED
    assert get_exp.experiment_schema is not None

    # Update the experiment.
    aclient.update_experiment(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=parsed_experiment_id,
        body=UpdateExperimentRequest(name="updated"),
    )

    # Update an arm.
    updated_arm_id = parsed_arm_ids[0]
    assert updated_arm_id is not None
    aclient.update_arm(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=parsed_experiment_id,
        arm_id=updated_arm_id,
        body=UpdateArmRequest(name="updated arm"),
    )

    # Get that experiment.
    get_experiment_response = aclient.get_experiment_for_ui(
        datasource_id=testing_datasource.datasource_id, experiment_id=parsed_experiment_id
    ).data
    assert get_experiment_response.config.experiment_id == parsed_experiment_id
    assert get_experiment_response.config.design_spec.experiment_name == "updated"
    arm = next((arm for arm in get_experiment_response.config.design_spec.arms if arm.arm_id == updated_arm_id), None)
    assert arm is not None
    assert arm.arm_name == "updated arm"

    # List org experiments.
    experiment_list = aclient.list_organization_experiments(organization_id=testing_datasource.organization_id).data
    assert len(experiment_list.items) == 1, experiment_list
    experiment_config_0 = experiment_list.items[0]
    assert experiment_config_0.experiment_id == parsed_experiment_id

    # Analyze experiment
    experiment_analysis = aclient.analyze_experiment(
        datasource_id=testing_datasource.datasource_id, experiment_id=parsed_experiment_id
    ).data
    assert experiment_analysis.experiment_id == parsed_experiment_id

    # Get assignments for the experiment as CSV.
    # Use aclient.client directly because generated client does not correctly handle CSV response types.
    assignments_csv_response = aclient.client.get(
        f"/v1/m/datasources/{testing_datasource.datasource_id}/experiments/{parsed_experiment_id}/assignments/csv"
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
        aclient_unpriv.delete_experiment(
            datasource_id=testing_datasource.datasource_id, experiment_id=parsed_experiment_id
        )

    # Delete the experiment.
    aclient.delete_experiment(datasource_id=testing_datasource.datasource_id, experiment_id=parsed_experiment_id)

    # Delete the experiment again.
    with expect_status_code(404):
        aclient.delete_experiment(datasource_id=testing_datasource.datasource_id, experiment_id=parsed_experiment_id)

    # Delete the experiment again w/allow_missing.
    aclient.delete_experiment(
        datasource_id=testing_datasource.datasource_id, experiment_id=parsed_experiment_id, allow_missing=True
    )


async def test_delete_experiment_scopes_resource_to_datasource(
    testing_datasource,
    aclient: AdminAPIClient,
    aclient_unpriv: AdminAPIClient,
):
    victim_experiment = await make_freq_online_experiment(aclient, testing_datasource.datasource_id)
    victim_experiment_id = victim_experiment.config.experiment_id
    _caller_org_id, caller_datasource_id = create_org_with_default_datasource(
        aclient_unpriv, "experiment-delete-caller-org"
    )

    with expect_status_code(404):
        aclient_unpriv.delete_experiment(
            datasource_id=caller_datasource_id,
            experiment_id=victim_experiment_id,
        )

    response = aclient.get_experiment_for_ui(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=victim_experiment_id,
    ).data
    assert response.config.experiment_id == victim_experiment_id


async def test_abandon_experiment(testing_datasource, aclient: AdminAPIClient):
    datasource_id = testing_datasource.datasource_id
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
        desired_n=1,
    )
    parsed_response = aclient.create_experiment(
        datasource_id=datasource_id,
        body=CreateExperimentRequest(design_spec=design_spec),
    ).data
    assert parsed_response.state == ExperimentState.ASSIGNED
    parsed_experiment_id = parsed_response.experiment_id

    aclient.abandon_experiment(datasource_id=datasource_id, experiment_id=parsed_experiment_id)

    response = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=parsed_experiment_id)
    assert response.data.config.state == ExperimentState.ABANDONED


async def test_power_check_with_unbalanced_arms(testing_datasource, aclient: AdminAPIClient):
    """Test power check endpoint with balanced vs unbalanced arms."""
    ds_id = testing_datasource.datasource_id
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
    power_response = aclient.power_check(datasource_id=ds_id, body=PowerRequest(design_spec=design_spec)).data
    assert len(power_response.analyses) == 1
    metric_analysis = power_response.analyses[0]
    assert metric_analysis.metric_spec.field_name == "current_income"
    assert metric_analysis.target_n == 474
    assert metric_analysis.sufficient_n is True

    # Now check with unbalanced arms
    design_spec.arms[0].arm_weight = 20.0
    design_spec.arms[1].arm_weight = 80.0
    power_response2 = aclient.power_check(datasource_id=ds_id, body=PowerRequest(design_spec=design_spec)).data
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
    power_response3 = aclient.power_check(datasource_id=ds_id, body=PowerRequest(design_spec=design_spec)).data
    assert len(power_response3.analyses) == 1
    metric_analysis3 = power_response3.analyses[0]
    assert metric_analysis3.metric_spec.field_name == "current_income"
    assert metric_analysis3.target_n is not None
    # Min ratio is still 4:1 (the smallest treatment arm) as in the previous case, but the control
    # is now only 10% of the total instead of 20%, so we need more participants than before to
    # ensure that comparison with the smaller arm still has sufficient power.
    assert metric_analysis3.target_n == math.ceil(metric_analysis2.target_n * 0.2 / 0.10)


async def test_power_check_also_sets_pct_change_with_desired_n(testing_datasource, aclient: AdminAPIClient):
    """design_spec.desired_n populates pct_change_with_desired_n on power check analyses."""
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="test power check desired_n",
        description="design_spec.desired_n drives optional MDE field",
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
        desired_n=500,
    )

    power_response = aclient.power_check(
        datasource_id=testing_datasource.datasource_id,
        body=PowerRequest(design_spec=design_spec),
    ).data
    assert len(power_response.analyses) == 1
    analysis = power_response.analyses[0]
    assert analysis.pct_change_with_desired_n == pytest.approx(0.0973, rel=1e-3)
    assert analysis.target_n == 474  # min sample size
    assert analysis.msg is not None
    assert analysis.msg.type == MetricPowerAnalysisMessageType.SUFFICIENT
    assert "There are enough units available." in analysis.msg.msg

    # And when not set, we get only the default minimum sample size calculation.
    design_spec_plain = design_spec.model_copy(update={"desired_n": None})
    power_response_plain = aclient.power_check(
        datasource_id=testing_datasource.datasource_id,
        body=PowerRequest(design_spec=design_spec_plain),
    ).data
    analysis_plain = power_response_plain.analyses[0]
    assert analysis_plain.pct_change_with_desired_n is None
    assert analysis_plain.target_n == 474
    assert analysis_plain.msg is not None
    assert analysis_plain.msg.type == MetricPowerAnalysisMessageType.SUFFICIENT
    assert "There are enough units available." in analysis.msg.msg


async def test_power_check_when_sample_size_insufficient_and_desired_n_should_otherwise_pass(
    testing_datasource,
    aclient: AdminAPIClient,
):
    """
    When design_spec.desired_n set but there are insufficient units, MDE enrichment should still
    succeed by otherwise assuming there are enough units to meet the desired_n. Simulates an
    exploratory use of the power calculator functionality.
    """
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="test power check desired_n failure",
        description="design_spec.desired_n should surface MDE validation errors",
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
        # Constrain data to force insufficient sample size for the desired metric_pct_change, but
        # still allow for a valid MDE calculation given a desired_n.
        filters=[Filter(field_name="id", relation=Relation.BETWEEN, value=[1, 100])],
        desired_n=1600,  # 4x the min size should allow for an MDE that's 1/2 the original MDE.
    )

    power_response = aclient.power_check(
        datasource_id=testing_datasource.datasource_id,
        body=PowerRequest(design_spec=design_spec),
    ).data
    assert len(power_response.analyses) == 1
    analysis = power_response.analyses[0]
    assert analysis.target_n == 400
    assert analysis.msg is not None
    assert analysis.msg.type == MetricPowerAnalysisMessageType.INSUFFICIENT
    # There should be no MDE calculation result because the data validation error prevented it.
    assert analysis.pct_change_with_desired_n == pytest.approx(0.0498, rel=1e-3)


async def test_power_check_when_sample_size_insufficient_and_desired_n_has_data_validation_error(
    testing_datasource,
    aclient: AdminAPIClient,
):
    """
    When both the sample size and MDE calculation fail, we should still see the minimum sample size
    result (an error message) as the primary, with no MDE calculation result.
    """
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="test power check desired_n failure",
        description="design_spec.desired_n should surface MDE validation errors",
        table_name="dwh",
        primary_key="id",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC) + timedelta(days=1),
        arms=[
            Arm(arm_name="control", arm_description="Control group"),
            Arm(arm_name="treatment", arm_description="Treatment group"),
        ],
        metrics=[DesignSpecMetricRequest(field_name="is_onboarded_onetime", metric_pct_change=0.1)],
        strata=[],
        # Constrain data to 1 unit available with only a NULL to force insufficient sample size, and
        # trigger a metric baseline error (due to zero non-nulls) in the MDE calculation.
        filters=[Filter(field_name="id", relation=Relation.INCLUDES, value=["1"])],
        desired_n=500,
    )

    power_response = aclient.power_check(
        datasource_id=testing_datasource.datasource_id,
        body=PowerRequest(design_spec=design_spec),
    ).data
    assert len(power_response.analyses) == 1
    analysis = power_response.analyses[0]
    assert analysis.target_n is None
    assert analysis.msg is not None
    assert analysis.msg.type == MetricPowerAnalysisMessageType.INSUFFICIENT
    # There should be no MDE calculation result because the data validation error prevented it.
    assert analysis.pct_change_with_desired_n is None


def test_power_check_when_sample_size_sufficient_and_desired_n_fails(testing_datasource, aclient: AdminAPIClient):
    """Invalid desired_n raises StatsPowerError."""
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="test power check desired_n failure",
        description="design_spec.desired_n should surface MDE validation errors",
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
        desired_n=0,  # This should raise a ValueError as a precheck during the MDE calculation.
    )

    with expect_status_code(422, detail_contains="Chosen sample size must be positive"):
        aclient.power_check(datasource_id=testing_datasource.datasource_id, body=PowerRequest(design_spec=design_spec))


async def test_power_check_validations(testing_datasource, aclient: AdminAPIClient):
    """Test power check validations."""
    ds_id = testing_datasource.datasource_id
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
    power_response = aclient.power_check(datasource_id=ds_id, body=PowerRequest(design_spec=design_spec)).data
    assert len(power_response.analyses) == 1
    assert power_response.analyses[0].metric_spec.field_name == "current_income"
    assert power_response.analyses[0].target_n is not None

    # Now check various failure scenarios
    with expect_status_code(404, message_contains="The table '' does not exist."):
        bad_design_spec = design_spec.model_copy(deep=True)
        bad_design_spec.table_name = ""
        aclient.power_check(datasource_id=ds_id, body=PowerRequest(design_spec=bad_design_spec))

    with expect_status_code(422, detail_contains="columns that do not exist in the table: no_such_primary_key"):
        bad_design_spec = design_spec.model_copy(deep=True)
        bad_design_spec.primary_key = "no_such_primary_key"
        aclient.power_check(datasource_id=ds_id, body=PowerRequest(design_spec=bad_design_spec))

    with expect_status_code(
        422, detail_contains="columns that do not exist in the table: bad_filter, bad_metric, bad_stratum"
    ):
        bad_design_spec = design_spec.model_copy(deep=True)
        bad_design_spec.metrics = [DesignSpecMetricRequest(field_name="bad_metric", metric_pct_change=0.1)]
        bad_design_spec.strata = [Stratum(field_name="bad_stratum")]
        bad_design_spec.filters = [Filter(field_name="bad_filter", relation=Relation.INCLUDES, value=["value"])]
        aclient.power_check(datasource_id=ds_id, body=PowerRequest(design_spec=bad_design_spec))

    with expect_status_code(422, detail_contains="Invalid metric field(s): (gender). Only boolean or numeric"):
        bad_design_spec = design_spec.model_copy(deep=True)
        bad_design_spec.metrics = [DesignSpecMetricRequest(field_name="gender", metric_pct_change=0.1)]
        aclient.power_check(datasource_id=ds_id, body=PowerRequest(design_spec=bad_design_spec))


async def test_create_experiment_with_invalid_design_url(testing_datasource, aclient: AdminAPIClient):
    datasource_id = testing_datasource.datasource_id
    # Work with the raw json to construct a bad request
    request = make_createexperimentrequest_json(desired_n=1)
    request["design_spec"]["design_url"] = "example.com/"

    with expect_status_code(422, detail_contains="Input should be a valid URL, relative URL without a base"):
        aclient.create_experiment(datasource_id=datasource_id, body=request)

    # Now check that a too long URL is rejected.
    request["design_spec"]["design_url"] = "http://example.com/" + "a" * 500
    with expect_status_code(422, detail_contains="URL should have at most 500 characters"):
        aclient.create_experiment(datasource_id=datasource_id, body=request)

    # And we need a host.
    request["design_spec"]["design_url"] = "https://"
    with expect_status_code(422, detail_contains="Input should be a valid URL, empty host"):
        aclient.create_experiment(datasource_id=datasource_id, body=request)


async def test_create_experiment_with_primary_key_as_strata_fails(testing_datasource, aclient: AdminAPIClient):
    datasource_id = testing_datasource.datasource_id
    request = make_createexperimentrequest_json(desired_n=1)
    primary_key = request["design_spec"]["primary_key"]
    request["design_spec"]["strata"] = [Stratum(field_name=primary_key)]
    with expect_status_code(422, detail_contains=f"Primary key {primary_key} cannot be used in strata."):
        aclient.create_experiment(datasource_id=datasource_id, body=request)


async def test_create_and_get_freq_preassigned_experiment(
    testing_datasource,
    use_deterministic_random,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    datasource_id = testing_datasource.datasource_id
    request_obj = make_create_preassigned_experiment_request(desired_n=100)

    created_experiment = aclient.create_experiment(datasource_id=datasource_id, body=request_obj, random_state=42).data
    parsed_experiment_id = created_experiment.experiment_id
    assert parsed_experiment_id is not None
    parsed_arm_ids = {arm.arm_id for arm in created_experiment.design_spec.arms}
    assert len(parsed_arm_ids) == 2
    assert isinstance(created_experiment.design_spec, PreassignedFrequentistExperimentSpec)
    assert len(created_experiment.design_spec.strata) == 1, created_experiment.design_spec.strata
    assert created_experiment.design_spec.strata[0].field_name == "gender"

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
    assert admin_experiment.experiment_schema is not None
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
    # Verify that participant field metadata was stored correctly.
    experiment_fields = admin_experiment.experiment_schema.fields
    assert len(experiment_fields) == 3
    unique_id_field = next((f for f in experiment_fields if f.is_unique_id), None)
    assert unique_id_field is not None
    assert unique_id_field.data_type == "bigint"
    gender_field = next((f for f in experiment_fields if f.field_name == "gender"), None)
    assert gender_field is not None
    assert gender_field.is_strata
    assert gender_field.data_type == "character varying"
    is_onboarded_field = next((f for f in experiment_fields if f.field_name == "is_onboarded"), None)
    assert is_onboarded_field is not None
    assert is_onboarded_field.is_metric
    assert is_onboarded_field.is_strata is False
    assert is_onboarded_field.data_type == "boolean"

    # Verify assignments were created
    actual_assignments = eclient.get_experiment_assignments(
        api_key=testing_datasource.key, experiment_id=experiment_id
    ).data
    assert len(actual_assignments.assignments) == 100

    # Check one assignment to see if it looks roughly right
    sample_assignment = actual_assignments.assignments[0]
    assert sample_assignment.arm_id in {arm1_id, arm2_id}
    assert sample_assignment.strata is not None
    assert len(sample_assignment.strata) == 1
    assert sample_assignment.strata[0].field_name == "gender"

    # Check for approximate balance in arm assignment
    num_control = sum(1 for a in actual_assignments.assignments if a.arm_id == arm1_id)
    num_treat = sum(1 for a in actual_assignments.assignments if a.arm_id == arm2_id)
    assert abs(num_control - num_treat) <= 5  # Allow some wiggle room


async def test_create_freq_preassigned_experiment_fields_use_roundtrip(
    testing_datasource,
    aclient: AdminAPIClient,
):
    datasource_id = testing_datasource.datasource_id
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
            desired_n=100,
        ),
        webhooks=[],
    )

    created_experiment = aclient.create_experiment(
        datasource_id=datasource_id, body=experiment_request, random_state=42
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
    datasource_id = testing_datasource.datasource_id
    request_obj = make_create_preassigned_experiment_request(desired_n=100)

    created = aclient.create_experiment(datasource_id=datasource_id, body=request_obj, random_state=42).data
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
    assert all(arm_size.cluster_count is None for arm_size in create_summary.arm_sizes)
    assert all(arm_size.cluster_count is None for arm_size in get_summary.arm_sizes)
    assert len(create_summary.arm_sizes) == len(get_summary.arm_sizes)
    for create_arm, get_arm in zip(create_summary.arm_sizes, get_summary.arm_sizes, strict=True):
        assert create_arm.arm.arm_id == get_arm.arm.arm_id
        assert create_arm.arm.arm_name == get_arm.arm.arm_name
        assert create_arm.size == get_arm.size
        assert create_arm.cluster_count == get_arm.cluster_count


def test_create_and_get_freq_online_experiment(testing_datasource, aclient: AdminAPIClient):
    datasource_id = testing_datasource.datasource_id
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
    assert fetched_resp.experiment_schema is not None
    ui_experiment = fetched_resp.config
    diff = DeepDiff(
        created_experiment,
        ui_experiment,
        ignore_type_in_groups=[(CreateExperimentResponse, ExperimentConfig)],
    )
    assert not diff, f"Objects differ:\n{diff.pretty()}"


@pytest.mark.parametrize(
    ("reward_type", "prior_type"),
    [
        (LikelihoodTypes.BERNOULLI, PriorTypes.BETA),
        (LikelihoodTypes.NORMAL, PriorTypes.NORMAL),
        (LikelihoodTypes.BERNOULLI, PriorTypes.NORMAL),
    ],
)
def test_create_and_get_online_mab_experiment(testing_datasource, aclient: AdminAPIClient, reward_type, prior_type):
    datasource_id = testing_datasource.datasource_id
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
    assert fetched_resp.experiment_schema is None
    ui_experiment = fetched_resp.config
    diff = DeepDiff(
        created_experiment,
        ui_experiment,
        ignore_type_in_groups=[(CreateExperimentResponse, ExperimentConfig)],
    )
    assert not diff, f"Objects differ:\n{diff.pretty()}"


@pytest.mark.parametrize(
    ("reward_type", "prior_type"),
    [
        (LikelihoodTypes.NORMAL, PriorTypes.NORMAL),
        (LikelihoodTypes.BERNOULLI, PriorTypes.NORMAL),
    ],
)
def test_create_online_cmab_experiment(testing_datasource, aclient: AdminAPIClient, reward_type, prior_type):
    datasource_id = testing_datasource.datasource_id
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
    ("experiment_type", "reward_type", "prior_type"),
    [
        (ExperimentsType.MAB_ONLINE, LikelihoodTypes.NORMAL, PriorTypes.NORMAL),
        (ExperimentsType.MAB_ONLINE, LikelihoodTypes.BERNOULLI, PriorTypes.BETA),
        (ExperimentsType.CMAB_ONLINE, LikelihoodTypes.NORMAL, PriorTypes.NORMAL),
    ],
)
def test_create_online_mab_and_cmab_experiment_with_arm_weights(
    testing_datasource, aclient: AdminAPIClient, experiment_type, reward_type, prior_type
):
    datasource_id = testing_datasource.datasource_id
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
    organization_id = testing_experiment.organization_id
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


async def test_update_experiment_invalid(testing_datasource, testing_experiment, aclient: AdminAPIClient):
    """Test experiment update validation checks."""
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id

    request = UpdateExperimentRequest(start_date=testing_experiment.end_date + timedelta(days=1))
    with expect_status_code(422, detail_eq="New start date must be before end date."):
        aclient.update_experiment(datasource_id=datasource_id, experiment_id=experiment_id, body=request)

    request = UpdateExperimentRequest(end_date=testing_experiment.start_date - timedelta(days=1))
    with expect_status_code(422, detail_eq="New end date must be after start date."):
        aclient.update_experiment(datasource_id=datasource_id, experiment_id=experiment_id, body=request)

    # Lastly check invalid experiment state. Preassigned experiments are created in ASSIGNED
    # state before they are committed.
    assigned_experiment_id = aclient.create_experiment(
        datasource_id=testing_datasource.datasource_id,
        body=make_create_preassigned_experiment_request(desired_n=10),
        random_state=42,
    ).data.experiment_id

    with expect_status_code(422, detail_eq="Experiment must have been committed to be updated."):
        aclient.update_experiment(
            datasource_id=testing_datasource.datasource_id,
            experiment_id=assigned_experiment_id,
            body=UpdateExperimentRequest(name="updated"),
        )


async def test_update_arm(testing_experiment, aclient: AdminAPIClient):
    """Test updating an arm's metadata."""
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id
    arm_id = testing_experiment.arms[0].arm_id
    assert arm_id is not None
    request = UpdateArmRequest(name="updated name", description="updated desc")
    aclient.update_arm(datasource_id=datasource_id, experiment_id=experiment_id, arm_id=arm_id, body=request)

    updated_response = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=experiment_id)
    design_spec = updated_response.data.config.design_spec
    arm = next((arm for arm in design_spec.arms if arm.arm_id == arm_id), None)
    assert arm is not None
    assert arm.arm_name == "updated name"
    assert arm.arm_description == "updated desc"


async def test_update_arm_invalid(testing_datasource, testing_experiment, aclient: AdminAPIClient):
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
    assigned_experiment = aclient.create_experiment(
        datasource_id=testing_datasource.datasource_id,
        body=make_create_preassigned_experiment_request(desired_n=10),
        random_state=42,
    ).data
    assigned_arm_id = assigned_experiment.design_spec.arms[0].arm_id
    assert assigned_arm_id is not None

    with expect_status_code(422, detail_eq="Experiment must have been committed to update arms."):
        aclient.update_arm(
            datasource_id=testing_datasource.datasource_id,
            experiment_id=assigned_experiment.experiment_id,
            arm_id=assigned_arm_id,
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
    assert experiment_analysis.num_missing_participants == 0
    assert datetime.now(UTC) - experiment_analysis.created_at < timedelta(seconds=5)
    # Verify that only the first arm is marked as baseline by default
    metric_analysis = experiment_analysis.metric_analyses[0]
    baseline_arms = [arm for arm in metric_analysis.arm_analyses if arm.is_baseline]
    assert len(baseline_arms) == 1
    assert baseline_arms[0].is_baseline
    for metric_analysis in experiment_analysis.metric_analyses:
        assert {arm.arm_id for arm in metric_analysis.arm_analyses} == arm_ids(testing_experiment.arms)
        assert sum([arm.num_missing_values for arm in metric_analysis.arm_analyses]) == 0
        for analysis in metric_analysis.arm_analyses:
            assert analysis.ci_lower is not None
            assert analysis.ci_upper is not None
            assert analysis.ci_lower <= analysis.estimate <= analysis.ci_upper
            assert analysis.mean_ci_lower is not None
            assert analysis.mean_ci_upper is not None
            assert analysis.mean_ci_lower <= analysis.mean_ci_upper


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
    assert {arm.arm_id for arm in analyses} == arm_ids(testing_bandit_experiment.arms)
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
    design_spec = testing_bandit_experiment.config.design_spec
    assert isinstance(design_spec, CMABExperimentSpec)
    assert design_spec.contexts is not None

    experiment_analysis = aclient.analyze_cmab_experiment(
        datasource_id=datasource_id,
        experiment_id=experiment_id,
        body=CMABContextInputRequest(
            context_inputs=[
                ContextInput(context_id=context.context_id or "", context_value=1.0)
                for context in sorted(design_spec.contexts, key=lambda c: c.context_id or "")
            ]
        ),
    ).data
    assert isinstance(experiment_analysis, BanditExperimentAnalysisResponse)
    assert experiment_analysis.experiment_id == experiment_id
    assert len(experiment_analysis.arm_analyses) == len(testing_bandit_experiment.arms)
    assert experiment_analysis.n_outcomes == 10
    assert datetime.now(UTC) - experiment_analysis.created_at < timedelta(seconds=5)
    analyses = experiment_analysis.arm_analyses
    assert {arm.arm_id for arm in analyses} == arm_ids(testing_bandit_experiment.arms)
    for analysis in analyses:
        assert analysis.prior_pred_mean is not None
        assert analysis.prior_pred_stdev is not None
        assert analysis.post_pred_mean is not None
        assert analysis.post_pred_stdev is not None


async def test_create_and_update_outcome_mab_dwh_happy_path(
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    """End-to-end happy path for a MAB-DWH experiment through the public APIs: create it via the
    admin client, assign a participant, push an outcome, and read the outcome back."""
    experiment = await make_bandit_online_experiment(
        aclient,
        testing_datasource.datasource_id,
        experiment_type=ExperimentsType.MAB_ONLINE_DWH,
        prior_type=PriorTypes.NORMAL,
        reward_type=LikelihoodTypes.NORMAL,
    )
    experiment_id = experiment.config.experiment_id
    assert experiment.config.design_spec.experiment_type == ExperimentsType.MAB_ONLINE_DWH

    # Assign a participant via the experiments API.
    assignment = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=experiment_id,
        participant_id="p1",
    ).data.assignment
    assert assignment is not None
    assert assignment.outcome is None

    # Push an outcome. The default target column ("is_onboarded") is boolean, so the outcome must be 0 or 1.
    eclient.update_bandit_arm_with_participant_outcome(
        api_key=testing_datasource.key,
        body=UpdateBanditArmOutcomeRequest(outcome=1.0),
        experiment_id=experiment_id,
        participant_id="p1",
    )

    updated = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=experiment_id,
        participant_id="p1",
        create_if_none=False,
    ).data.assignment
    assert updated is not None
    assert updated.outcome == 1.0
    assert updated.observed_at is not None


async def test_mab_experiments_analyze_ignores_unobserved_draws_with_single_outcome(
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment = await make_bandit_online_experiment(
        aclient, testing_datasource.datasource_id, prior_type=PriorTypes.NORMAL, reward_type=LikelihoodTypes.NORMAL
    )

    eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=experiment.config.experiment_id,
        participant_id="p1",
    )
    eclient.update_bandit_arm_with_participant_outcome(
        api_key=testing_datasource.key,
        body=UpdateBanditArmOutcomeRequest(outcome=1.0),
        experiment_id=experiment.config.experiment_id,
        participant_id="p1",
    )

    analysis_before_unobserved_draw = aclient.analyze_experiment(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=experiment.config.experiment_id,
    ).data
    assert isinstance(analysis_before_unobserved_draw, BanditExperimentAnalysisResponse)

    eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=experiment.config.experiment_id,
        participant_id="p2",
    )

    analysis_after_unobserved_draw = aclient.analyze_experiment(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=experiment.config.experiment_id,
    ).data
    assert isinstance(analysis_after_unobserved_draw, BanditExperimentAnalysisResponse)

    assert normalize_bandit_analysis(analysis_after_unobserved_draw) == normalize_bandit_analysis(
        analysis_before_unobserved_draw
    )


async def test_mab_experiments_analyze_ignores_unobserved_draws_with_multiple_outcomes(
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment = await make_bandit_online_experiment(
        aclient, testing_datasource.datasource_id, prior_type=PriorTypes.NORMAL, reward_type=LikelihoodTypes.NORMAL
    )

    for participant_id, outcome in [("p1", 1.0), ("p2", 0.0)]:
        eclient.get_assignment(
            api_key=testing_datasource.key,
            experiment_id=experiment.config.experiment_id,
            participant_id=participant_id,
        )
        eclient.update_bandit_arm_with_participant_outcome(
            api_key=testing_datasource.key,
            body=UpdateBanditArmOutcomeRequest(outcome=outcome),
            experiment_id=experiment.config.experiment_id,
            participant_id=participant_id,
        )

    analysis_before_unobserved_draw = aclient.analyze_experiment(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=experiment.config.experiment_id,
    ).data
    assert isinstance(analysis_before_unobserved_draw, BanditExperimentAnalysisResponse)

    eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=experiment.config.experiment_id,
        participant_id="p3",
    )

    analysis_after_unobserved_draw = aclient.analyze_experiment(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=experiment.config.experiment_id,
    ).data
    assert isinstance(analysis_after_unobserved_draw, BanditExperimentAnalysisResponse)

    assert normalize_bandit_analysis(analysis_after_unobserved_draw) == normalize_bandit_analysis(
        analysis_before_unobserved_draw
    )


async def test_mab_experiments_analyze_with_assigned_but_unobserved_participants_matches_prior(
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment = await make_bandit_online_experiment(
        aclient, testing_datasource.datasource_id, prior_type=PriorTypes.NORMAL, reward_type=LikelihoodTypes.NORMAL
    )

    for participant_id in ["p1", "p2"]:
        eclient.get_assignment(
            api_key=testing_datasource.key,
            experiment_id=experiment.config.experiment_id,
            participant_id=participant_id,
        )

    experiment_analysis = aclient.analyze_experiment(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=experiment.config.experiment_id,
    ).data
    assert isinstance(experiment_analysis, BanditExperimentAnalysisResponse)

    assert experiment_analysis.n_outcomes == 0
    assert experiment_analysis.contexts is None
    for analysis in experiment_analysis.arm_analyses:
        assert analysis.post_pred_mean == analysis.prior_pred_mean
        assert analysis.post_pred_stdev == analysis.prior_pred_stdev


async def test_analyze_experiment_with_no_participants(testing_datasource, aclient: AdminAPIClient):
    datasource_id = testing_datasource.datasource_id
    experiment_id = (await make_freq_online_experiment(aclient, datasource_id)).config.experiment_id

    with expect_status_code(422, detail_eq="No participants found for experiment."):
        aclient.analyze_experiment(datasource_id=datasource_id, experiment_id=experiment_id)


async def test_analyze_experiment_whose_assignments_have_no_dwh_data(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    datasource_id = testing_datasource.datasource_id
    experiment_id = (await make_freq_online_experiment(aclient, datasource_id)).config.experiment_id

    eclient.get_assignment(api_key=testing_datasource.key, experiment_id=experiment_id, participant_id="0")

    with expect_status_code(
        422,
        detail_contains="Check that ids used in assignment are usable with your unique identifier (id)",
    ):
        aclient.analyze_experiment(datasource_id=datasource_id, experiment_id=experiment_id)


async def test_analyze_experiment_with_no_assignments_in_one_arm_yet(
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    datasource_id = testing_datasource.datasource_id
    experiment_id = (await make_freq_online_experiment(aclient, datasource_id)).config.experiment_id

    expected_num_assignments = 3
    assigned_arm_id: str | None = None
    for i in range(1, 1 + expected_num_assignments):
        # ?random_state=N ensures all assignments will use the same arm.
        response = eclient.client.get(
            f"/v1/experiments/{experiment_id}/assignments/{i}?random_state=0",
            headers={"X-API-Key": testing_datasource.key},
        )
        assert response.status_code == HTTPStatus.OK, response.content
        assignment = GetParticipantAssignmentResponse.model_validate_json(response.text).assignment
        assert assignment is not None
        assigned_arm_id = assigned_arm_id or assignment.arm_id
        assert assignment.arm_id == assigned_arm_id

    assert assigned_arm_id is not None

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
    assert {arm.arm_id for arm in analyses} == arm_ids(testing_bandit_experiment.arms)
    for analysis in analyses:
        assert analysis.prior_pred_mean is not None
        assert analysis.prior_pred_stdev is not None
        assert analysis.post_pred_mean == analysis.prior_pred_mean
        assert analysis.post_pred_stdev == analysis.prior_pred_stdev


@pytest.mark.parametrize(
    ("endpoint", "initial_state", "expected_status", "expected_detail"),
    [
        ("commit", ExperimentState.ASSIGNED, 204, None),  # Success case
        ("commit", ExperimentState.COMMITTED, 204, None),  # No-op
        ("abandon", ExperimentState.ASSIGNED, 204, None),  # Success case
        ("abandon", ExperimentState.ABANDONED, 204, None),  # No-op
        ("abandon", ExperimentState.COMMITTED, 409, "Invalid state: committed"),
    ],
)
def test_admin_experiment_state_setting(
    testing_datasource,
    endpoint,
    initial_state,
    expected_status,
    expected_detail,
    aclient: AdminAPIClient,
):
    datasource_id = testing_datasource.datasource_id

    def create_experiment_in_state(state: ExperimentState) -> str:
        experiment_id = aclient.create_experiment(
            datasource_id=datasource_id,
            body=make_create_preassigned_experiment_request(desired_n=10),
            random_state=42,
        ).data.experiment_id
        if state == ExperimentState.ASSIGNED:
            return experiment_id
        if state == ExperimentState.COMMITTED:
            aclient.commit_experiment(datasource_id=datasource_id, experiment_id=experiment_id)
            return experiment_id
        if state == ExperimentState.ABANDONED:
            aclient.abandon_experiment(datasource_id=datasource_id, experiment_id=experiment_id)
            return experiment_id
        raise ValueError(f"Unsupported public setup state: {state}")

    experiment_id = create_experiment_in_state(initial_state)

    if expected_detail:
        expect_kwargs = {"text": expected_detail}
    else:
        expect_kwargs = {}

    if endpoint == "commit":
        if expected_status == 204:
            response = aclient.commit_experiment(datasource_id=datasource_id, experiment_id=experiment_id).response
        else:
            with expect_status_code(expected_status, **expect_kwargs):
                aclient.commit_experiment(datasource_id=datasource_id, experiment_id=experiment_id)
    elif expected_status == 204:
        response = aclient.abandon_experiment(datasource_id=datasource_id, experiment_id=experiment_id).response
    else:
        with expect_status_code(expected_status, **expect_kwargs):
            aclient.abandon_experiment(datasource_id=datasource_id, experiment_id=experiment_id)

    # Verify
    if expected_status == 204:
        assert response.status_code == expected_status
        expected_state = ExperimentState.ABANDONED if endpoint == "abandon" else ExperimentState.COMMITTED
        experiment = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=experiment_id).data
        assert experiment.config.state == expected_state


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
        aclient.delete_api_key(datasource_id=testing_datasource.datasource_id, api_key_id="sample-key-id")


async def test_delete_apikey_authorized_and_nonexistent_allow_missing(testing_datasource, aclient: AdminAPIClient):
    aclient.delete_api_key(
        datasource_id=testing_datasource.datasource_id, api_key_id="sample-key-id", allow_missing=True
    )


async def test_delete_apikey_authorized_and_exists(testing_datasource, aclient: AdminAPIClient):
    aclient.delete_api_key(datasource_id=testing_datasource.datasource_id, api_key_id=testing_datasource.key_id)


async def test_delete_apikey_authorized_and_exists_allow_missing(testing_datasource, aclient: AdminAPIClient):
    aclient.delete_api_key(
        datasource_id=testing_datasource.datasource_id,
        api_key_id=testing_datasource.key_id,
        allow_missing=True,
    )


async def test_delete_apikey_authorized_and_exists_idempotency(testing_datasource, aclient: AdminAPIClient):
    aclient.delete_api_key(datasource_id=testing_datasource.datasource_id, api_key_id=testing_datasource.key_id)

    with expect_status_code(404):
        aclient.delete_api_key(datasource_id=testing_datasource.datasource_id, api_key_id=testing_datasource.key_id)

    aclient.delete_api_key(
        datasource_id=testing_datasource.datasource_id,
        api_key_id=testing_datasource.key_id,
        allow_missing=True,
    )


async def test_manage_apikeys(testing_datasource, aclient: AdminAPIClient):
    ds_id = testing_datasource.datasource_id
    first_key_id = testing_datasource.key_id

    create_api_key_response = aclient.create_api_key(datasource_id=ds_id).data
    assert create_api_key_response.datasource_id == ds_id
    created_key_id = create_api_key_response.id

    list_api_keys_response = aclient.list_api_keys(datasource_id=ds_id).data
    assert len(list_api_keys_response.items) == 2

    aclient.delete_api_key(datasource_id=ds_id, api_key_id=created_key_id)

    list_api_keys_response = aclient.list_api_keys(datasource_id=ds_id).data
    assert len(list_api_keys_response.items) == 1

    aclient.delete_api_key(datasource_id=ds_id, api_key_id=first_key_id)


async def test_experiment_webhook_integration(testing_datasource, aclient: AdminAPIClient):
    """Test creating an experiment with webhook associations and verifying webhook IDs in response."""
    org_id = testing_datasource.organization_id
    datasource_id = testing_datasource.datasource_id

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
            desired_n=100,
        ),
        webhooks=[webhook1_id],  # Only include the first webhook
    )

    create_response = aclient.create_experiment(datasource_id=datasource_id, body=experiment_request).data

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
            desired_n=100,
        ),
        # No webhooks field - should default to empty list
    )

    create_response_no_webhooks = aclient.create_experiment(
        datasource_id=datasource_id, body=experiment_request_no_webhooks
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
                desired_n=100,
            ),
        ),
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

    aclient.delete_snapshot(
        organization_id=create_organization_response.id,
        datasource_id=create_datasource_response.id,
        experiment_id=experiment_id,
        snapshot_id=success_snapshot.id,
    )

    with expect_status_code(404):
        aclient.delete_snapshot(
            organization_id=create_organization_response.id,
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


async def test_delete_snapshot_scopes_resource_to_datasource(
    aclient: AdminAPIClient,
    aclient_unpriv: AdminAPIClient,
):
    victim_org_id, victim_datasource_id = create_org_with_default_datasource(aclient, "snapshot-delete-victim-org")
    victim_experiment = await make_freq_online_experiment(aclient, victim_datasource_id)
    snapshot = aclient.create_snapshot(
        organization_id=victim_org_id,
        datasource_id=victim_datasource_id,
        experiment_id=victim_experiment.config.experiment_id,
    ).data
    caller_org_id, caller_datasource_id = create_org_with_default_datasource(
        aclient_unpriv, "snapshot-delete-caller-org"
    )

    with expect_status_code(404):
        aclient_unpriv.delete_snapshot(
            organization_id=caller_org_id,
            datasource_id=caller_datasource_id,
            experiment_id=victim_experiment.config.experiment_id,
            snapshot_id=snapshot.id,
        )

    response = aclient.get_snapshot(
        organization_id=victim_org_id,
        datasource_id=victim_datasource_id,
        experiment_id=victim_experiment.config.experiment_id,
        snapshot_id=snapshot.id,
    ).data
    assert response.snapshot.id == snapshot.id


def test_snapshot_on_ineligible_experiments(testing_datasource, aclient: AdminAPIClient):
    ds_id = testing_datasource.datasource_id
    org_id = testing_datasource.organization_id
    # The experiment created below is both too old and not yet committed.
    experiment_id = aclient.create_experiment(
        datasource_id=ds_id,
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
                desired_n=20,
            ),
        ),
    ).data.experiment_id

    # Assert non-committed experiments cannot be snapshotted.
    with expect_status_code(422, detail_eq="You can only snapshot committed experiments."):
        aclient.create_snapshot(
            organization_id=org_id,
            datasource_id=ds_id,
            experiment_id=experiment_id,
        )

    # So commit the experiment.
    aclient.commit_experiment(datasource_id=ds_id, experiment_id=experiment_id)

    # Assert old experiments cannot be snapshotted.
    with expect_status_code(422, detail_eq="You can only snapshot active experiments."):
        aclient.create_snapshot(
            organization_id=org_id,
            datasource_id=ds_id,
            experiment_id=experiment_id,
        )

    # But recently ended experiments can be snapshotted within a 1 day buffer.
    experiment_id = aclient.create_experiment(
        datasource_id=ds_id,
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
                desired_n=20,
            ),
        ),
    ).data.experiment_id
    aclient.commit_experiment(datasource_id=ds_id, experiment_id=experiment_id)
    aclient.create_snapshot(organization_id=org_id, datasource_id=ds_id, experiment_id=experiment_id)


def test_snapshot_with_nan(testing_datasource, aclient: AdminAPIClient):
    """Test that a snapshot with a NaN t-stat/p-value is handled correctly roundtrip."""
    ds_id = testing_datasource.datasource_id
    org_id = testing_datasource.organization_id
    experiment_id = aclient.create_experiment(
        datasource_id=ds_id,
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
                desired_n=10,
            ),
        ),
    ).data.experiment_id

    aclient.commit_experiment(datasource_id=ds_id, experiment_id=experiment_id)

    # Take a snapshot.
    create_snapshot_response = aclient.create_snapshot(
        organization_id=org_id, datasource_id=ds_id, experiment_id=experiment_id
    ).data

    # Verify the snapshot.
    snapshot_id = create_snapshot_response.id
    snapshot = aclient.get_snapshot(
        organization_id=org_id,
        datasource_id=ds_id,
        experiment_id=experiment_id,
        snapshot_id=snapshot_id,
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
    ds_id = testing_datasource.datasource_id
    with expect_status_code(404):
        aclient.delete_experiment_data(
            datasource_id=ds_id, experiment_id="not-an-experiment", body=DeleteExperimentDataRequest(snapshots=True)
        )


async def test_delete_experiment_data_assignments(
    testing_experiment: TestExperiment,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    """Test deleting arm assignments for an experiment."""
    ds_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id

    assignments_before = eclient.get_experiment_assignments(
        api_key=testing_experiment.api_key,
        experiment_id=experiment_id,
    ).data
    assert len(assignments_before.assignments) > 0
    assert assignments_before.sample_size > 0

    aclient.delete_experiment_data(
        datasource_id=ds_id,
        experiment_id=experiment_id,
        body=DeleteExperimentDataRequest(assignments=True),
    )

    assignments_after = eclient.get_experiment_assignments(
        api_key=testing_experiment.api_key,
        experiment_id=experiment_id,
    ).data
    assert assignments_after.assignments == []
    assert assignments_after.sample_size == 0
    experiment = aclient.get_experiment_for_ui(datasource_id=ds_id, experiment_id=experiment_id).data
    assert experiment.config.assign_summary is not None
    assert experiment.config.assign_summary.arm_sizes is not None
    assert all(arm_size.size == 0 for arm_size in experiment.config.assign_summary.arm_sizes)


@pytest.mark.parametrize(
    "testing_bandit_experiment",
    [(ExperimentsType.MAB_ONLINE, PriorTypes.BETA, LikelihoodTypes.BERNOULLI, 10)],
    indirect=True,
)
async def test_delete_experiment_data_draws(
    testing_bandit_experiment: TestExperiment,
    aclient: AdminAPIClient,
):
    """Test deleting draws for a bandit experiment."""
    ds_id = testing_bandit_experiment.datasource_id
    experiment_id = testing_bandit_experiment.id

    experiment = aclient.get_experiment_for_ui(datasource_id=ds_id, experiment_id=experiment_id).data
    assert experiment.config is not None
    assert experiment.config.assign_summary is not None
    assert experiment.config.assign_summary.arm_sizes is not None
    assert experiment.config.assign_summary.sample_size > 0

    aclient.delete_experiment_data(
        datasource_id=ds_id, experiment_id=experiment_id, body=DeleteExperimentDataRequest(assignments=True)
    )

    experiment = aclient.get_experiment_for_ui(datasource_id=ds_id, experiment_id=experiment_id).data
    assert experiment.config is not None
    assert experiment.config.assign_summary is not None
    assert experiment.config.assign_summary.arm_sizes is not None
    assert len(experiment.config.assign_summary.arm_sizes) > 0
    assert all(a.size == 0 for a in experiment.config.assign_summary.arm_sizes)


async def test_delete_experiment_data_snapshots(
    testing_experiment: TestExperiment,
    aclient: AdminAPIClient,
):
    """Test deleting snapshots for an experiment."""
    ds_id = testing_experiment.datasource_id
    org_id = testing_experiment.organization_id
    experiment_id = testing_experiment.id

    snapshot = aclient.create_snapshot(
        organization_id=org_id,
        datasource_id=ds_id,
        experiment_id=experiment_id,
    ).data
    snapshots_before = aclient.list_snapshots(
        organization_id=org_id,
        datasource_id=ds_id,
        experiment_id=experiment_id,
    ).data
    assert [item.id for item in snapshots_before.items] == [snapshot.id]

    aclient.delete_experiment_data(
        datasource_id=ds_id,
        experiment_id=experiment_id,
        body=DeleteExperimentDataRequest(snapshots=True),
    )

    snapshots_after = aclient.list_snapshots(
        organization_id=org_id,
        datasource_id=ds_id,
        experiment_id=experiment_id,
    ).data
    assert snapshots_after.items == []
    with expect_status_code(404):
        aclient.get_snapshot(
            organization_id=org_id,
            datasource_id=ds_id,
            experiment_id=experiment_id,
            snapshot_id=snapshot.id,
        )


async def test_delete_experiment_data_multiple(
    testing_experiment: TestExperiment,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    """Test deleting multiple data types at once."""
    ds_id = testing_experiment.datasource_id
    org_id = testing_experiment.organization_id
    experiment_id = testing_experiment.id

    aclient.create_snapshot(organization_id=org_id, datasource_id=ds_id, experiment_id=experiment_id)
    assignments_before = eclient.get_experiment_assignments(
        api_key=testing_experiment.api_key,
        experiment_id=experiment_id,
    ).data
    snapshots_before = aclient.list_snapshots(
        organization_id=org_id,
        datasource_id=ds_id,
        experiment_id=experiment_id,
    ).data
    assert assignments_before.assignments
    assert snapshots_before.items

    aclient.delete_experiment_data(
        datasource_id=ds_id,
        experiment_id=experiment_id,
        body=DeleteExperimentDataRequest(assignments=True, snapshots=True),
    )

    assignments_after = eclient.get_experiment_assignments(
        api_key=testing_experiment.api_key,
        experiment_id=experiment_id,
    ).data
    snapshots_after = aclient.list_snapshots(
        organization_id=org_id,
        datasource_id=ds_id,
        experiment_id=experiment_id,
    ).data
    assert assignments_after.assignments == []
    assert snapshots_after.items == []


async def test_delete_experiment_data_none_specified(
    testing_experiment: TestExperiment,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    """Test that specifying no data types deletes nothing."""
    ds_id = testing_experiment.datasource_id
    org_id = testing_experiment.organization_id
    experiment_id = testing_experiment.id

    snapshot = aclient.create_snapshot(organization_id=org_id, datasource_id=ds_id, experiment_id=experiment_id).data
    assignments_before = eclient.get_experiment_assignments(
        api_key=testing_experiment.api_key,
        experiment_id=experiment_id,
    ).data
    assert assignments_before.assignments

    aclient.delete_experiment_data(datasource_id=ds_id, experiment_id=experiment_id, body=DeleteExperimentDataRequest())

    assignments_after = eclient.get_experiment_assignments(
        api_key=testing_experiment.api_key,
        experiment_id=experiment_id,
    ).data
    snapshots_after = aclient.list_snapshots(
        organization_id=org_id,
        datasource_id=ds_id,
        experiment_id=experiment_id,
    ).data
    assert assignments_after.assignments == assignments_before.assignments
    assert [item.id for item in snapshots_after.items] == [snapshot.id]


async def test_create_freq_online_experiment_with_table_name_and_primary_key(
    testing_datasource, aclient: AdminAPIClient
):
    """Test creating an experiment with table_name and primary_key."""
    ds_id = testing_datasource.datasource_id

    request_json = make_createexperimentrequest_json(experiment_type=ExperimentsType.FREQ_ONLINE)
    initial_participant_count = len(aclient.list_participant_types(datasource_id=ds_id).data.items)
    experiment_request = CreateExperimentRequest.model_validate(request_json)

    created = aclient.create_experiment(
        datasource_id=ds_id,
        body=experiment_request,
        random_state=42,
    ).data

    # Verify no participant type was persisted to datasource config.
    assert len(aclient.list_participant_types(datasource_id=ds_id).data.items) == initial_participant_count

    # Verify datasource_table is set to the requested table name
    experiment = aclient.get_experiment_for_ui(datasource_id=ds_id, experiment_id=created.experiment_id).data
    assert isinstance(experiment.config.design_spec, OnlineFrequentistExperimentSpec)
    assert experiment.config.design_spec.table_name == "dwh"


def test_create_experiment_freq_design_spec_requires_primary_key(testing_datasource, aclient: AdminAPIClient):
    ds_id = testing_datasource.datasource_id
    request_json = make_createexperimentrequest_json(experiment_type=ExperimentsType.FREQ_ONLINE)
    del request_json["design_spec"]["primary_key"]
    with expect_status_code(422, text="primary_key", detail_eq="Field required"):
        aclient.create_experiment(datasource_id=ds_id, body=request_json)


def test_create_experiment_freq_design_spec_requires_table_name(testing_datasource, aclient: AdminAPIClient):
    ds_id = testing_datasource.datasource_id
    request_json = make_createexperimentrequest_json(experiment_type=ExperimentsType.FREQ_ONLINE)
    del request_json["design_spec"]["table_name"]
    with expect_status_code(422, text="table_name", detail_eq="Field required"):
        aclient.create_experiment(datasource_id=ds_id, body=request_json)


async def test_create_preassigned_experiment_with_table_name_and_primary_key(
    testing_datasource, aclient: AdminAPIClient
):
    ds_id = testing_datasource.datasource_id
    request_json = make_createexperimentrequest_json(experiment_type=ExperimentsType.FREQ_PREASSIGNED, desired_n=100)
    experiment_request = CreateExperimentRequest.model_validate(request_json)

    created = aclient.create_experiment(
        datasource_id=ds_id,
        body=experiment_request,
        random_state=42,
    ).data

    assert created.assign_summary is not None
    assert created.assign_summary.sample_size == 100

    experiment = aclient.get_experiment_for_ui(datasource_id=ds_id, experiment_id=created.experiment_id).data
    assert isinstance(experiment.config.design_spec, PreassignedFrequentistExperimentSpec)
    assert experiment.config.design_spec.table_name == "dwh"


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
                desired_n=100,
            ),
        ),
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
    org_id = testing_datasource.organization_id
    ds_id = testing_datasource.datasource_id

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
                    desired_n=100,
                ),
            ),
        ).data
        aclient.commit_experiment(datasource_id=ds_id, experiment_id=experiment.experiment_id)

    # Get all events first to know how many we have
    all_events = aclient.list_organization_events(organization_id=org_id).data
    total = len(all_events.items)
    assert total >= 3
    experiment_created_events = [ev for ev in all_events.items if ev.type == "experiment.created"]
    assert experiment_created_events
    assert all(ev.status_icon == "info" for ev in experiment_created_events)

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


def _create_experiment_created_event(aclient: AdminAPIClient, *, datasource_id: str, organization_id: str):
    experiment = aclient.create_experiment(
        datasource_id=datasource_id,
        body=CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_type=ExperimentsType.FREQ_PREASSIGNED,
                table_name="dwh",
                primary_key="id",
                experiment_name="resend event test",
                description="resend event test",
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime.now(UTC) + timedelta(days=1),
                arms=[
                    Arm(arm_name="control", arm_description="Control"),
                    Arm(arm_name="treatment", arm_description="Treatment"),
                ],
                metrics=[DesignSpecMetricRequest(field_name="income", metric_pct_change=5)],
                strata=[],
                filters=[],
                desired_n=100,
            ),
        ),
    ).data
    aclient.commit_experiment(datasource_id=datasource_id, experiment_id=experiment.experiment_id)

    expected_link = f"/datasources/{datasource_id}/experiments/{experiment.experiment_id}"
    events = aclient.list_organization_events(organization_id=organization_id).data.items
    return next(event for event in events if event.type == "experiment.created" and event.link == expected_link)


async def test_resend_organization_event_missing_event(aclient: AdminAPIClient):
    """Resending an unknown event id returns 404."""
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="resend-missing")).data.id
    with expect_status_code(404, text="Event not found"):
        aclient.resend_organization_event(organization_id=org_id, event_id="evt_does_not_exist")


def test_resend_organization_event_rejects_non_webhook_event(testing_datasource, aclient: AdminAPIClient):
    event = _create_experiment_created_event(
        aclient,
        datasource_id=testing_datasource.datasource_id,
        organization_id=testing_datasource.organization_id,
    )

    with expect_status_code(404, text="cannot be resent"):
        aclient.resend_organization_event(organization_id=testing_datasource.organization_id, event_id=event.id)


def test_resend_organization_event_cross_org_isolation(
    testing_datasource,
    testing_datasource_other,
    aclient: AdminAPIClient,
):
    event = _create_experiment_created_event(
        aclient,
        datasource_id=testing_datasource.datasource_id,
        organization_id=testing_datasource.organization_id,
    )

    with expect_status_code(404, text="Event not found"):
        aclient.resend_organization_event(
            organization_id=testing_datasource_other.organization_id,
            event_id=event.id,
        )


async def test_list_experiments(
    testing_datasource,
    testing_datasource_other,
    aclient: AdminAPIClient,
):
    """Test that listing experiments returns only non-abandoned/aborted experiments, in reverse chronological order,
    and scoped to the correct organization."""
    ds_id = testing_datasource.datasource_id
    org_id = testing_datasource.organization_id

    # Create three experiments: one will be committed, one left as assigned, one abandoned.
    exp1 = aclient.create_experiment(
        datasource_id=ds_id,
        body=make_create_freq_online_experiment_request(),
        random_state=42,
    ).data
    exp2 = aclient.create_experiment(
        datasource_id=ds_id,
        body=make_create_freq_online_experiment_request(),
        random_state=42,
    ).data
    exp3 = aclient.create_experiment(
        datasource_id=ds_id,
        body=make_create_freq_online_experiment_request(),
        random_state=42,
    ).data
    aclient.commit_experiment(datasource_id=ds_id, experiment_id=exp1.experiment_id)
    aclient.abandon_experiment(datasource_id=ds_id, experiment_id=exp3.experiment_id)

    # Create an experiment on a *different* organization's datasource to verify isolation.
    other_ds_id = testing_datasource_other.datasource_id
    aclient.add_member_to_organization(
        organization_id=testing_datasource_other.organization_id,
        body=AddMemberToOrganizationRequest(email=PRIVILEGED_EMAIL),
    )
    aclient.create_experiment(
        datasource_id=other_ds_id,
        body=make_create_freq_online_experiment_request(),
        random_state=42,
    )

    experiments = aclient.list_organization_experiments(organization_id=org_id).data
    experiment_ids = [item.experiment_id for item in experiments.items]

    # exp3 (abandoned) should be excluded; the other-org experiment should be excluded.
    assert len(experiments.items) == 2
    assert exp3.experiment_id not in experiment_ids

    # Verify ordering: most recently created first (exp2 before exp1).
    assert experiment_ids == [exp2.experiment_id, exp1.experiment_id]

    # Verify states.
    states = {item.experiment_id: item.state for item in experiments.items}
    assert states[exp1.experiment_id] == ExperimentState.COMMITTED
    assert states[exp2.experiment_id] == ExperimentState.ASSIGNED

    # Verify design_spec round-trips correctly.
    for item in experiments.items:
        assert item.design_spec is not None
        assert isinstance(item.design_spec, OnlineFrequentistExperimentSpec)


async def test_list_experiments_empty(
    testing_datasource,
    aclient: AdminAPIClient,
):
    """Test that listing experiments for an organization with no experiments returns an empty list."""
    org_id = testing_datasource.organization_id
    experiments = aclient.list_organization_experiments(organization_id=org_id).data
    assert experiments.items == []


async def test_power_check_with_missing_cluster_key_raises(testing_datasource, aclient: AdminAPIClient):
    """Power check raises a validation error if the cluster key column does not exist in the table."""
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="test missing cluster key",
        description="test power check with missing cluster key",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC) + timedelta(days=1),
        table_name=WIDE_DWH_PARTICIPANT_DEF.table_name,
        primary_key="id",
        arms=[Arm(arm_name="control", arm_description="C"), Arm(arm_name="treatment", arm_description="T")],
        metrics=[DesignSpecMetricRequest(field_name="household_income", metric_pct_change=0.1)],
        strata=[],
        filters=[],
        cluster_key="missing_key",
    )

    with expect_status_code(422, detail_contains="columns that do not exist in the table: missing_key"):
        aclient.power_check(
            datasource_id=testing_datasource.datasource_id,
            body=PowerRequest(design_spec=design_spec),
        )


async def test_power_check_with_manual_icc_and_nulls_in_cluster_key(testing_datasource, aclient: AdminAPIClient):
    """Power check accepts user-supplied ICC values and returns cluster analysis and handles nulls correctly."""
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="test cluster power",
        description="Verify null cluster key rows are excluded from manual ICC in power check.",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC) + timedelta(days=1),
        table_name=WIDE_DWH_PARTICIPANT_DEF.table_name,
        primary_key="id",
        arms=[Arm(arm_name="control", arm_description="C"), Arm(arm_name="treatment", arm_description="T")],
        metrics=[
            DesignSpecMetricRequest(
                field_name="household_income",
                metric_pct_change=0.1,
                icc=0.015,
                avg_cluster_size=10,
                cv=0.1,
            )
        ],
        strata=[],
        filters=[],
        cluster_key="age",
    )

    result = aclient.power_check(
        datasource_id=testing_datasource.datasource_id,
        body=PowerRequest(design_spec=design_spec),
    )

    assert len(result.data.analyses) == 1
    analysis = result.data.analyses[0]
    # Primary test: verify that the user-provided values are used.
    assert analysis.metric_spec.icc == 0.015
    assert analysis.metric_spec.avg_cluster_size == 10
    # assert analysis.metric_spec.cv == 0.3

    # Verify that the 28 null cluster key (age) rows are excluded from the analysis:
    assert analysis.metric_spec.available_n == 972
    # Verify that the 24 null outcome rows + 28 null cluster key rows are excluded here:
    assert analysis.metric_spec.available_nonnull_n == 948

    # Verify this was analyzed as a cluster-randomized experiment, and that with the user-supplied
    # estimates about the data we should have enough clusters to sample from despite the minimum
    # units needed being inflated by the design effect:
    assert analysis.num_clusters_total is not None
    assert analysis.num_clusters_total == 92
    assert analysis.design_effect is not None
    assert analysis.design_effect == pytest.approx(1.137, abs=1e-3)
    assert analysis.msg is not None
    assert analysis.msg.type == MetricPowerAnalysisMessageType.SUFFICIENT


async def test_power_check_with_db_derived_icc_and_nulls_in_cluster_key(testing_datasource, aclient: AdminAPIClient):
    """DB-derived ICC excludes rows with a null cluster key, and available_n is consistent.

    Uses wide_dwh with cluster_key="age", which has 28 null rows in a 1000-row table.
    This exercises the calculate_icc_and_cv_from_database path (no manual ICC supplied).

    The invariant being tested: both available_n (from get_stats_on_metrics) and the
    cluster size stats (from get_cluster_size_stats) should operate on the same 972
    non-null-age rows, so that the ICC and available_n passed to the power formula are
    consistent.
    """
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="test db-derived icc null exclusion",
        description="Verify null cluster key rows are excluded from DB-derived ICC in power check.",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC) + timedelta(days=1),
        table_name=WIDE_DWH_PARTICIPANT_DEF.table_name,
        primary_key="id",
        arms=[Arm(arm_name="control", arm_description="C"), Arm(arm_name="treatment", arm_description="T")],
        metrics=[DesignSpecMetricRequest(field_name="household_income", metric_pct_change=0.1)],
        strata=[],
        filters=[],
        cluster_key="age",
    )

    result = aclient.power_check(
        datasource_id=testing_datasource.datasource_id,
        body=PowerRequest(design_spec=design_spec),
    )

    assert len(result.data.analyses) == 1
    analysis = result.data.analyses[0]
    # ICC and cluster stats are computed from the DB (no manual values supplied).
    assert analysis.metric_spec.icc == pytest.approx(0.021, abs=1e-3)
    assert analysis.metric_spec.avg_cluster_size == pytest.approx(13.5, abs=1e-3)
    assert analysis.metric_spec.cv == pytest.approx(0.282, abs=1e-3)

    # The 28 null age rows are excluded from the eligible population as is the case with a manual ICC.
    assert analysis.metric_spec.available_n == 972
    # 24 rows have a null household_income among the 972 non-null-age rows.
    assert analysis.metric_spec.available_nonnull_n == 948

    # In this case, due to clustering we do not have a sufficient number of clusters (and units) to sample from.
    assert analysis.num_clusters_total == 78
    assert analysis.design_effect is not None
    assert analysis.design_effect == pytest.approx(1.287, abs=1e-3)
    assert analysis.target_n is not None
    assert analysis.target_n > analysis.metric_spec.available_nonnull_n
    assert analysis.msg is not None
    assert analysis.msg.type == MetricPowerAnalysisMessageType.INSUFFICIENT


async def test_power_check_with_calculated_icc(testing_datasource, aclient: AdminAPIClient):
    """Power check calculates ICC from the database when cluster_column is provided."""
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="test cluster power from DB",
        description="test power check calculating ICC from database",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC) + timedelta(days=1),
        table_name="clustered_dwh",
        primary_key="participant_id",
        arms=[
            Arm(arm_name="control", arm_description="Control"),
            Arm(arm_name="treatment", arm_description="Treatment"),
        ],
        metrics=[DesignSpecMetricRequest(field_name="test_score", metric_pct_change=0.1)],
        strata=[],
        filters=[],
        cluster_key="cluster_powerlaw",
    )

    result = aclient.power_check(
        datasource_id=testing_datasource.datasource_id,
        body=PowerRequest(design_spec=design_spec),
    )

    assert len(result.data.analyses) == 1
    analysis = result.data.analyses[0]
    assert analysis.metric_spec.icc is not None
    assert 0.15 <= analysis.metric_spec.icc <= 0.25
    assert analysis.metric_spec.avg_cluster_size is not None
    assert analysis.metric_spec.avg_cluster_size > 0
    assert analysis.metric_spec.cv is not None
    assert analysis.metric_spec.cv > 2.0
    assert analysis.design_effect is not None
    assert analysis.design_effect > 1.0
    assert analysis.num_clusters_total is not None
    assert analysis.num_clusters_total > 0


async def test_power_check_cluster_with_manual_and_db_derived_metrics(testing_datasource, aclient: AdminAPIClient):
    """Per-metric ICC: user-provided overrides apply only when icc is set; other metrics use the DWH."""
    manual_icc = 0.01
    manual_avg_cluster_size = 42.0
    manual_cv = 3.14

    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="test mixed cluster metrics",
        description="manual ICC for one metric, calculated for another",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC) + timedelta(days=1),
        table_name="clustered_dwh",
        primary_key="participant_id",
        arms=[Arm(arm_name="control", arm_description="C"), Arm(arm_name="treatment", arm_description="T")],
        metrics=[
            DesignSpecMetricRequest(
                field_name="test_score",
                metric_pct_change=0.1,
                icc=manual_icc,
                avg_cluster_size=manual_avg_cluster_size,
                cv=manual_cv,
            ),
            DesignSpecMetricRequest(field_name="converted", metric_pct_change=0.1),
        ],
        strata=[],
        filters=[],
        cluster_key="cluster_moderate",
    )

    result = aclient.power_check(
        datasource_id=testing_datasource.datasource_id, body=PowerRequest(design_spec=design_spec)
    )

    assert len(result.data.analyses) == 2
    analyses_by_field = {a.metric_spec.field_name: a for a in result.data.analyses}
    user_analysis = analyses_by_field["test_score"]
    dwh_analysis = analyses_by_field["converted"]

    # For this metric, check that we used the user-provided values:
    assert user_analysis.metric_spec.icc == manual_icc
    assert user_analysis.metric_spec.avg_cluster_size == manual_avg_cluster_size
    assert user_analysis.metric_spec.cv == manual_cv
    # Now check a few other properties of the response:
    assert user_analysis.msg is not None
    assert user_analysis.msg.type == MetricPowerAnalysisMessageType.SUFFICIENT
    assert user_analysis.design_effect is not None
    assert user_analysis.design_effect == pytest.approx(5.551, abs=1e-3)
    assert user_analysis.effective_sample_size is not None and user_analysis.target_n is not None
    assert user_analysis.effective_sample_size == pytest.approx(
        math.floor(user_analysis.target_n / user_analysis.design_effect)
    )
    assert user_analysis.num_clusters_total is not None
    assert user_analysis.num_clusters_total == math.ceil(
        user_analysis.target_n / user_analysis.metric_spec.avg_cluster_size
    )
    assert user_analysis.clusters_per_arm == [8, 8]

    # Check derived values are about what we would expect for the <cluster_moderate, converted>
    # grouping and metric as generated with:
    #    uv run python tools/generate_clustered_data.py --n-participants 10000 --n-clusters 1000
    assert dwh_analysis.metric_spec.icc is not None
    assert dwh_analysis.metric_spec.icc == pytest.approx(0.023, abs=1e-3)
    assert dwh_analysis.metric_spec.avg_cluster_size is not None
    assert dwh_analysis.metric_spec.avg_cluster_size == pytest.approx(10.0, abs=1e-3)
    assert dwh_analysis.metric_spec.cv == pytest.approx(0.347, abs=1e-3)
    # Now check a few other properties of the response:
    assert dwh_analysis.msg is not None
    assert dwh_analysis.msg.type == MetricPowerAnalysisMessageType.INSUFFICIENT
    assert dwh_analysis.design_effect is not None
    assert dwh_analysis.design_effect == pytest.approx(1.231, abs=1e-3)
    assert dwh_analysis.effective_sample_size is not None and dwh_analysis.target_n is not None
    assert dwh_analysis.effective_sample_size == pytest.approx(
        math.floor(dwh_analysis.target_n / dwh_analysis.design_effect)
    )
    assert dwh_analysis.num_clusters_total is not None
    assert dwh_analysis.num_clusters_total == math.ceil(
        dwh_analysis.target_n / dwh_analysis.metric_spec.avg_cluster_size
    )
    assert dwh_analysis.clusters_per_arm == [602, 602]


def test_create_freq_preassigned_experiment_with_cluster_key_roundtrips(
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    """cluster_key set on the design spec should survive save, reload, and assignment export."""
    ds_id = testing_datasource.datasource_id
    experiment_request = CreateExperimentRequest(
        design_spec=PreassignedFrequentistExperimentSpec(
            experiment_type="freq_preassigned",
            experiment_name="Cluster key roundtrip",
            description="Verify cluster_key is persisted and read back.",
            table_name="clustered_dwh",
            primary_key="participant_id",
            cluster_key="cluster_equal",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, 23, 59, 59, tzinfo=UTC),
            arms=[
                Arm(arm_name="control", arm_description="Control"),
                Arm(arm_name="treatment", arm_description="Treatment"),
            ],
            metrics=[DesignSpecMetricRequest(field_name="test_score", metric_pct_change=5)],
            strata=[],
            filters=[],
            desired_n=100,
            desired_n_clusters=10,
        ),
        webhooks=[],
    )

    created = aclient.create_experiment(datasource_id=ds_id, body=experiment_request, random_state=42).data
    assert isinstance(created.design_spec, PreassignedFrequentistExperimentSpec)
    assert created.design_spec.cluster_key == "cluster_equal"
    assert created.design_spec.desired_n_clusters == 10

    fetched = aclient.get_experiment_for_ui(datasource_id=ds_id, experiment_id=created.experiment_id).data
    assert isinstance(fetched.config.design_spec, PreassignedFrequentistExperimentSpec)
    assert fetched.config.design_spec.cluster_key == "cluster_equal"
    assert fetched.config.design_spec.desired_n_clusters == 10

    # Verify the created and fetched experiment assign_summary objects match via pydantic model equality.
    create_summary = created.assign_summary
    assert create_summary is not None
    assert create_summary.arm_sizes is not None
    fetched_summary = fetched.config.assign_summary
    assert create_summary == fetched_summary

    assignments = eclient.get_experiment_assignments(
        api_key=testing_datasource.key,
        experiment_id=created.experiment_id,
    ).data.assignments
    assert len(assignments) == 100
    # Verify the clusters seen in the individual assignments match the assign_summary cluster counts.
    clusters_by_arm: dict[str, set[str]] = defaultdict(set)
    participants_by_arm: dict[str, set[str]] = defaultdict(set)
    for assignment in assignments:
        assert assignment.cluster_key is not None
        clusters_by_arm[assignment.arm_id].add(assignment.cluster_key)
        participants_by_arm[assignment.arm_id].add(assignment.participant_id)
    for arm_size in create_summary.arm_sizes:
        arm_id = arm_size.arm.arm_id
        assert arm_id is not None
        assert len(clusters_by_arm[arm_id]) > 0
        assert arm_size.cluster_count == len(clusters_by_arm[arm_id])
        assert arm_size.size == len(participants_by_arm[arm_id])

    # Verify assignment exports via integration API above and UI API below are consistent and contain the cluster key.
    csv_response = aclient.client.get(f"/v1/m/datasources/{ds_id}/experiments/{created.experiment_id}/assignments/csv")
    assert csv_response.status_code == HTTPStatus.OK, csv_response.content
    csv_reader = csv.DictReader(io.StringIO(csv_response.text))
    assert csv_reader.fieldnames == ["participant_id", "cluster_key", "arm_id", "arm_name", "created_at"]
    csv_rows = {row["participant_id"]: row for row in csv_reader}
    assert set(csv_rows) == {assignment.participant_id for assignment in assignments}
    for assignment in assignments:
        csv_assignment = csv_rows[assignment.participant_id]
        assert csv_assignment["cluster_key"] == assignment.cluster_key
        assert csv_assignment["arm_id"] == assignment.arm_id
        assert csv_assignment["arm_name"] == assignment.arm_name
        created_at = assignment.created_at
        assert created_at is not None
        assert csv_assignment["created_at"] == created_at.isoformat(timespec="seconds").replace("+00:00", "Z")


def test_analyze_cluster_preassigned_experiment(testing_datasource, aclient: AdminAPIClient):
    """Basic test of clustered preassigned experiment analysis via the admin API."""
    datasource_id = testing_datasource.datasource_id
    experiment_request = CreateExperimentRequest(
        design_spec=PreassignedFrequentistExperimentSpec(
            experiment_type="freq_preassigned",
            experiment_name="Cluster analyze e2e",
            description="Verify clustered analysis via admin API.",
            table_name="clustered_dwh",
            primary_key="participant_id",
            cluster_key="cluster_equal",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, 23, 59, 59, tzinfo=UTC),
            arms=[Arm(arm_name="control", arm_description="C"), Arm(arm_name="treatment", arm_description="T")],
            metrics=[DesignSpecMetricRequest(field_name="converted", metric_pct_change=5)],
            strata=[],
            filters=[],
            desired_n=100,
            desired_n_clusters=10,
        ),
    )

    created = aclient.create_experiment(datasource_id=datasource_id, body=experiment_request, random_state=42).data
    exp_analysis = aclient.analyze_experiment(datasource_id=datasource_id, experiment_id=created.experiment_id).data
    assert isinstance(exp_analysis, FreqExperimentAnalysisResponse)
    assert exp_analysis.experiment_id == created.experiment_id
    assert exp_analysis.num_participants == 100
    assert exp_analysis.num_missing_participants == 0
    assert len(exp_analysis.metric_analyses) == 1
    metric_analysis = exp_analysis.metric_analyses[0]
    assert metric_analysis.metric_name == "converted"
    assert len(metric_analysis.arm_analyses) == 2
    for arm_analysis in metric_analysis.arm_analyses:
        assert arm_analysis.estimate is not None
        assert not np.isnan(arm_analysis.estimate)
        assert arm_analysis.std_error is not None
        assert not np.isnan(arm_analysis.std_error)


async def test_create_freq_preassigned_experiment_with_missing_cluster_key_raises(
    testing_datasource,
    aclient: AdminAPIClient,
):
    """Creating a preassigned experiment with a missing cluster key should raise a validation error."""
    datasource_id = testing_datasource.datasource_id
    experiment_request = CreateExperimentRequest(
        design_spec=PreassignedFrequentistExperimentSpec(
            experiment_type="freq_preassigned",
            experiment_name="Missing cluster key",
            description="Cluster key column does not exist in the table.",
            table_name=WIDE_DWH_PARTICIPANT_DEF.table_name,
            primary_key="id",
            cluster_key="missing_key",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, 23, 59, 59, tzinfo=UTC),
            arms=[Arm(arm_name="control", arm_description="C"), Arm(arm_name="treatment", arm_description="T")],
            metrics=[DesignSpecMetricRequest(field_name="household_income", metric_pct_change=5)],
            strata=[],
            filters=[],
            desired_n=100,
            desired_n_clusters=10,
        ),
        webhooks=[],
    )

    with expect_status_code(422, detail_contains="columns that do not exist in the table: missing_key"):
        aclient.create_experiment(datasource_id=datasource_id, body=experiment_request, random_state=42)


async def test_create_freq_preassigned_experiment_cluster_key_has_nulls(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Creating a cluster-randomized experiment with a cluster key that has nulls should exclude those rows."""
    datasource_id = testing_datasource.datasource_id
    experiment_request = CreateExperimentRequest(
        design_spec=PreassignedFrequentistExperimentSpec(
            experiment_type="freq_preassigned",
            experiment_name="Cluster key with null values",
            description="Cluster key has null values that should be excluded.",
            table_name=WIDE_DWH_PARTICIPANT_DEF.table_name,
            primary_key="id",
            cluster_key="age",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, 23, 59, 59, tzinfo=UTC),
            arms=[Arm(arm_name="control", arm_description="C"), Arm(arm_name="treatment", arm_description="T")],
            metrics=[DesignSpecMetricRequest(field_name="household_income", metric_pct_change=5)],
            strata=[],
            filters=[],
            desired_n=1000,
            desired_n_clusters=1000,
        ),
        webhooks=[],
    )

    created = aclient.create_experiment(datasource_id=datasource_id, body=experiment_request, random_state=42).data

    created_summary = created.assign_summary
    assert created_summary is not None
    # There are 1k rows in the wide_dwh table, and 28 rows where age=NULL that should be excluded:
    assert created_summary.sample_size == 972
    assert created_summary.arm_sizes is not None
    assert len(created_summary.arm_sizes) == 2
    assert created_summary.arm_sizes[0].size == 492
    assert created_summary.arm_sizes[1].size == 480
    assert created_summary.arm_sizes[0].cluster_count == 37
    assert created_summary.arm_sizes[1].cluster_count == 35

    assignments = eclient.get_experiment_assignments(
        api_key=testing_datasource.key,
        experiment_id=created.experiment_id,
    ).data.assignments
    clusters_by_arm: dict[str, set[str]] = defaultdict(set)
    for assignment in assignments:
        assert assignment.cluster_key is not None
        clusters_by_arm[assignment.arm_id].add(assignment.cluster_key)
    for arm_size in created_summary.arm_sizes:
        arm_id = arm_size.arm.arm_id
        assert arm_id is not None
        assert arm_size.cluster_count is not None
        assert arm_size.cluster_count == len(clusters_by_arm[arm_id])

    aclient.commit_experiment(datasource_id=datasource_id, experiment_id=created.experiment_id)
    fetched_summary = aclient.get_experiment_for_ui(
        datasource_id=datasource_id, experiment_id=created.experiment_id
    ).data.config.assign_summary
    assert fetched_summary is not None
    assert fetched_summary.arm_sizes is not None
    for create_arm, get_arm in zip(created_summary.arm_sizes, fetched_summary.arm_sizes, strict=True):
        assert create_arm.arm.arm_id == get_arm.arm.arm_id
        assert create_arm.size == get_arm.size
        assert create_arm.cluster_count == get_arm.cluster_count
    for arm_size in fetched_summary.arm_sizes:
        arm_id = arm_size.arm.arm_id
        assert arm_id is not None
        assert arm_size.cluster_count is not None
        assert arm_size.cluster_count == len(clusters_by_arm[arm_id])
