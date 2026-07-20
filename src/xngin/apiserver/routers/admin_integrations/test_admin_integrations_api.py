from datetime import UTC, datetime, timedelta
from typing import ClassVar

import httpx2
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.conftest import expect_status_code
from xngin.apiserver.routers.admin.admin_api_types import CreateOrganizationRequest
from xngin.apiserver.routers.admin.test_admin_api import make_bandit_online_experiment
from xngin.apiserver.routers.admin_integrations.admin_integrations_api_types import (
    SetConnectionToTurnRequest,
    SetTurnArmJourneyMappingRequest,
)
from xngin.apiserver.routers.common_api_types import (
    Arm,
    ArmBandit,
    CreateExperimentRequest,
    DesignSpecMetricRequest,
    Filter,
    LikelihoodTypes,
    MABExperimentSpec,
    OnlineFrequentistExperimentSpec,
    PreassignedFrequentistExperimentSpec,
    PriorTypes,
)
from xngin.apiserver.routers.common_enums import ExperimentsType, Relation
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.admin_integrations_api_client import AdminIntegrationsAPIClient
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_PARTICIPANT_DEF


class FakeAsyncClient:
    call_log: ClassVar[int] = 0
    stacks: ClassVar[list[dict]] = [{"name": "Arm A", "uuid": "arm-a-uuid"}]
    expected_status: ClassVar[int] = 200

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def request(self, method, url, headers=None):
        FakeAsyncClient.call_log += 1
        return httpx2.Response(
            status_code=FakeAsyncClient.expected_status,
            json=list(FakeAsyncClient.stacks),
            request=httpx2.Request(method, url),
        )


@pytest.fixture(name="testing_design_spec")
async def fixture_testing_design_spec() -> MABExperimentSpec:
    """Create a preassigned experiment directly in our app db on the datasource with proper user permissions."""

    return MABExperimentSpec(
        experiment_type=ExperimentsType.MAB_ONLINE,
        experiment_name="test experiment",
        description="test experiment",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC) + timedelta(days=1),
        arms=[
            ArmBandit(arm_name="Arm 1", arm_description="Arm 1", alpha_init=1, beta_init=1),
            ArmBandit(arm_name="Arm 2", arm_description="Arm 2", alpha_init=1, beta_init=1),
        ],
        prior_type=PriorTypes.BETA,
        reward_type=LikelihoodTypes.BERNOULLI,
    )


async def test_turn_connection_lifecycle(
    monkeypatch: pytest.MonkeyPatch, aclient: AdminAPIClient, iaclient: AdminIntegrationsAPIClient
):
    """Test creating, rotating, previewing, and deleting an organization's Turn.io connection."""
    monkeypatch.setattr(FakeAsyncClient, "call_log", 0)
    monkeypatch.setattr(FakeAsyncClient, "stacks", [{"name": "Arm A", "uuid": "arm-a-uuid"}])
    monkeypatch.setattr(httpx2, "AsyncClient", FakeAsyncClient)

    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="test_turn_connection_lifecycle")).data.id

    # GET before a connection exists -> 404.
    with expect_status_code(404):
        iaclient.get_organization_turn_connection(organization_id=org_id)

    # GET with allow_missing=True -> 200.
    assert (
        iaclient.get_organization_turn_connection(organization_id=org_id, allow_missing=True).data.token_preview == ""
    )

    # Create the connection.
    initial_token = "abcde" * 67  # 335 chars
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token=initial_token),
    )

    # GET returns a preview of the last 4 chars of the token.
    preview = iaclient.get_organization_turn_connection(organization_id=org_id).data.token_preview
    assert preview == initial_token[-4:]

    # Check that Journeys were also fetched from client
    journeys = iaclient.get_organization_turn_journeys(organization_id=org_id).data.journeys
    assert FakeAsyncClient.call_log == 1
    assert {journey.name: journey.uuid for journey in journeys} == {"Arm A": "arm-a-uuid"}

    # Rotate: PUT with a new token.
    rotated_token = "fghij" * 67  # 335 chars
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token=rotated_token),
    )

    # Preview now reflects the new token.
    preview = iaclient.get_organization_turn_connection(organization_id=org_id).data.token_preview
    assert preview == rotated_token[-4:]

    # Check that Journeys were refetched from client after rotation
    journeys = iaclient.get_organization_turn_journeys(organization_id=org_id).data.journeys
    assert FakeAsyncClient.call_log == 2
    assert {journey.name: journey.uuid for journey in journeys} == {"Arm A": "arm-a-uuid"}

    # Delete.
    iaclient.delete_turn_connection_from_organization(organization_id=org_id)

    # GET after delete with allow_missing=False -> 404.
    with expect_status_code(404):
        iaclient.get_organization_turn_connection(organization_id=org_id)

    # GET Journeys after delete -> 404.
    with expect_status_code(404):
        iaclient.get_organization_turn_journeys(organization_id=org_id)

    # Get Journeys with  after delete -> 200 with empty journeys.
    # journeys = iaclient.get_organization_turn_journeys(organization_id=org_id).data.journeys
    # assert journeys == {}

    # Delete again without allow_missing -> 404.
    with expect_status_code(404):
        iaclient.delete_turn_connection_from_organization(organization_id=org_id)

    # Delete again with allow_missing -> 204.
    iaclient.delete_turn_connection_from_organization(organization_id=org_id, allow_missing=True)


async def test_turn_connection_encrypted_at_rest(
    monkeypatch: pytest.MonkeyPatch,
    xngin_session: AsyncSession,
    aclient: AdminAPIClient,
    iaclient: AdminIntegrationsAPIClient,
):
    """The Turn.io API token must be encrypted at rest and recoverable via get_turn_api_token()."""
    monkeypatch.setattr(FakeAsyncClient, "call_log", 0)
    monkeypatch.setattr(FakeAsyncClient, "stacks", [{"name": "Arm A", "uuid": "arm-a-uuid"}])
    monkeypatch.setattr(httpx2, "AsyncClient", FakeAsyncClient)

    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="test_turn_connection_encrypted")).data.id

    token = "abcde" * 67  # 335 chars
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token=token),
    )

    row = (
        await xngin_session.execute(
            select(tables.TurnConnection).where(tables.TurnConnection.organization_id == org_id)
        )
    ).scalar_one()

    assert token not in row.encrypted_turn_api_token
    assert row.turn_api_token_preview == token[-4:]
    assert row.get_turn_api_token() == token


async def test_turn_journeys_api_error_handling(
    monkeypatch: pytest.MonkeyPatch, aclient: AdminAPIClient, iaclient: AdminIntegrationsAPIClient
):
    """GET /turn-connection/journeys must handle errors from the Turn API gracefully."""
    # Reset the FakeAsyncClient's class-level state before the test.
    monkeypatch.setattr(FakeAsyncClient, "call_log", 0)
    monkeypatch.setattr(httpx2, "AsyncClient", FakeAsyncClient)

    org_id = aclient.create_organizations(
        body=CreateOrganizationRequest(name="test_turn_journeys_api_error_handling")
    ).data.id

    # Simulate a non-2xx response.
    monkeypatch.setattr(FakeAsyncClient, "expected_status", 403)
    with expect_status_code(502, text="Turn.io API returned non-2xx status"):
        iaclient.set_organization_turn_connection(
            organization_id=org_id,
            body=SetConnectionToTurnRequest(turn_api_token="a" * 335),
        )
    # Check that API key is not set
    with expect_status_code(404):
        iaclient.get_organization_turn_connection(organization_id=org_id)

    # Simulate an incorrect response structure (missing 'name' and 'uuid').
    monkeypatch.setattr(FakeAsyncClient, "stacks", [{"wrong_field": "value"}])
    monkeypatch.setattr(FakeAsyncClient, "expected_status", 200)
    with expect_status_code(422, text="The retrieved journeys from Turn.io did not have the expected fields"):
        iaclient.set_organization_turn_connection(
            organization_id=org_id,
            body=SetConnectionToTurnRequest(turn_api_token="a" * 335),
        )

    # Simulate a network error.
    async def _raise_request_error(self, method, url, headers=None):
        raise httpx2.RequestError("Network error", request=httpx2.Request(method, url))

    monkeypatch.setattr(FakeAsyncClient, "request", _raise_request_error)
    with expect_status_code(502, text="Failed to reach Turn.io API"):
        iaclient.set_organization_turn_connection(
            organization_id=org_id,
            body=SetConnectionToTurnRequest(turn_api_token="a" * 335),
        )
    # Check that API key is not set
    with expect_status_code(404):
        iaclient.get_organization_turn_connection(organization_id=org_id)


async def test_turn_journey_mapping_lifecycle(
    testing_datasource,
    testing_design_spec: MABExperimentSpec,
    aclient: AdminAPIClient,
    iaclient: AdminIntegrationsAPIClient,
    monkeypatch: pytest.MonkeyPatch,
):
    """Full lifecycle of the per-experiment arm->journey mapping: PUT, GET, update, DELETE."""
    ds_id = testing_datasource.datasource_id
    org_id = testing_datasource.organization_id
    experiment = aclient.create_experiment(
        datasource_id=ds_id, body=CreateExperimentRequest(design_spec=testing_design_spec)
    ).data
    experiment_id = experiment.experiment_id
    arm_ids = [arm.arm_id for arm in experiment.design_spec.arms]
    monkeypatch.setattr(FakeAsyncClient, "call_log", 0)
    monkeypatch.setattr(
        FakeAsyncClient,
        "stacks",
        [{"name": "journey-0", "uuid": "journey-0-uuid"}, {"name": "journey-1", "uuid": "journey-1-uuid"}],
    )
    monkeypatch.setattr(httpx2, "AsyncClient", FakeAsyncClient)

    # PUT without a Turn connection configured for the org -> 409.
    with expect_status_code(409, text="No Turn.io connection"):
        iaclient.set_turn_arm_journey_mapping(
            datasource_id=ds_id,
            experiment_id=experiment_id,
            body=SetTurnArmJourneyMappingRequest(
                arm_to_journeys={arm_ids[0]: "journey-0", arm_ids[1]: "journey-1"},
            ),
        )

    # Configure a Turn connection so subsequent calls can proceed.
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token="a" * 335),
    )

    # GET before any mapping has been saved -> 404.
    with expect_status_code(404):
        iaclient.get_turn_arm_journey_mapping(datasource_id=ds_id, experiment_id=experiment_id)

    # PUT saves a valid mapping.
    initial_mapping = {arm_ids[0]: "journey-0-uuid", arm_ids[1]: "journey-1-uuid"}
    iaclient.set_turn_arm_journey_mapping(
        datasource_id=ds_id,
        experiment_id=experiment_id,
        body=SetTurnArmJourneyMappingRequest(arm_to_journeys=initial_mapping),
    )

    # GET returns the saved mapping.
    got = iaclient.get_turn_arm_journey_mapping(datasource_id=ds_id, experiment_id=experiment_id).data
    assert got.arm_to_journeys == initial_mapping
    assert got.stale_arm_ids == []

    # PUT again with new values updates the mapping
    rotated_mapping = {arm_ids[0]: "new-j0", arm_ids[1]: "new-j1"}
    iaclient.set_turn_arm_journey_mapping(
        datasource_id=ds_id,
        experiment_id=experiment_id,
        body=SetTurnArmJourneyMappingRequest(arm_to_journeys=rotated_mapping),
    )
    got = iaclient.get_turn_arm_journey_mapping(datasource_id=ds_id, experiment_id=experiment_id).data
    assert got.arm_to_journeys == rotated_mapping
    assert set(got.stale_arm_ids) == set(arm_ids)

    # DELETE turn connection from org, which should also delete the mapping.
    iaclient.delete_turn_connection_from_organization(organization_id=org_id)

    # GET after delete -> 404.
    with expect_status_code(404):
        iaclient.get_turn_arm_journey_mapping(datasource_id=ds_id, experiment_id=experiment_id)

    # Re-establish the Turn connection
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token="a" * 335),
    )

    # PUT mapping again.
    initial_mapping = {arm_ids[0]: "journey-0-uuid", arm_ids[1]: "journey-1-uuid"}
    iaclient.set_turn_arm_journey_mapping(
        datasource_id=ds_id,
        experiment_id=experiment_id,
        body=SetTurnArmJourneyMappingRequest(arm_to_journeys=initial_mapping),
    )

    # Then DELETE the mapping directly.
    iaclient.delete_turn_arm_journey_mapping(datasource_id=ds_id, experiment_id=experiment_id)

    # DELETE again without allow_missing -> 404.
    with expect_status_code(404):
        iaclient.delete_turn_arm_journey_mapping(datasource_id=ds_id, experiment_id=experiment_id)

    # DELETE with allow_missing -> 204.
    iaclient.delete_turn_arm_journey_mapping(datasource_id=ds_id, experiment_id=experiment_id, allow_missing=True)


async def test_turn_journey_mapping_rejects_mismatched_arm_ids(
    testing_datasource,
    testing_design_spec: MABExperimentSpec,
    aclient: AdminAPIClient,
    iaclient: AdminIntegrationsAPIClient,
    monkeypatch: pytest.MonkeyPatch,
):
    """PUT rejects a mapping whose keys do not exactly match the experiment's arm IDs."""
    ds_id = testing_datasource.datasource_id
    org_id = testing_datasource.organization_id
    experiment = aclient.create_experiment(
        datasource_id=ds_id, body=CreateExperimentRequest(design_spec=testing_design_spec)
    ).data
    experiment_id = experiment.experiment_id
    arm_ids = [arm.arm_id for arm in experiment.design_spec.arms]
    monkeypatch.setattr(FakeAsyncClient, "call_log", 0)
    monkeypatch.setattr(
        FakeAsyncClient,
        "stacks",
        [{"name": "journey-0", "uuid": "journey-0-uuid"}, {"name": "journey-1", "uuid": "journey-1-uuid"}],
    )
    monkeypatch.setattr(httpx2, "AsyncClient", FakeAsyncClient)

    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token="a" * 335),
    )

    # Missing one arm -> 400 mentioning the missing id.
    with expect_status_code(400, text="Missing") as match:
        iaclient.set_turn_arm_journey_mapping(
            datasource_id=ds_id,
            experiment_id=experiment_id,
            body=SetTurnArmJourneyMappingRequest(
                arm_to_journeys={arm_ids[0]: "journey-0-uuid"},
            ),
        )
    assert arm_ids[1] in match.http_response().text

    # Extra arm id not belonging to the experiment -> 400 mentioning the extra id.
    extra_id = "arm_not_in_experiment_1"
    with expect_status_code(400, text="Extra") as match:
        iaclient.set_turn_arm_journey_mapping(
            datasource_id=ds_id,
            experiment_id=experiment_id,
            body=SetTurnArmJourneyMappingRequest(
                arm_to_journeys={
                    arm_ids[0]: "journey-0-uuid",
                    arm_ids[1]: "journey-1-uuid",
                    extra_id: "journey-extra",
                },
            ),
        )
    assert extra_id in match.http_response().text


async def test_resetting_same_token_preserves_arm_journey_mapping(
    testing_datasource,
    testing_design_spec: MABExperimentSpec,
    aclient: AdminAPIClient,
    iaclient: AdminIntegrationsAPIClient,
    monkeypatch: pytest.MonkeyPatch,
):
    """PUT with the same token must be idempotent: arm->journey mappings are preserved.

    Guards against an infrastructure-retried PUT silently wiping configured journey
    mappings just because the request was replayed with the same token.
    """
    ds_id = testing_datasource.datasource_id
    org_id = testing_datasource.organization_id
    experiment = aclient.create_experiment(
        datasource_id=ds_id, body=CreateExperimentRequest(design_spec=testing_design_spec)
    ).data
    experiment_id = experiment.experiment_id
    arm_ids = [arm.arm_id for arm in experiment.design_spec.arms]
    monkeypatch.setattr(FakeAsyncClient, "call_log", 0)
    monkeypatch.setattr(
        FakeAsyncClient,
        "stacks",
        [{"name": "journey-0", "uuid": "journey-0-uuid"}, {"name": "journey-1", "uuid": "journey-1-uuid"}],
    )
    monkeypatch.setattr(httpx2, "AsyncClient", FakeAsyncClient)

    token = "a" * 335
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token=token),
    )

    mapping = {arm_ids[0]: "journey-0", arm_ids[1]: "journey-1"}
    iaclient.set_turn_arm_journey_mapping(
        datasource_id=ds_id,
        experiment_id=experiment_id,
        body=SetTurnArmJourneyMappingRequest(arm_to_journeys=mapping),
    )

    # Replay the PUT with the same token.
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token=token),
    )

    got = iaclient.get_turn_arm_journey_mapping(datasource_id=ds_id, experiment_id=experiment_id).data
    assert got.arm_to_journeys == mapping


async def test_get_experiment_sample_calls_for_mab(
    testing_datasource, aclient: AdminAPIClient, iaclient: AdminIntegrationsAPIClient
):
    """A MAB experiment's sample calls carry example calls with a type-correct outcome."""
    experiment = await make_bandit_online_experiment(
        aclient,
        testing_datasource.datasource_id,
        prior_type=PriorTypes.NORMAL,
        reward_type=LikelihoodTypes.NORMAL,
    )
    sample_calls = iaclient.get_experiment_sample_calls(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=experiment.config.experiment_id,
    ).data
    assert sample_calls is not None
    assert [c.label for c in sample_calls.calls] == ["Get assignment", "Report outcome"]
    outcome_call = sample_calls.calls[1]
    assert outcome_call.method == "POST"
    assert outcome_call.path.endswith("/outcome")
    assert experiment.config.experiment_id in outcome_call.path
    assert outcome_call.body == {"outcome": 1.5}  # NORMAL reward => real-valued example


async def test_get_experiment_sample_calls_none_for_cmab(
    testing_datasource, aclient: AdminAPIClient, iaclient: AdminIntegrationsAPIClient
):
    """CMAB assignment needs a context vector, so there are no sample calls yet (deferred)."""
    experiment = await make_bandit_online_experiment(
        aclient,
        testing_datasource.datasource_id,
        experiment_type=ExperimentsType.CMAB_ONLINE,
        prior_type=PriorTypes.NORMAL,
        reward_type=LikelihoodTypes.NORMAL,
    )
    sample_calls = iaclient.get_experiment_sample_calls(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=experiment.config.experiment_id,
    ).data
    assert sample_calls is None


async def test_get_experiment_sample_calls_preassigned_frequentist_get_assignment_only(
    testing_datasource, aclient: AdminAPIClient, iaclient: AdminIntegrationsAPIClient
):
    """A preassigned frequentist experiment gets only the get-assignment example: report-outcome is
    bandit-only, and assign_with_filters is freq_online-only."""
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
    sample_calls = iaclient.get_experiment_sample_calls(
        datasource_id=datasource_id,
        experiment_id=created.experiment_id,
    ).data
    assert sample_calls is not None
    assert [c.label for c in sample_calls.calls] == ["Get assignment"]
    assert all("outcome" not in c.path for c in sample_calls.calls)


async def test_get_experiment_sample_calls_freq_online_with_filters(
    testing_datasource, aclient: AdminAPIClient, iaclient: AdminIntegrationsAPIClient
):
    """A freq_online experiment with filters also gets an assign_with_filters example, so integrators
    apply their filters when creating assignments instead of unintentionally assigning everyone."""
    datasource_id = testing_datasource.datasource_id
    created = aclient.create_experiment(
        datasource_id=datasource_id,
        body=CreateExperimentRequest(
            design_spec=OnlineFrequentistExperimentSpec(
                experiment_type=ExperimentsType.FREQ_ONLINE,
                experiment_name="test experiment",
                description="test experiment",
                table_name=TESTING_DWH_PARTICIPANT_DEF.table_name,
                primary_key="id",
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime.now(UTC) + timedelta(days=1),
                arms=[Arm(arm_name="C", arm_description="C"), Arm(arm_name="T", arm_description="T")],
                metrics=[DesignSpecMetricRequest(field_name="is_engaged", metric_pct_change=0.1)],
                strata=[],
                filters=[Filter(field_name="gender", relation=Relation.INCLUDES, value=["Male"])],
            ),
        ),
        random_state=42,
    ).data
    aclient.commit_experiment(datasource_id=datasource_id, experiment_id=created.experiment_id)
    sample_calls = iaclient.get_experiment_sample_calls(
        datasource_id=datasource_id,
        experiment_id=created.experiment_id,
    ).data
    assert sample_calls is not None
    # The assigning POST comes first: it's the call filtered integrators should use. The GET is
    # read-only (create_if_none=false) since a plain GET can't evaluate filters.
    assert [c.label for c in sample_calls.calls] == ["Get assignment (with filters)", "Get existing assignment"]
    filtered_call = sample_calls.calls[0]
    assert filtered_call.method == "POST"
    assert filtered_call.path.endswith("/assign_with_filters")
    assert filtered_call.body == {"properties": [{"field_name": "gender", "value": "<value>"}]}
    get_call = sample_calls.calls[1]
    assert get_call.method == "GET"
    assert get_call.path.endswith("?create_if_none=false")
