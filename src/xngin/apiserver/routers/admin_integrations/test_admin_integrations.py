from datetime import UTC, datetime, timedelta
from typing import ClassVar

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.conftest import expect_status_code
from xngin.apiserver.routers.admin.admin_api_types import CreateOrganizationRequest
from xngin.apiserver.routers.admin_integrations.admin_integrations_api_types import (
    SetConnectionToTurnRequest,
    SetTurnArmJourneyMappingRequest,
)
from xngin.apiserver.routers.common_api_types import (
    ArmBandit,
    CreateExperimentRequest,
    LikelihoodTypes,
    MABExperimentSpec,
    PriorTypes,
)
from xngin.apiserver.routers.common_enums import ExperimentsType
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.admin_integrations_api_client import AdminIntegrationsAPIClient


class FakeAsyncClient:
    call_log: ClassVar[int] = 0
    stacks: ClassVar[list[dict]] = [{"name": "Arm A", "uuid": "arm-a-uuid"}]

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def get(self, url, headers=None):
        FakeAsyncClient.call_log += 1
        return httpx.Response(
            status_code=200,
            json=list(FakeAsyncClient.stacks),
            request=httpx.Request("GET", url),
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


async def test_turn_connection_lifecycle(aclient: AdminAPIClient, iaclient: AdminIntegrationsAPIClient):
    """Test creating, rotating, previewing, and deleting an organization's Turn.io connection."""
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

    # Rotate: PUT with a new token.
    rotated_token = "fghij" * 67  # 335 chars
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token=rotated_token),
    )

    # Preview now reflects the new token.
    preview = iaclient.get_organization_turn_connection(organization_id=org_id).data.token_preview
    assert preview == rotated_token[-4:]

    # Delete.
    iaclient.delete_turn_connection_from_organization(organization_id=org_id)

    # GET after delete with allow_missing=False -> 404.
    with expect_status_code(404):
        iaclient.get_organization_turn_connection(organization_id=org_id)

    # Delete again without allow_missing -> 404.
    with expect_status_code(404):
        iaclient.delete_turn_connection_from_organization(organization_id=org_id)

    # Delete again with allow_missing -> 204.
    iaclient.delete_turn_connection_from_organization(organization_id=org_id, allow_missing=True)


async def test_turn_connection_encrypted_at_rest(
    xngin_session: AsyncSession, aclient: AdminAPIClient, iaclient: AdminIntegrationsAPIClient
):
    """The Turn.io API token must be encrypted at rest and recoverable via get_turn_api_token()."""
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


async def test_turn_journeys_caching(
    monkeypatch: pytest.MonkeyPatch, aclient: AdminAPIClient, iaclient: AdminIntegrationsAPIClient
):
    """GET /turn-connection/journeys serves from cache until TTL expires or the token is rotated."""
    # Reset the FakeAsyncClient's class-level state before the test.
    monkeypatch.setattr(FakeAsyncClient, "call_log", 0)
    monkeypatch.setattr(FakeAsyncClient, "stacks", [{"name": "Arm A", "uuid": "arm-a-uuid"}])
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="test_turn_journeys_caching")).data.id
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token="a" * 335),
    )

    # First call: cache miss, one hit on the Turn API
    journeys = iaclient.get_organization_turn_journeys(organization_id=org_id).data.journeys
    assert FakeAsyncClient.call_log == 1
    assert journeys == {"Arm A": "arm-a-uuid"}

    # Second call inside the TTL: served from the DB cache.
    journeys = iaclient.get_organization_turn_journeys(organization_id=org_id).data.journeys
    assert FakeAsyncClient.call_log == 1
    assert journeys == {"Arm A": "arm-a-uuid"}

    # Rotating the token must invalidate the cache.
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token="b" * 335),
    )
    FakeAsyncClient.stacks[:] = [{"name": "Arm B", "uuid": "arm-b-uuid"}]

    journeys = iaclient.get_organization_turn_journeys(organization_id=org_id).data.journeys
    assert FakeAsyncClient.call_log == 2
    assert journeys == {"Arm B": "arm-b-uuid"}


async def test_turn_journey_mapping_lifecycle(
    testing_datasource,
    testing_design_spec: MABExperimentSpec,
    aclient: AdminAPIClient,
    iaclient: AdminIntegrationsAPIClient,
):
    """Full lifecycle of the per-experiment arm->journey mapping: PUT, GET, update, DELETE."""
    ds_id = testing_datasource.ds.id
    org_id = testing_datasource.org.id
    experiment = aclient.create_experiment(
        datasource_id=ds_id, body=CreateExperimentRequest(design_spec=testing_design_spec)
    ).data
    experiment_id = experiment.experiment_id
    arm_ids = [arm.arm_id for arm in experiment.design_spec.arms]

    # PUT without a Turn connection configured for the org -> 400.
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
    got = iaclient.get_turn_arm_journey_mapping(datasource_id=ds_id, experiment_id=experiment_id).data.arm_to_journeys
    assert got == initial_mapping

    # PUT again with new values.
    rotated_mapping = {arm_ids[0]: "new-j0", arm_ids[1]: "new-j1"}
    iaclient.set_turn_arm_journey_mapping(
        datasource_id=ds_id,
        experiment_id=experiment_id,
        body=SetTurnArmJourneyMappingRequest(arm_to_journeys=rotated_mapping),
    )
    got = iaclient.get_turn_arm_journey_mapping(datasource_id=ds_id, experiment_id=experiment_id).data.arm_to_journeys
    assert got == rotated_mapping

    # DELETE.
    iaclient.delete_turn_arm_journey_mapping(datasource_id=ds_id, experiment_id=experiment_id)

    # GET after delete -> 404.
    with expect_status_code(404):
        iaclient.get_turn_arm_journey_mapping(datasource_id=ds_id, experiment_id=experiment_id)

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
):
    """PUT rejects a mapping whose keys do not exactly match the experiment's arm IDs."""
    ds_id = testing_datasource.ds.id
    org_id = testing_datasource.org.id
    experiment = aclient.create_experiment(
        datasource_id=ds_id, body=CreateExperimentRequest(design_spec=testing_design_spec)
    ).data
    experiment_id = experiment.experiment_id
    arm_ids = [arm.arm_id for arm in experiment.design_spec.arms]

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
                arm_to_journeys={arm_ids[0]: "journey-0"},
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
                    arm_ids[0]: "journey-0",
                    arm_ids[1]: "journey-1",
                    extra_id: "journey-extra",
                },
            ),
        )
    assert extra_id in match.http_response().text


async def test_rotating_or_deleting_token_wipes_arm_journey_mapping(
    testing_datasource,
    testing_design_spec: MABExperimentSpec,
    aclient: AdminAPIClient,
    iaclient: AdminIntegrationsAPIClient,
):
    """Rotating an org's Turn token deletes the org's arm->journey mappings."""
    ds_id = testing_datasource.ds.id
    org_id = testing_datasource.org.id
    experiment = aclient.create_experiment(
        datasource_id=ds_id, body=CreateExperimentRequest(design_spec=testing_design_spec)
    ).data
    experiment_id = experiment.experiment_id
    arm_ids = [arm.arm_id for arm in experiment.design_spec.arms]

    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token="a" * 335),
    )
    iaclient.set_turn_arm_journey_mapping(
        datasource_id=ds_id,
        experiment_id=experiment_id,
        body=SetTurnArmJourneyMappingRequest(
            arm_to_journeys={arm_ids[0]: "journey-0", arm_ids[1]: "journey-1"},
        ),
    )

    # Rotate the token.
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token="b" * 335),
    )

    # Mapping is wiped.
    with expect_status_code(404):
        iaclient.get_turn_arm_journey_mapping(datasource_id=ds_id, experiment_id=experiment_id)

    iaclient.set_turn_arm_journey_mapping(
        datasource_id=ds_id,
        experiment_id=experiment_id,
        body=SetTurnArmJourneyMappingRequest(
            arm_to_journeys={arm_ids[0]: "journey-2", arm_ids[1]: "journey-3"},
        ),
    )
    # Delete the token.
    iaclient.delete_turn_connection_from_organization(organization_id=org_id)

    # Mapping is wiped.
    with expect_status_code(404):
        iaclient.get_turn_arm_journey_mapping(datasource_id=ds_id, experiment_id=experiment_id)


async def test_resetting_same_token_preserves_arm_journey_mapping(
    testing_datasource,
    testing_design_spec: MABExperimentSpec,
    aclient: AdminAPIClient,
    iaclient: AdminIntegrationsAPIClient,
):
    """PUT with the same token must be idempotent: arm->journey mappings are preserved.

    Guards against an infrastructure-retried PUT silently wiping configured journey
    mappings just because the request was replayed with the same token.
    """
    ds_id = testing_datasource.ds.id
    org_id = testing_datasource.org.id
    experiment = aclient.create_experiment(
        datasource_id=ds_id, body=CreateExperimentRequest(design_spec=testing_design_spec)
    ).data
    experiment_id = experiment.experiment_id
    arm_ids = [arm.arm_id for arm in experiment.design_spec.arms]

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

    got = iaclient.get_turn_arm_journey_mapping(datasource_id=ds_id, experiment_id=experiment_id).data.arm_to_journeys
    assert got == mapping


async def test_resetting_same_token_preserves_journeys_cache(
    monkeypatch: pytest.MonkeyPatch,
    aclient: AdminAPIClient,
    iaclient: AdminIntegrationsAPIClient,
):
    """PUT with the same token must not invalidate the cached Turn.io journey list."""
    org_id = aclient.create_organizations(
        body=CreateOrganizationRequest(name="test_resetting_same_token_preserves_journeys_cache")
    ).data.id

    token = "a" * 335
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token=token),
    )

    # Reset the FakeAsyncClient's class-level state before the test.
    monkeypatch.setattr(FakeAsyncClient, "call_log", 0)
    monkeypatch.setattr(FakeAsyncClient, "stacks", [{"name": "Arm A", "uuid": "arm-a-uuid"}])
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    # Prime the cache.
    iaclient.get_organization_turn_journeys(organization_id=org_id)
    assert FakeAsyncClient.call_log == 1

    # Replay the PUT with the same token; cache must remain valid.
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token=token),
    )

    iaclient.get_organization_turn_journeys(organization_id=org_id)
    assert FakeAsyncClient.call_log == 1, "Same-token PUT must not invalidate the journeys cache"
