from datetime import UTC, datetime, timedelta

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.conftest import expect_status_code
from xngin.apiserver.routers.admin.admin_api_types import CreateOrganizationRequest
from xngin.apiserver.routers.admin_integrations.admin_integration_api_types import (
    SetConnectionToTurnRequest,
    SetTurnArmJourneyMappingRequest,
)
from xngin.apiserver.routers.common_api_types import Arm, DesignSpecMetricRequest, PreassignedFrequentistExperimentSpec
from xngin.apiserver.routers.common_enums import ExperimentState, ExperimentsType, StopAssignmentReason
from xngin.apiserver.routers.experiments.experiments_common import fetch_fields_or_raise
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.admin_integrations_api_client import AdminIntegrationsAPIClient
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_PARTICIPANT_DEF


@pytest.fixture(name="testing_experiment")
async def fixture_testing_experiment(xngin_session: AsyncSession, testing_datasource) -> tables.Experiment:
    """Create a preassigned experiment directly in our app db on the datasource with proper user permissions."""
    datasource = testing_datasource.ds

    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="test experiment",
        description="test experiment",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC) + timedelta(days=1),
        arms=[Arm(arm_name="Arm 1", arm_description="Arm 1"), Arm(arm_name="Arm 2", arm_description="Arm 2")],
        metrics=[DesignSpecMetricRequest(field_name="is_engaged", metric_pct_change=0.1)],
        strata=[],
        filters=[],
    )
    table_name = TESTING_DWH_PARTICIPANT_DEF.table_name
    primary_key = "id"
    field_type_map = await fetch_fields_or_raise(datasource, design_spec, table_name, primary_key)

    experiment_converter = ExperimentStorageConverter.init_from_components(
        datasource_id=datasource.id,
        organization_id=datasource.organization_id,
        experiment_type=design_spec.experiment_type,
        design_spec=design_spec,
        state=ExperimentState.COMMITTED,
        stopped_assignments_at=datetime.now(UTC),
        stopped_assignments_reason=StopAssignmentReason.PREASSIGNED,
        table_name=table_name,
        field_type_map=field_type_map,
        unique_id_name=primary_key,
    )
    experiment = experiment_converter.get_experiment()
    xngin_session.add(experiment)
    await xngin_session.commit()

    return experiment


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
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="test_turn_journeys_caching")).data.id
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token="a" * 335),
    )

    call_log: int = 0
    stacks: list[dict] = [{"name": "Arm A", "uuid": "arm-a-uuid"}]

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url, headers=None):
            nonlocal call_log
            call_log += 1
            return httpx.Response(
                status_code=200,
                json=list(stacks),
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    # First call: cache miss, one hit on the Turn API
    journeys = iaclient.get_organization_turn_journeys(organization_id=org_id).data.journeys
    assert call_log == 1
    assert journeys == {"Arm A": "arm-a-uuid"}

    # Second call inside the TTL: served from the DB cache.
    journeys = iaclient.get_organization_turn_journeys(organization_id=org_id).data.journeys
    assert call_log == 1
    assert journeys == {"Arm A": "arm-a-uuid"}

    # Rotating the token must invalidate the cache.
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token="b" * 335),
    )
    stacks[:] = [{"name": "Arm B", "uuid": "arm-b-uuid"}]

    journeys = iaclient.get_organization_turn_journeys(organization_id=org_id).data.journeys
    assert call_log == 2
    assert journeys == {"Arm B": "arm-b-uuid"}


async def test_turn_journey_mapping_lifecycle(
    testing_datasource,
    testing_experiment: tables.Experiment,
    aclient: AdminAPIClient,
    iaclient: AdminIntegrationsAPIClient,
):
    """Full lifecycle of the per-experiment arm->journey mapping: PUT, GET, update, DELETE."""
    ds_id = testing_datasource.ds.id
    org_id = testing_datasource.org.id
    experiment_id = testing_experiment.id
    arm_ids = [arm.id for arm in testing_experiment.arms]

    # PUT without a Turn connection configured for the org -> 400.
    with expect_status_code(400, text="No Turn.io connection"):
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
    testing_datasource, testing_experiment: tables.Experiment, iaclient: AdminIntegrationsAPIClient
):
    """PUT rejects a mapping whose keys do not exactly match the experiment's arm IDs."""
    ds_id = testing_datasource.ds.id
    org_id = testing_datasource.org.id
    experiment_id = testing_experiment.id
    arm_ids = [arm.id for arm in testing_experiment.arms]

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


async def test_rotating_token_wipes_arm_journey_mapping(
    testing_datasource,
    testing_experiment: tables.Experiment,
    iaclient: AdminIntegrationsAPIClient,
):
    """Rotating an org's Turn token deletes the org's arm->journey mappings."""
    ds_id = testing_datasource.ds.id
    org_id = testing_datasource.org.id
    experiment_id = testing_experiment.id
    arm_ids = [arm.id for arm in testing_experiment.arms]

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
