from collections.abc import Generator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.conftest import expect_status_code
from xngin.apiserver.routers.admin_integrations.admin_integration_api_types import (
    SetConnectionToTurnRequest,
    SetTurnArmJourneyMappingRequest,
)
from xngin.apiserver.routers.common_api_types import (
    Arm,
    DesignSpecMetricRequest,
    PreassignedFrequentistExperimentSpec,
)
from xngin.apiserver.routers.common_enums import (
    ExperimentState,
    ExperimentsType,
    StopAssignmentReason,
)
from xngin.apiserver.routers.experiments.experiments_common import fetch_fields_or_raise
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter
from xngin.apiserver.testing.admin_integrations_api_client import AdminIntegrationsAPIClient
from xngin.apiserver.testing.integrations_api_client import IntegrationsAPIClient
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_PARTICIPANT_DEF


@pytest.fixture(name="testing_experiment")
async def fixture_testing_experiment(xngin_session: AsyncSession, testing_datasource) -> tables.Experiment:
    """Create a preassigned experiment directly in the app db on the datasource with proper user permissions."""
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


@pytest.fixture(name="arm_to_journey_mapping")
def fixture_arm_to_journey_mapping(
    iaclient: AdminIntegrationsAPIClient,
    testing_datasource,
    testing_experiment: tables.Experiment,
) -> Generator[dict[str, str]]:
    """Configure a Turn connection for the org and save an arm→journey mapping for the experiment."""
    arm_to_journeys = {arm.id: f"journey-{arm.id}-uuid" for arm in testing_experiment.arms}
    iaclient.set_organization_turn_connection(
        organization_id=testing_datasource.org.id,
        body=SetConnectionToTurnRequest(turn_api_token="a" * 335),
    )
    iaclient.set_turn_arm_journey_mapping(
        datasource_id=testing_datasource.ds.id,
        experiment_id=testing_experiment.id,
        body=SetTurnArmJourneyMappingRequest(arm_to_journeys=arm_to_journeys),
    )

    yield arm_to_journeys

    iaclient.delete_turn_arm_journey_mapping(
        datasource_id=testing_datasource.ds.id, experiment_id=testing_experiment.id, allow_missing=True
    )
    iaclient.delete_turn_connection_from_organization(organization_id=testing_datasource.org.id, allow_missing=True)


async def test_get_turn_app_config_returns_mapping(
    testing_experiment: tables.Experiment,
    testing_datasource,
    iclient: IntegrationsAPIClient,
    arm_to_journey_mapping,
):
    """Valid API key + existing mapping returns the configured arm→journey map."""
    turn_config = iclient.get_turn_app_config(experiment_id=testing_experiment.id, api_key=testing_datasource.key).data

    assert turn_config.experiment_id == testing_experiment.id
    assert turn_config.experiment_name == testing_experiment.name
    assert turn_config.arm_journey_map == arm_to_journey_mapping


async def test_get_turn_app_config_404_when_no_mapping(
    testing_experiment: tables.Experiment,
    testing_datasource,
    arm_to_journey_mapping,
    iaclient: AdminIntegrationsAPIClient,
    iclient: IntegrationsAPIClient,
):
    """Experiment exists but has no ExperimentTurnConfig row."""
    iaclient.delete_turn_arm_journey_mapping(
        datasource_id=testing_experiment.datasource_id, experiment_id=testing_experiment.id
    )
    with expect_status_code(404):
        iclient.get_turn_app_config(experiment_id=testing_experiment.id, api_key=testing_datasource.key)
