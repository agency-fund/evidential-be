from collections.abc import Generator
from datetime import UTC, datetime, timedelta

import pytest

from xngin.apiserver.conftest import DatasourceMetadata, expect_status_code
from xngin.apiserver.routers.admin_integrations.admin_integrations_api_types import (
    SetConnectionToTurnRequest,
    SetTurnArmJourneyMappingRequest,
)
from xngin.apiserver.routers.common_api_types import (
    ArmBandit,
    CreateExperimentRequest,
    MABExperimentSpec,
    TurnConfigResponse,
)
from xngin.apiserver.routers.common_enums import (
    ExperimentsType,
    LikelihoodTypes,
    PriorTypes,
)
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.admin_integrations_api_client import AdminIntegrationsAPIClient
from xngin.apiserver.testing.integrations_api_client import IntegrationsAPIClient


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


@pytest.fixture(name="turn_config_response")
def fixture_turn_config_response(
    aclient: AdminAPIClient,
    iaclient: AdminIntegrationsAPIClient,
    testing_datasource: DatasourceMetadata,
    testing_design_spec: MABExperimentSpec,
) -> Generator[TurnConfigResponse]:
    """Configure a Turn connection for the org and save an arm→journey mapping for the experiment."""
    ds_id = testing_datasource.ds.id
    experiment = aclient.create_experiment(
        datasource_id=ds_id, body=CreateExperimentRequest(design_spec=testing_design_spec)
    ).data
    arm_ids = [arm.arm_id for arm in experiment.design_spec.arms]
    assert arm_ids[0] is not None
    assert arm_ids[1] is not None
    arm_to_journeys = {
        arm_ids[0]: f"journey-{arm_ids[0]}-uuid",
        arm_ids[1]: f"journey-{arm_ids[1]}-uuid",
    }
    iaclient.set_organization_turn_connection(
        organization_id=testing_datasource.org.id,
        body=SetConnectionToTurnRequest(turn_api_token="a" * 335),
    )
    iaclient.set_turn_arm_journey_mapping(
        datasource_id=testing_datasource.ds.id,
        experiment_id=experiment.experiment_id,
        body=SetTurnArmJourneyMappingRequest(arm_to_journeys=arm_to_journeys),
    )

    yield TurnConfigResponse.model_validate({
        "experiment_id": experiment.experiment_id,
        "experiment_name": "Test experiment",
        "arm_journey_map": arm_to_journeys,
    })

    iaclient.delete_turn_arm_journey_mapping(
        datasource_id=testing_datasource.ds.id, experiment_id=experiment.experiment_id, allow_missing=True
    )
    iaclient.delete_turn_connection_from_organization(organization_id=testing_datasource.org.id, allow_missing=True)


async def test_get_turn_app_config_returns_mapping(
    testing_datasource: DatasourceMetadata,
    iclient: IntegrationsAPIClient,
    turn_config_response: TurnConfigResponse,
):
    """Valid API key + existing mapping returns the configured arm→journey map."""
    turn_config = iclient.get_turn_app_config(
        experiment_id=turn_config_response.experiment_id, api_key=testing_datasource.key
    ).data

    assert turn_config.experiment_id == turn_config_response.experiment_id
    assert turn_config.arm_journey_map == turn_config_response.arm_journey_map


async def test_get_turn_app_config_404_when_no_mapping(
    testing_datasource: DatasourceMetadata,
    turn_config_response: TurnConfigResponse,
    iaclient: AdminIntegrationsAPIClient,
    iclient: IntegrationsAPIClient,
):
    """Experiment exists but has no ExperimentTurnConfig row."""
    iaclient.delete_turn_arm_journey_mapping(
        datasource_id=testing_datasource.ds.id, experiment_id=turn_config_response.experiment_id
    )
    with expect_status_code(404):
        iclient.get_turn_app_config(experiment_id=turn_config_response.experiment_id, api_key=testing_datasource.key)
