from collections.abc import Generator
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.conftest import DatasourceMetadata, expect_status_code
from xngin.apiserver.routers.admin.admin_api_types import AddWebhookToOrganizationRequest
from xngin.apiserver.routers.admin_integrations.admin_integrations_api_types import (
    SetConnectionToTurnRequest,
    SetTurnArmJourneyMappingRequest,
)
from xngin.apiserver.routers.admin_integrations.test_admin_integrations_api import (
    FakeAsyncClient,
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
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.admin_integrations_api_client import AdminIntegrationsAPIClient
from xngin.apiserver.testing.integrations_api_client import IntegrationsAPIClient
from xngin.tq.task_payload_types import TURN_JOURNEYS_CHANGED_TASK_TYPE


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
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TurnConfigResponse]:
    """Configure a Turn connection for the org and save an arm→journey mapping for the experiment."""
    ds_id = testing_datasource.datasource_id
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

    monkeypatch.setattr(FakeAsyncClient, "call_log", 0)
    monkeypatch.setattr(
        FakeAsyncClient,
        "stacks",
        [
            {"name": "Journey 0", "uuid": f"journey-{arm_ids[0]}-uuid"},
            {"name": "Journey 1", "uuid": f"journey-{arm_ids[1]}-uuid"},
        ],
    )
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    iaclient.set_organization_turn_connection(
        organization_id=testing_datasource.organization_id,
        body=SetConnectionToTurnRequest(turn_api_token="a" * 335),
    )
    iaclient.set_turn_arm_journey_mapping(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=experiment.experiment_id,
        body=SetTurnArmJourneyMappingRequest(arm_to_journeys=arm_to_journeys),
    )

    yield TurnConfigResponse.model_validate({
        "experiment_id": experiment.experiment_id,
        "experiment_name": "Test experiment",
        "arm_journey_map": arm_to_journeys,
    })

    iaclient.delete_turn_connection_from_organization(
        organization_id=testing_datasource.organization_id, allow_missing=True
    )


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
        datasource_id=testing_datasource.datasource_id, experiment_id=turn_config_response.experiment_id
    )
    with expect_status_code(404):
        iclient.get_turn_app_config(experiment_id=turn_config_response.experiment_id, api_key=testing_datasource.key)


@pytest.fixture(name="inbound_turn_webhook")
async def fixture_inbound_turn_webhook(
    aclient: AdminAPIClient,
    testing_datasource: DatasourceMetadata,
):
    """Creates an inbound turn.journeys_changed webhook; yields (webhook_id, auth_token)."""
    webhook = aclient.add_webhook_to_organization(
        organization_id=testing_datasource.organization_id,
        body=AddWebhookToOrganizationRequest(
            direction="inbound",
            type="turn.journeys_changed",
            name="test-inbound-turn-webhook",
            url=None,
        ),
    ).data
    return webhook.id, webhook.auth_token


async def test_turn_webhook_enqueues_task(
    iclient: IntegrationsAPIClient,
    xngin_session: AsyncSession,
    testing_datasource: DatasourceMetadata,
    inbound_turn_webhook: tuple[str, str | None],
):
    """Valid Webhook-Token results in 204 and a turn.journeys_changed task in the queue."""
    webhook_id, auth_token = inbound_turn_webhook

    iclient.turn_webhook(webhook_id=webhook_id, auth_token=auth_token)

    tasks = (
        (
            await xngin_session.execute(
                select(tables.Task).where(tables.Task.task_type == TURN_JOURNEYS_CHANGED_TASK_TYPE)
            )
        )
        .scalars()
        .all()
    )

    assert len(tasks) == 1
    assert tasks[0].payload == {"organization_id": testing_datasource.organization_id}


async def test_turn_webhook_404_for_unknown_id(
    iclient: IntegrationsAPIClient,
):
    """An unrecognised webhook_id returns 404."""
    with expect_status_code(404):
        iclient.turn_webhook(webhook_id="wh_doesnotexist", auth_token="any-token")


async def test_turn_webhook_401_for_wrong_token(
    iclient: IntegrationsAPIClient,
    inbound_turn_webhook: tuple[str, str | None],
):
    """Correct webhook_id but wrong Webhook-Token returns 401."""
    webhook_id, _ = inbound_turn_webhook
    with expect_status_code(401):
        iclient.turn_webhook(webhook_id=webhook_id, auth_token="wrong-token")


async def test_turn_webhook_401_for_missing_token(
    iclient: IntegrationsAPIClient,
    inbound_turn_webhook: tuple[str, str | None],
):
    """No Webhook-Token header returns 401."""
    webhook_id, _ = inbound_turn_webhook
    with expect_status_code(401):
        iclient.turn_webhook(webhook_id=webhook_id)
