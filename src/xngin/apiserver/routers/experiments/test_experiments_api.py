from datetime import UTC, datetime, timedelta

from deepdiff import DeepDiff
from sqlalchemy import select

from xngin.apiserver import constants
from xngin.apiserver.models import tables
from xngin.apiserver.models.enums import ExperimentState, StopAssignmentReason
from xngin.apiserver.models.storage_format_converters import ExperimentStorageConverter
from xngin.apiserver.routers.common_api_types import (
    CreateExperimentResponse,
    GetParticipantAssignmentResponse,
    ListExperimentsResponse,
    PreassignedExperimentSpec,
)
from xngin.apiserver.routers.experiments.test_experiments_common import (
    insert_experiment_and_arms,
    make_create_preassigned_experiment_request,
)


def test_create_experiment_impl_invalid_design_spec(client_v1):
    """Test creating an experiment and saving assignments to the database."""
    request = make_create_preassigned_experiment_request(with_ids=True)

    response = client_v1.post(
        "/experiments/with-assignment",
        params={"chosen_n": 100},
        headers={constants.HEADER_CONFIG_ID: "testing"},
        content=request.model_dump_json(),
    )
    assert response.status_code == 422, request
    assert "UUIDs must not be set" in response.json()["message"]


async def test_create_experiment_with_assignment_sl(
    testing_datasource, use_deterministic_random, client_v1
):
    """Test creating an experiment and saving assignments to the database."""
    request = make_create_preassigned_experiment_request()

    response = client_v1.post(
        "/experiments/with-assignment",
        params={"chosen_n": 100},
        headers={
            constants.HEADER_CONFIG_ID: testing_datasource.ds.id,
            constants.HEADER_API_KEY: testing_datasource.key,
        },
        content=request.model_dump_json(),
    )

    # Verify basic response
    assert response.status_code == 200, request
    experiment_config = CreateExperimentResponse.model_validate(response.json())
    assert experiment_config.design_spec.experiment_id is not None
    assert experiment_config.design_spec.arms[0].arm_id is not None
    assert experiment_config.design_spec.arms[1].arm_id is not None
    assert experiment_config.datasource_id == testing_datasource.ds.id
    assert experiment_config.state == ExperimentState.ASSIGNED


async def test_list_experiments_sl_without_api_key(
    xngin_session, testing_datasource, client_v1
):
    """Tests that listing experiments tied to a db datasource requires an API key."""
    await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, state=ExperimentState.ASSIGNED
    )

    response = client_v1.get(
        "/experiments",
        headers={constants.HEADER_CONFIG_ID: testing_datasource.ds.id},
    )
    assert response.status_code == 403
    assert response.json()["message"] == "API key missing or invalid."


async def test_list_experiments_sl_with_api_key(
    xngin_session, testing_datasource, client_v1
):
    """Tests that listing experiments tied to a db datasource with an API key works."""
    expected_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, state=ExperimentState.ASSIGNED
    )

    response = client_v1.get(
        "/experiments",
        headers={
            constants.HEADER_CONFIG_ID: testing_datasource.ds.id,
            constants.HEADER_API_KEY: testing_datasource.key,
        },
    )
    assert response.status_code == 200
    experiments = ListExperimentsResponse.model_validate(response.json())
    assert len(experiments.items) == 1
    assert experiments.items[0].state == ExperimentState.ASSIGNED
    expected_design_spec = ExperimentStorageConverter(
        expected_experiment
    ).get_design_spec()
    diff = DeepDiff(expected_design_spec, experiments.items[0].design_spec)
    assert not diff, f"Objects differ:\n{diff.pretty()}"


async def test_get_experiment(xngin_session, testing_datasource, client_v1):
    new_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        state=ExperimentState.DESIGNING,
    )

    response = client_v1.get(
        f"/experiments/{new_experiment.id!s}",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
    )

    assert response.status_code == 200, response.content

    experiment_json = response.json()
    assert experiment_json["datasource_id"] == new_experiment.datasource_id
    assert experiment_json["state"] == new_experiment.state
    actual = PreassignedExperimentSpec.model_validate(experiment_json["design_spec"])
    expected = ExperimentStorageConverter(new_experiment).get_design_spec()
    diff = DeepDiff(actual, expected)
    assert not diff, f"Objects differ:\n{diff.pretty()}"


def test_get_experiment_assignments_not_found(testing_datasource, client_v1):
    """Test getting assignments for a non-existent experiment."""
    response = client_v1.get(
        f"/experiments/{tables.experiment_id_factory()}/assignments",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
    )
    assert response.status_code == 404, response.json()
    assert response.json()["detail"] == "Experiment not found or not authorized."


async def test_get_experiment_assignments_wrong_datasource(
    xngin_session, testing_datasource, testing_datasource_with_user, client_v1
):
    """Test getting assignments for an experiment from a different datasource."""
    # Create experiment in one datasource
    experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, state=ExperimentState.COMMITTED
    )

    # Try to get testing_datasource's experiment from another datasource's key.
    response = client_v1.get(
        f"/experiments/{experiment.id!s}/assignments",
        headers={constants.HEADER_API_KEY: testing_datasource_with_user.key},
    )
    assert response.status_code == 404, response.json()
    assert response.json()["detail"] == "Experiment not found or not authorized."


async def test_get_assignment_for_participant_with_apikey_preassigned(
    xngin_session, testing_datasource, client_v1
):
    preassigned_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds
    )
    assignment = tables.ArmAssignment(
        experiment_id=preassigned_experiment.id,
        participant_id="assigned_id",
        participant_type=preassigned_experiment.participant_type,
        arm_id=preassigned_experiment.arms[0].id,
        strata=[],
    )
    xngin_session.add(assignment)
    await xngin_session.commit()

    response = client_v1.get(
        f"/experiments/{preassigned_experiment.id!s}/assignments/unassigned_id?random_state=42",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
    )
    assert response.status_code == 200
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == preassigned_experiment.id
    assert parsed.participant_id == "unassigned_id"
    assert parsed.assignment is None

    response = client_v1.get(
        f"/experiments/{preassigned_experiment.id!s}/assignments/assigned_id",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
    )
    assert response.status_code == 200
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == preassigned_experiment.id
    assert parsed.participant_id == "assigned_id"
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == "control"


async def test_get_assignment_for_participant_with_apikey_online(
    xngin_session, testing_datasource, client_v1
):
    """Test endpoint that gets an assignment for a participant via API key."""
    online_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        assignment_type="online",
    )

    response = client_v1.get(
        f"/experiments/{online_experiment.id!s}/assignments/1",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
    )
    assert response.status_code == 200
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == online_experiment.id
    assert parsed.participant_id == "1"
    arms_map = {arm.id: arm.name for arm in online_experiment.arms}
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == arms_map[str(parsed.assignment.arm_id)]
    assert parsed.assignment.arm_name == "control"
    assert not parsed.assignment.strata

    # Test that we get the same assignment for the same participant.
    response2 = client_v1.get(
        f"/experiments/{online_experiment.id!s}/assignments/1",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
    )
    assert response2.status_code == 200
    assert response2.json() == response.json()

    # Make sure there's only one db entry.
    assignment = (
        await xngin_session.scalars(
            select(tables.ArmAssignment).where(
                tables.ArmAssignment.experiment_id == online_experiment.id
            )
        )
    ).one()
    assert assignment.participant_id == "1"
    assert assignment.arm_id == str(parsed.assignment.arm_id)

    # Verify no update to experiment lifecycle info.
    assert assignment.experiment.stopped_assignments_at is None
    assert assignment.experiment.stopped_assignments_reason is None


async def test_get_assignment_for_participant_with_apikey_online_dont_create(
    xngin_session, testing_datasource, client_v1
):
    """Verify endpoint doesn't create an assignment when create_if_none=False."""
    online_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        assignment_type="online",
    )

    response = client_v1.get(
        f"/experiments/{online_experiment.id!s}/assignments/1",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
        params={"create_if_none": False},
    )
    assert response.status_code == 200
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == online_experiment.id
    assert parsed.participant_id == "1"
    assert parsed.assignment is None


async def test_get_assignment_for_participant_with_apikey_online_past_end_date(
    xngin_session, testing_datasource, client_v1
):
    """Verify endpoint doesn't create an assignment for an online experiment that has ended."""
    online_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        assignment_type="online",
        end_date=datetime.now(UTC) - timedelta(days=1),
    )

    response = client_v1.get(
        f"/experiments/{online_experiment.id!s}/assignments/1",
        headers={
            constants.HEADER_CONFIG_ID: testing_datasource.ds.id,
            constants.HEADER_API_KEY: testing_datasource.key,
        },
    )
    assert response.status_code == 200
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == online_experiment.id
    assert parsed.participant_id == "1"
    assert parsed.assignment is None

    # Verify side effect to experiment lifecycle info.
    await xngin_session.refresh(online_experiment)
    assert online_experiment.stopped_assignments_at is not None
    assert online_experiment.stopped_assignments_reason == StopAssignmentReason.END_DATE
