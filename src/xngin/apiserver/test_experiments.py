from deepdiff import DeepDiff
from fastapi.testclient import TestClient
from sqlalchemy import select
from xngin.apiserver import conftest, constants
from xngin.apiserver.models import tables
from xngin.apiserver.models.storage_format_converters import ExperimentStorageConverter
from xngin.apiserver.routers.stateless_api_types import (
    PreassignedExperimentSpec,
)
from xngin.apiserver.main import app
from xngin.apiserver.models.enums import ExperimentState
from xngin.apiserver.routers.experiments_api_types import (
    CreateExperimentResponse,
    GetParticipantAssignmentResponse,
    ListExperimentsResponse,
)
from xngin.apiserver.test_experiments_common import (  # pylint: disable=unused-import
    fixture_teardown,  # noqa: F401
    insert_experiment_and_arms,
    make_create_preassigned_experiment_request,
)

conftest.setup(app)
client = TestClient(app)
client.base_url = client.base_url.join(constants.API_PREFIX_V1)


def test_create_experiment_impl_invalid_design_spec():
    """Test creating an experiment and saving assignments to the database."""
    request = make_create_preassigned_experiment_request(with_ids=True)

    response = client.post(
        "/experiments/with-assignment",
        params={"chosen_n": 100},
        headers={constants.HEADER_CONFIG_ID: "testing"},
        content=request.model_dump_json(),
    )
    assert response.status_code == 422, request
    assert "UUIDs must not be set" in response.json()["message"]


def test_create_experiment_with_assignment_sl(xngin_session, use_deterministic_random):
    """Test creating an experiment and saving assignments to the database."""
    # First create a datasource to maintain proper referential integrity, but with a local config so we know we can read our dwh data.
    ds_metadata = conftest.make_datasource_metadata(
        xngin_session, datasource_id="testing"
    )
    request = make_create_preassigned_experiment_request()

    response = client.post(
        "/experiments/with-assignment",
        params={"chosen_n": 100},
        headers={
            constants.HEADER_CONFIG_ID: ds_metadata.ds.id,
            constants.HEADER_API_KEY: ds_metadata.key,
        },
        content=request.model_dump_json(),
    )

    # Verify basic response
    assert response.status_code == 200, request
    experiment_config = CreateExperimentResponse.model_validate(response.json())
    assert experiment_config.design_spec.experiment_id is not None
    assert experiment_config.design_spec.arms[0].arm_id is not None
    assert experiment_config.design_spec.arms[1].arm_id is not None
    assert experiment_config.datasource_id == ds_metadata.ds.id
    assert experiment_config.state == ExperimentState.ASSIGNED


def test_list_experiments_sl_without_api_key(xngin_session, testing_datasource):
    """Tests that listing experiments tied to a db datasource requires an API key."""
    insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, state=ExperimentState.ASSIGNED
    )

    response = client.get(
        "/experiments",
        headers={constants.HEADER_CONFIG_ID: testing_datasource.ds.id},
    )
    assert response.status_code == 403
    assert response.json()["message"] == "API key missing or invalid."


def test_list_experiments_sl_with_api_key(xngin_session, testing_datasource):
    """Tests that listing experiments tied to a db datasource with an API key works."""
    expected_experiment, _ = insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, state=ExperimentState.ASSIGNED
    )

    response = client.get(
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


def test_get_experiment(xngin_session, testing_datasource):
    new_experiment, _ = insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        state=ExperimentState.DESIGNING,
    )

    response = client.get(
        f"/experiments/{new_experiment.id!s}",
        headers={
            constants.HEADER_CONFIG_ID: testing_datasource.ds.id,
            constants.HEADER_API_KEY: testing_datasource.key,
        },
    )

    assert response.status_code == 200, response.content

    experiment_json = response.json()
    assert experiment_json["datasource_id"] == new_experiment.datasource_id
    assert experiment_json["state"] == new_experiment.state
    actual = PreassignedExperimentSpec.model_validate(experiment_json["design_spec"])
    expected = ExperimentStorageConverter(new_experiment).get_design_spec()
    diff = DeepDiff(actual, expected)
    assert not diff, f"Objects differ:\n{diff.pretty()}"


def test_get_experiment_assignments_not_found():
    """Test getting assignments for a non-existent experiment."""
    response = client.get(
        f"/experiments/{tables.experiment_id_factory()}/assignments",
        headers={constants.HEADER_CONFIG_ID: "testing"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment not found"


def test_get_experiment_assignments_wrong_datasource(xngin_session, testing_datasource):
    """Test getting assignments for an experiment from a different datasource."""
    # Create experiment in one datasource
    experiment, _ = insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, state=ExperimentState.COMMITTED
    )

    # Try to get it from another datasource
    response = client.get(
        f"/experiments/{experiment.id!s}/assignments",
        headers={constants.HEADER_CONFIG_ID: "testing-inline-schema"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment not found"


def test_get_assignment_for_preassigned_participant_with_apikey(
    xngin_session, testing_datasource
):
    preassigned_experiment, arms = insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
    )
    assignment = tables.ArmAssignment(
        experiment_id=preassigned_experiment.id,
        participant_id="assigned_id",
        participant_type=preassigned_experiment.participant_type,
        arm_id=arms[0].id,
        strata=[],
    )
    xngin_session.add(assignment)
    xngin_session.commit()

    response = client.get(
        f"/experiments/{preassigned_experiment.id!s}/assignments/unassigned_id?random_state=42",
        headers={
            constants.HEADER_CONFIG_ID: testing_datasource.ds.id,
            constants.HEADER_API_KEY: testing_datasource.key,
        },
    )
    assert response.status_code == 200
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == preassigned_experiment.id
    assert parsed.participant_id == "unassigned_id"
    assert parsed.assignment is None

    response = client.get(
        f"/experiments/{preassigned_experiment.id!s}/assignments/assigned_id",
        headers={
            constants.HEADER_CONFIG_ID: testing_datasource.ds.id,
            constants.HEADER_API_KEY: testing_datasource.key,
        },
    )
    assert response.status_code == 200
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == preassigned_experiment.id
    assert parsed.participant_id == "assigned_id"
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == "control"


def test_get_assignment_for_online_participant_with_apikey(
    xngin_session, testing_datasource
):
    """Test endpoint that gets an assignment for a participant via API key."""
    online_experiment, arms = insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type="online",
    )

    response = client.get(
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
    arms_map = {arm.id: arm.name for arm in arms}
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == arms_map[str(parsed.assignment.arm_id)]
    assert parsed.assignment.arm_name == "control"
    assert not parsed.assignment.strata

    # Test that we get the same assignment for the same participant.
    response2 = client.get(
        f"/experiments/{online_experiment.id!s}/assignments/1",
        headers={
            constants.HEADER_CONFIG_ID: testing_datasource.ds.id,
            constants.HEADER_API_KEY: testing_datasource.key,
        },
    )
    assert response2.status_code == 200
    assert response2.json() == response.json()

    # Make sure there's only one db entry.
    assignment = xngin_session.scalars(
        select(tables.ArmAssignment).where(
            tables.ArmAssignment.experiment_id == online_experiment.id
        )
    ).one()
    assert assignment.participant_id == "1"
    assert assignment.arm_id == str(parsed.assignment.arm_id)
