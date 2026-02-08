from datetime import UTC, datetime, timedelta

import pytest
from deepdiff import DeepDiff
from pydantic import TypeAdapter
from sqlalchemy import select

from xngin.apiserver import constants
from xngin.apiserver.routers.common_api_types import (
    ExperimentsType,
    Filter,
    GetParticipantAssignmentResponse,
    ListExperimentsResponse,
    OnlineFrequentistExperimentSpec,
    PreassignedFrequentistExperimentSpec,
)
from xngin.apiserver.routers.common_enums import ExperimentState, Relation, StopAssignmentReason
from xngin.apiserver.routers.experiments.test_experiments_common import (
    insert_experiment_and_arms,
    make_insertable_experiment,
)
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter


@pytest.mark.parametrize(
    "key,expected_status,expected_message",
    [
        ("", 400, "request header is required"),
        ("a", 400, "must start with"),
        ("xat_", 403, "invalid or does not have access"),
        ("xata", 403, "invalid or does not have access"),
        (None, 400, "request header is required"),
    ],
)
async def test_list_experiments_with_various_insufficient_headers(
    xngin_session, testing_datasource, client_v1, key, expected_status, expected_message
):
    """Tests that listing experiments tied to a db datasource requires an API key."""
    await insert_experiment_and_arms(xngin_session, testing_datasource.ds, state=ExperimentState.ASSIGNED)

    headers = {constants.HEADER_CONFIG_ID: testing_datasource.ds.id}
    if key is not None:
        headers[constants.HEADER_API_KEY] = key
    response = client_v1.get(
        "/experiments",
        headers=headers,
    )
    assert response.status_code == expected_status
    assert expected_message in response.json()["message"], response.content


async def test_list_experiments_with_api_key(xngin_session, testing_datasource, client_v1):
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
    expected_design_spec = await ExperimentStorageConverter(expected_experiment).get_design_spec()
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
    actual = PreassignedFrequentistExperimentSpec.model_validate(experiment_json["design_spec"])
    expected = await ExperimentStorageConverter(new_experiment).get_design_spec()
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
    experiment = await insert_experiment_and_arms(xngin_session, testing_datasource.ds, state=ExperimentState.COMMITTED)

    # Try to get testing_datasource's experiment from another datasource's key.
    response = client_v1.get(
        f"/experiments/{experiment.id!s}/assignments",
        headers={constants.HEADER_API_KEY: testing_datasource_with_user.key},
    )
    assert response.status_code == 404, response.json()
    assert response.json()["detail"] == "Experiment not found or not authorized."


async def test_get_assignment_preassigned(xngin_session, testing_datasource, client_v1):
    preassigned_experiment = await insert_experiment_and_arms(xngin_session, testing_datasource.ds)
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


async def test_get_assignment_online(xngin_session, testing_datasource, client_v1):
    """Test endpoint that gets an assignment for a participant via API key."""
    online_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=ExperimentsType.FREQ_ONLINE,
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
            select(tables.ArmAssignment).where(tables.ArmAssignment.experiment_id == online_experiment.id)
        )
    ).one()
    assert assignment.participant_id == "1"
    assert assignment.arm_id == str(parsed.assignment.arm_id)

    # Verify no update to experiment lifecycle info.
    assert assignment.experiment.stopped_assignments_at is None
    assert assignment.experiment.stopped_assignments_reason is None


async def test_get_assignment_mab_online(xngin_session, testing_datasource, client_v1):
    """Test endpoint that gets an assignment for a participant via API key."""
    online_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=ExperimentsType.MAB_ONLINE,
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
    assert not parsed.assignment.strata
    assert parsed.assignment.observed_at is None
    assert parsed.assignment.outcome is None
    assert parsed.assignment.context_values is None

    # Test that we get the same assignment for the same participant.
    response2 = client_v1.get(
        f"/experiments/{online_experiment.id!s}/assignments/1",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
    )
    assert response2.status_code == 200
    assert response2.json() == response.json()

    # Make sure there's only one db entry.
    assignment = (
        await xngin_session.scalars(select(tables.Draw).where(tables.Draw.experiment_id == online_experiment.id))
    ).one()
    assert assignment.participant_id == "1"
    assert assignment.arm_id == str(parsed.assignment.arm_id)

    # Verify no update to experiment lifecycle info.
    assert assignment.observed_at is None
    assert assignment.outcome is None
    assert assignment.context_vals is None
    assert assignment.current_mu is None
    assert assignment.current_covariance is None
    assert assignment.current_alpha is None
    assert assignment.current_beta is None
    assert assignment.experiment.stopped_assignments_at is None
    assert assignment.experiment.stopped_assignments_reason is None


async def test_get_assignment_online_dont_create(xngin_session, testing_datasource, client_v1):
    """Verify endpoint doesn't create an assignment when create_if_none=False."""
    online_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=ExperimentsType.FREQ_ONLINE,
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


async def test_get_assignment_online_past_end_date(xngin_session, testing_datasource, client_v1):
    """Verify endpoint doesn't create an assignment for an online experiment that has ended."""
    online_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=ExperimentsType.FREQ_ONLINE,
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


async def test_get_cmab_experiment_assignment_for_online_participant(xngin_session, testing_datasource, client_v1):
    """
    Test getting the assignment for a participant in a CMAB online experiment.
    """
    online_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=ExperimentsType.CMAB_ONLINE,
    )

    input_data = {
        "context_inputs": [
            {"context_id": context.id, "context_value": 1.0}
            for context in sorted(online_experiment.contexts, key=lambda c: c.id)
        ]
    }

    response = client_v1.post(
        f"/experiments/{online_experiment.id}/assignments/1/assign_cmab",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
        json=input_data,
    )
    assert response.status_code == 200
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == online_experiment.id
    assert parsed.participant_id == "1"
    arms_map = {arm.id: arm.name for arm in online_experiment.arms}
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == arms_map[str(parsed.assignment.arm_id)]
    assert not parsed.assignment.strata
    assert parsed.assignment.observed_at is None
    assert parsed.assignment.outcome is None
    assert parsed.assignment.context_values == [1.0, 1.0]

    # Test that we get the same assignment for the same participant.
    response2 = client_v1.get(
        f"/experiments/{online_experiment.id!s}/assignments/1",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
    )
    assert response2.status_code == 200
    assert response2.json() == response.json()

    # Make sure there's only one db entry.
    assignment = (
        await xngin_session.scalars(select(tables.Draw).where(tables.Draw.experiment_id == online_experiment.id))
    ).one()
    assert assignment.participant_id == "1"
    assert assignment.arm_id == str(parsed.assignment.arm_id)
    assert assignment.context_vals == [1.0, 1.0]

    # Verify no update to experiment lifecycle info.
    assert assignment.observed_at is None
    assert assignment.outcome is None
    assert assignment.experiment.stopped_assignments_at is None
    assert assignment.experiment.stopped_assignments_reason is None


async def test_get_cmab_experiment_assignment_for_online_participant_glific_unwrap(
    xngin_session, testing_datasource, client_v1
):
    """
    Verifies that a call resembling the webhook request from Glific can use the ?_unwrap=
    parameter to encapsulate an object satisfying the API's request body constraints in a
    message structure that is not fully under the client's control.

    This is the same as test_get_cmab_experiment_assignment_for_online_participant
    but with different HTTP client behavior.
    """
    online_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=ExperimentsType.CMAB_ONLINE,
    )

    input_data = {
        "context_inputs": [
            {"context_id": context.id, "context_value": 1.0}
            for context in sorted(online_experiment.contexts, key=lambda c: c.id)
        ]
    }

    fake_glific_request = {
        "@contact": "...",
        "@wa_group": "...",
        "organization_id": "1234",
        "@results": "...",
        "variables/custom": {  # requires JSONPointer escaping as: variables~1custom
            "controllable_field": input_data,
        },
    }
    response = client_v1.post(
        f"/experiments/{online_experiment.id}/assignments/1/assign_cmab?_unwrap=/variables~1custom/controllable_field",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
        json=fake_glific_request,
    )
    assert response.status_code == 200, response.content
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == online_experiment.id
    assert parsed.participant_id == "1"
    arms_map = {arm.id: arm.name for arm in online_experiment.arms}
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == arms_map[str(parsed.assignment.arm_id)]
    assert not parsed.assignment.strata
    assert parsed.assignment.observed_at is None
    assert parsed.assignment.outcome is None
    assert parsed.assignment.context_values == [1.0, 1.0]


async def test_assign_with_filters_wrong_experiment_type(xngin_session, testing_datasource, client_v1):
    """Test that assign_with_filters endpoint rejects non-FREQ_ONLINE experiments."""
    preassigned_exp = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
    )

    response = client_v1.post(
        f"/experiments/{preassigned_exp.id}/assignments/participant_1/assign_with_filters",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
        json={"properties": []},
    )
    assert response.status_code == 422
    detail_text = response.json().get("message")
    assert "is a freq_preassigned experiment, and not a freq_online experiment" in detail_text


async def test_assign_with_filters_participant_passes_filters(xngin_session, testing_datasource, client_v1):
    """Test that participant passing filters gets assigned."""
    # Create an experiment with a current_income filter: current_income BETWEEN 1000 and 5000
    experiment, design_spec = await make_insertable_experiment(
        datasource=testing_datasource.ds,
        state=ExperimentState.COMMITTED,
        experiment_type=ExperimentsType.FREQ_ONLINE,
    )
    design_spec = TypeAdapter(OnlineFrequentistExperimentSpec).validate_python(design_spec)
    design_spec.filters = [Filter(field_name="current_income", relation=Relation.BETWEEN, value=[1000, 5000])]
    # Get participants schema from datasource for data type resolution
    ds_config = testing_datasource.ds.get_config()
    participants_schema = ds_config.find_participants(design_spec.participant_type)
    experiment = (
        ExperimentStorageConverter(experiment).set_design_spec_fields(design_spec, participants_schema).get_experiment()
    )
    xngin_session.add(experiment)
    await xngin_session.commit()

    response = client_v1.post(
        f"/experiments/{experiment.id}/assignments/participant_1/assign_with_filters?random_state=42",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
        json={"properties": [{"field_name": "current_income", "value": 2500}]},
    )
    assert response.status_code == 200
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == experiment.id
    assert parsed.participant_id == "participant_1"
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name in {"control", "treatment"}
