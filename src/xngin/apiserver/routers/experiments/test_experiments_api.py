from datetime import UTC, datetime, timedelta
from http import HTTPStatus

import pytest
from deepdiff import DeepDiff
from pydantic import TypeAdapter
from sqlalchemy import select

from xngin.apiserver.routers.common_api_types import (
    CMABContextInputRequest,
    ExperimentsType,
    Filter,
    GetParticipantAssignmentResponse,
    OnlineAssignmentWithFiltersRequest,
    OnlineFrequentistExperimentSpec,
    ParticipantProperty,
    PreassignedFrequentistExperimentSpec,
    UpdateBanditArmOutcomeRequest,
)
from xngin.apiserver.routers.common_enums import ExperimentState, Relation, StopAssignmentReason
from xngin.apiserver.routers.experiments.test_experiments_common import (
    insert_experiment_and_arms,
    make_insertable_experiment,
)
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter
from xngin.apiserver.testing.experiments_api_client import ExperimentsAPIClientNotDefaultStatusError


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
    xngin_session, testing_datasource, eclient, key, expected_status, expected_message
):
    """Tests that listing experiments tied to a db datasource requires an API key."""
    await insert_experiment_and_arms(xngin_session, testing_datasource.ds, state=ExperimentState.ASSIGNED)

    kwargs = {"datasource_id": testing_datasource.ds.id}
    if key is not None:
        kwargs["api_key"] = key
    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.list_experiments(**kwargs)
    assert exc.value.result.status.value == expected_status
    assert expected_message in exc.value.result.data["message"], exc.value.result.response.content


async def test_list_experiments_with_api_key(xngin_session, testing_datasource, eclient):
    """Tests that listing experiments tied to a db datasource with an API key works."""
    expected_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, state=ExperimentState.ASSIGNED
    )
    experiments = eclient.list_experiments(api_key=testing_datasource.key, datasource_id=testing_datasource.ds.id).data
    assert len(experiments.items) == 1
    assert experiments.items[0].state == ExperimentState.ASSIGNED
    expected_design_spec = await ExperimentStorageConverter(expected_experiment).get_design_spec()
    diff = DeepDiff(expected_design_spec, experiments.items[0].design_spec)
    assert not diff, f"Objects differ:\n{diff.pretty()}"


async def test_get_experiment(xngin_session, testing_datasource, eclient):
    new_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, state=ExperimentState.DESIGNING
    )
    response = eclient.get_experiment(api_key=testing_datasource.key, experiment_id=new_experiment.id).data
    assert response.datasource_id == new_experiment.datasource_id
    assert response.state == new_experiment.state
    assert isinstance(response.design_spec, PreassignedFrequentistExperimentSpec)
    actual = response.design_spec
    expected = await ExperimentStorageConverter(new_experiment).get_design_spec()
    diff = DeepDiff(actual, expected)
    assert not diff, f"Objects differ:\n{diff.pretty()}"


def test_get_experiment_assignments_not_found(testing_datasource, eclient):
    """Test getting assignments for a non-existent experiment."""
    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.get_experiment_assignments(api_key=testing_datasource.key, experiment_id=tables.experiment_id_factory())
    assert exc.value.result.status == HTTPStatus.NOT_FOUND, exc.value.result.data
    assert exc.value.result.data["detail"] == "Experiment not found or not authorized."


async def test_get_experiment_assignments_wrong_datasource(
    xngin_session, testing_datasource, testing_datasource_with_user, eclient
):
    """Test getting assignments for an experiment from a different datasource."""
    # Create experiment in one datasource
    experiment = await insert_experiment_and_arms(xngin_session, testing_datasource.ds, state=ExperimentState.COMMITTED)

    # Try to get testing_datasource's experiment from another datasource's key.
    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.get_experiment_assignments(api_key=testing_datasource_with_user.key, experiment_id=experiment.id)
    assert exc.value.result.status == HTTPStatus.NOT_FOUND, exc.value.result.data
    assert exc.value.result.data["detail"] == "Experiment not found or not authorized."


async def test_get_experiment_assignments_success(xngin_session, testing_datasource, eclient):
    experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        state=ExperimentState.COMMITTED,
        experiment_type=ExperimentsType.FREQ_ONLINE,
    )
    first_assignment = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=experiment.id, participant_id="participant_1"
    ).data
    second_assignment = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=experiment.id, participant_id="participant_2"
    ).data

    assert first_assignment.assignment is not None
    assert second_assignment.assignment is not None

    parsed = eclient.get_experiment_assignments(api_key=testing_datasource.key, experiment_id=experiment.id).data
    assert parsed.experiment_id == experiment.id
    assert parsed.sample_size == 2
    assert parsed.balance_check is None
    assert {assignment.participant_id for assignment in parsed.assignments} == {"participant_1", "participant_2"}
    assert {assignment.arm_name for assignment in parsed.assignments}.issubset({"control", "treatment"})


async def test_get_experiment_assignments_as_csv_success(xngin_session, testing_datasource, eclient):
    experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        state=ExperimentState.COMMITTED,
        experiment_type=ExperimentsType.FREQ_ONLINE,
    )
    for i in range(10):
        assignment_response = eclient.get_assignment(
            api_key=testing_datasource.key,
            experiment_id=experiment.id,
            participant_id=f"participant_{i}",
        ).data
        assert assignment_response.assignment is not None

    response = eclient.client.get(
        f"/v1/experiments/{experiment.id}/assignments/csv",
        headers={"X-API-Key": testing_datasource.key},
    )

    assert response.status_code == HTTPStatus.OK, response.content
    assert response.headers["content-type"].startswith("text/csv")
    assert (
        response.headers["content-disposition"] == f'attachment; filename="experiment_{experiment.id}_assignments.csv"'
    )
    csv_lines = response.text.strip().splitlines()
    assert csv_lines[0] == "participant_id,arm_id,arm_name,created_at"
    assert len(csv_lines) == 11
    assert {line.split(",", 1)[0] for line in csv_lines[1:]} == {f"participant_{i}" for i in range(10)}
    assert all(any(arm.id in line for arm in experiment.arms) for line in csv_lines[1:])
    assert all(any(arm.name in line for arm in experiment.arms) for line in csv_lines[1:])


async def test_get_assignment_preassigned(xngin_session, testing_datasource, eclient):
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

    parsed = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=preassigned_experiment.id, participant_id="unassigned_id"
    ).data
    assert parsed.experiment_id == preassigned_experiment.id
    assert parsed.participant_id == "unassigned_id"
    assert parsed.assignment is None

    parsed = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=preassigned_experiment.id, participant_id="assigned_id"
    ).data
    assert parsed.experiment_id == preassigned_experiment.id
    assert parsed.participant_id == "assigned_id"
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == "control"


async def test_get_assignment_online(xngin_session, testing_datasource, eclient):
    """Test endpoint that gets an assignment for a participant via API key."""
    online_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, experiment_type=ExperimentsType.FREQ_ONLINE
    )

    parsed = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=online_experiment.id, participant_id="1"
    ).data
    assert parsed.experiment_id == online_experiment.id
    assert parsed.participant_id == "1"
    arms_map = {arm.id: arm.name for arm in online_experiment.arms}
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == arms_map[str(parsed.assignment.arm_id)]
    assert parsed.assignment.arm_name == "control"
    assert not parsed.assignment.strata

    # Test that we get the same assignment for the same participant.
    parsed2 = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=online_experiment.id, participant_id="1"
    ).data
    assert parsed2 == parsed

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


async def test_get_assignment_mab_online(xngin_session, testing_datasource, eclient):
    """Test endpoint that gets an assignment for a participant via API key."""
    online_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, experiment_type=ExperimentsType.MAB_ONLINE
    )

    parsed = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=online_experiment.id, participant_id="1"
    ).data
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
    parsed2 = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=online_experiment.id, participant_id="1"
    ).data
    assert parsed2 == parsed

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


async def test_get_assignment_online_dont_create(xngin_session, testing_datasource, eclient):
    """Verify endpoint doesn't create an assignment when create_if_none=False."""
    online_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, experiment_type=ExperimentsType.FREQ_ONLINE
    )

    parsed = eclient.get_assignment(
        api_key=testing_datasource.key, create_if_none=False, experiment_id=online_experiment.id, participant_id="1"
    ).data
    assert parsed.experiment_id == online_experiment.id
    assert parsed.participant_id == "1"
    assert parsed.assignment is None


async def test_get_assignment_online_past_end_date(xngin_session, testing_datasource, eclient):
    """Verify endpoint doesn't create an assignment for an online experiment that has ended."""
    online_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=ExperimentsType.FREQ_ONLINE,
        end_date=datetime.now(UTC) - timedelta(days=1),
    )

    parsed = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=online_experiment.id, participant_id="1"
    ).data
    assert parsed.experiment_id == online_experiment.id
    assert parsed.participant_id == "1"
    assert parsed.assignment is None

    # Verify side effect to experiment lifecycle info.
    await xngin_session.refresh(online_experiment)
    assert online_experiment.stopped_assignments_at is not None
    assert online_experiment.stopped_assignments_reason == StopAssignmentReason.END_DATE


async def test_get_cmab_experiment_assignment_for_online_participant(xngin_session, testing_datasource, eclient):
    """
    Test getting the assignment for a participant in a CMAB online experiment.
    """
    online_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, experiment_type=ExperimentsType.CMAB_ONLINE
    )

    context_inputs = [
        {"context_id": context.id, "context_value": 1.0}
        for context in sorted(online_experiment.contexts, key=lambda c: c.id)
    ]
    parsed = eclient.get_assignment_cmab(
        api_key=testing_datasource.key,
        body=CMABContextInputRequest(context_inputs=context_inputs),
        experiment_id=online_experiment.id,
        participant_id="1",
    ).data
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
    parsed2 = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=online_experiment.id, participant_id="1"
    ).data
    assert parsed2 == parsed

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
    xngin_session, testing_datasource, eclient
):
    """
    Verifies that a call resembling the webhook request from Glific can use the ?_unwrap=
    parameter to encapsulate an object satisfying the API's request body constraints in a
    message structure that is not fully under the client's control.

    This is the same as test_get_cmab_experiment_assignment_for_online_participant
    but with different HTTP client behavior.
    """
    online_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, experiment_type=ExperimentsType.CMAB_ONLINE
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
    response = eclient.client.post(
        f"/v1/experiments/{online_experiment.id}/assignments/1/assign_cmab"
        "?_unwrap=/variables~1custom/controllable_field",
        headers={"X-API-Key": testing_datasource.key},
        json=fake_glific_request,
    )
    assert response.status_code == HTTPStatus.OK, response.content
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


async def test_assign_cmab_wrong_experiment_type(xngin_session, testing_datasource, eclient):
    online_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=ExperimentsType.FREQ_ONLINE,
    )

    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.get_assignment_cmab(
            api_key=testing_datasource.key,
            body=CMABContextInputRequest(context_inputs=[]),
            experiment_id=online_experiment.id,
            participant_id="1",
        )
    assert exc.value.result.status == HTTPStatus.UNPROCESSABLE_CONTENT
    assert "is a freq_online experiment, and not a cmab_online experiment" in exc.value.result.data.detail[0].msg


async def test_assign_with_filters_wrong_experiment_type(xngin_session, testing_datasource, eclient):
    """Test that assign_with_filters endpoint rejects non-FREQ_ONLINE experiments."""
    preassigned_exp = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, experiment_type=ExperimentsType.FREQ_PREASSIGNED
    )

    # Expect a 422 because we are using the get_assignment_filtered endpoint incorrectly.
    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.get_assignment_filtered(
            api_key=testing_datasource.key,
            body=OnlineAssignmentWithFiltersRequest(properties=[]),
            experiment_id=preassigned_exp.id,
            participant_id="participant_1",
        )
    assert exc.value.result.status == HTTPStatus.UNPROCESSABLE_CONTENT
    assert "is a freq_preassigned experiment, and not a freq_online experiment" in exc.value.result.data.detail[0].msg


async def test_assign_with_filters_participant_passes_filters(xngin_session, testing_datasource, eclient):
    """Test that participant passing filters gets assigned."""
    # Create an experiment with a current_income filter: current_income BETWEEN 1000 and 5000
    experiment, design_spec = await make_insertable_experiment(
        datasource=testing_datasource.ds, state=ExperimentState.COMMITTED, experiment_type=ExperimentsType.FREQ_ONLINE
    )
    design_spec = TypeAdapter(OnlineFrequentistExperimentSpec).validate_python(design_spec)
    design_spec.filters = [Filter(field_name="current_income", relation=Relation.BETWEEN, value=[1000, 5000])]
    experiment = ExperimentStorageConverter(experiment).set_design_spec_fields(design_spec).get_experiment()
    xngin_session.add(experiment)
    await xngin_session.commit()

    parsed = eclient.get_assignment_filtered(
        api_key=testing_datasource.key,
        body=OnlineAssignmentWithFiltersRequest(properties=[{"field_name": "current_income", "value": 2500}]),
        experiment_id=experiment.id,
        participant_id="participant_1",
        random_state=42,
    ).data
    assert parsed.experiment_id == experiment.id
    assert parsed.participant_id == "participant_1"
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name in {"control", "treatment"}


async def test_assign_with_filters_ignores_missing_content_type_header(xngin_session, testing_datasource, eclient):
    experiment, design_spec = await make_insertable_experiment(
        datasource=testing_datasource.ds, state=ExperimentState.COMMITTED, experiment_type=ExperimentsType.FREQ_ONLINE
    )
    design_spec = TypeAdapter(OnlineFrequentistExperimentSpec).validate_python(design_spec)
    design_spec.filters = [Filter(field_name="current_income", relation=Relation.BETWEEN, value=[1000, 5000])]
    experiment = ExperimentStorageConverter(experiment).set_design_spec_fields(design_spec).get_experiment()
    xngin_session.add(experiment)
    await xngin_session.commit()

    response = eclient.client.post(
        f"/v1/experiments/{experiment.id}/assignments/participant_1/assign_with_filters?random_state=42",
        headers={"X-API-Key": testing_datasource.key},
        content=OnlineAssignmentWithFiltersRequest(
            properties=[ParticipantProperty(field_name="current_income", value=2500)]
        ).model_dump_json(),
    )
    assert "content-type" not in response.request.headers
    assert response.status_code == HTTPStatus.OK, response.content

    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == experiment.id
    assert parsed.participant_id == "participant_1"
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name in {"control", "treatment"}


async def test_get_assignment_preassigned_cache_headers(xngin_session, testing_datasource, eclient):
    """Test Cache-Control headers for preassigned experiments."""
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

    # No assignment = no cache header
    response = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=preassigned_experiment.id, participant_id="unassigned_id"
    )
    assert "Cache-Control" not in response.response.headers

    # With assignment = cache header
    response = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=preassigned_experiment.id, participant_id="assigned_id"
    )
    assert response.response.headers["Cache-Control"] == "private, max-age=3600"


async def test_get_assignment_online_cache_headers(xngin_session, testing_datasource, eclient):
    """Test Cache-Control headers for online experiments."""
    online_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, experiment_type=ExperimentsType.FREQ_ONLINE
    )

    # No assignment when create_if_none=false = no cache header
    response = eclient.get_assignment(
        api_key=testing_datasource.key, create_if_none=False, experiment_id=online_experiment.id, participant_id="1"
    )
    assert "Cache-Control" not in response.response.headers

    # Default max_age when assignment created
    response = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=online_experiment.id, participant_id="1"
    )
    assert response.response.headers["Cache-Control"] == "private, max-age=3600"

    # max_age=0 disables caching
    response = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=online_experiment.id, max_age=0, participant_id="1"
    )
    assert "Cache-Control" not in response.response.headers

    # Custom max_age
    response = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=online_experiment.id, max_age=100, participant_id="1"
    )
    assert response.response.headers["Cache-Control"] == "private, max-age=100"

    # Invalid max_age returns 422
    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.get_assignment(
            api_key=testing_datasource.key, experiment_id=online_experiment.id, max_age=-100, participant_id="1"
        )
    assert exc.value.result.status == HTTPStatus.UNPROCESSABLE_CONTENT


async def test_get_assignment_mab_cache_headers(xngin_session, testing_datasource, eclient):
    """Test Cache-Control headers for MAB experiments (only cached after outcome recorded)."""
    mab_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, experiment_type=ExperimentsType.MAB_ONLINE
    )

    # Get assignment - no cache header since no outcome yet
    response = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=mab_experiment.id, participant_id="1"
    )
    assert response.data.assignment is not None
    assert response.data.assignment.outcome is None
    assert "Cache-Control" not in response.response.headers

    # Record outcome
    _ = eclient.update_bandit_arm_with_participant_outcome(
        api_key=testing_datasource.key,
        body=UpdateBanditArmOutcomeRequest(outcome=1.0),
        experiment_id=mab_experiment.id,
        participant_id="1",
    )

    # Get assignment again - should have cache header now
    response = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=mab_experiment.id, participant_id="1"
    )
    assert response.data.assignment is not None
    assert response.data.assignment.outcome == 1.0
    assert response.response.headers["Cache-Control"] == "private, max-age=3600"


async def test_get_assignment_cmab_cache_headers(xngin_session, testing_datasource, eclient):
    """Test Cache-Control headers for CMAB experiments (only cached after outcome recorded)."""
    cmab_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, experiment_type=ExperimentsType.CMAB_ONLINE
    )

    context_inputs = [
        {"context_id": context.id, "context_value": 1.0}
        for context in sorted(cmab_experiment.contexts, key=lambda c: c.id)
    ]

    # Create assignment via CMAB endpoint
    _ = eclient.get_assignment_cmab(
        api_key=testing_datasource.key,
        body=CMABContextInputRequest(context_inputs=context_inputs),
        experiment_id=cmab_experiment.id,
        participant_id="1",
    )

    # Get assignment via GET - no cache header since no outcome yet
    response = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=cmab_experiment.id, participant_id="1"
    )
    assert response.data.assignment is not None
    assert response.data.assignment.outcome is None
    assert "Cache-Control" not in response.response.headers

    # Record outcome
    _ = eclient.update_bandit_arm_with_participant_outcome(
        api_key=testing_datasource.key,
        body=UpdateBanditArmOutcomeRequest(outcome=1.0),
        experiment_id=cmab_experiment.id,
        participant_id="1",
    )

    # Get assignment again - should have cache header now
    response = eclient.get_assignment(
        api_key=testing_datasource.key, experiment_id=cmab_experiment.id, participant_id="1"
    )
    assert response.data.assignment is not None
    assert response.data.assignment.outcome == 1.0
    assert response.response.headers["Cache-Control"] == "private, max-age=3600"
