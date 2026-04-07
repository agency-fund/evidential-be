from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest
from pydantic import TypeAdapter

from xngin.apiserver.routers.common_api_types import (
    Arm,
    ArmBandit,
    BaseDesignSpec,
    BaseFrequentistDesignSpec,
    CMABContextInputRequest,
    CMABExperimentSpec,
    Context,
    CreateExperimentRequest,
    DesignSpecMetricRequest,
    ExperimentConfig,
    ExperimentsType,
    Filter,
    GetParticipantAssignmentResponse,
    LikelihoodTypes,
    MABExperimentSpec,
    OnlineAssignmentWithFiltersRequest,
    OnlineFrequentistExperimentSpec,
    ParticipantProperty,
    PreassignedFrequentistExperimentSpec,
    PriorTypes,
    Stratum,
    UpdateBanditArmOutcomeRequest,
)
from xngin.apiserver.routers.common_enums import ContextType, ExperimentState, Relation, StopAssignmentReason
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.admin_api_client import AdminAPIClientHTTPValidationError
from xngin.apiserver.testing.experiments_api_client import ExperimentsAPIClientNotDefaultStatusError

if TYPE_CHECKING:
    from xngin.apiserver.testing.admin_api_client import AdminAPIClient
    from xngin.apiserver.testing.experiments_api_client import ExperimentsAPIClient


async def create_experiment(
    datasource_metadata,
    aclient: AdminAPIClient,
    *,
    experiment_type: ExperimentsType = ExperimentsType.FREQ_ONLINE,
    table_name: str | None = None,
    primary_key: str | None = None,
    end_date: datetime | None = None,
    filters: list[Filter] | None = None,
):
    """Creates an online experiment using the Admin API."""
    if experiment_type not in {
        ExperimentsType.FREQ_PREASSIGNED,
        ExperimentsType.FREQ_ONLINE,
        ExperimentsType.MAB_ONLINE,
        ExperimentsType.CMAB_ONLINE,
    }:
        raise ValueError(f"create_online_experiment only supports online experiment types, got {experiment_type}")

    if experiment_type in {ExperimentsType.FREQ_ONLINE, ExperimentsType.FREQ_PREASSIGNED}:
        # Set defaults for our frequentist experiments
        table_name = table_name or "dwh"
        primary_key = primary_key or "id"

    request = make_unvalidated_create_experiment_request(
        experiment_type=experiment_type,
        table_name=table_name,
        primary_key=primary_key,
        end_date=end_date,
        filters=filters,
    )
    request = CreateExperimentRequest.model_validate(request, from_attributes=True)
    if experiment_type == ExperimentsType.FREQ_PREASSIGNED:
        result = aclient.create_experiment(datasource_id=datasource_metadata.ds.id, body=request, desired_n=1)
    else:
        result = aclient.create_experiment(datasource_id=datasource_metadata.ds.id, body=request)
    created_experiment = result.data
    aclient.commit_experiment(datasource_id=datasource_metadata.ds.id, experiment_id=created_experiment.experiment_id)
    config = aclient.get_experiment_for_ui(
        datasource_id=datasource_metadata.ds.id,
        experiment_id=created_experiment.experiment_id,
    ).data.config
    return TypeAdapter(ExperimentConfig).validate_python(config)


async def create_preassigned_experiment(datasource_metadata, aclient: AdminAPIClient):
    """Creates a preassigned experiment using the Admin API."""
    return await create_experiment(datasource_metadata, aclient, experiment_type=ExperimentsType.FREQ_PREASSIGNED)


def make_unvalidated_create_experiment_request(
    *,
    experiment_type: ExperimentsType,
    table_name: str | None,
    primary_key: str | None,
    end_date: datetime | None = None,
    filters: list[Filter] | None = None,
) -> CreateExperimentRequest:
    end_date = end_date or datetime.now(UTC) + timedelta(days=1)
    filters = filters or []
    base_kwargs = BaseDesignSpec.model_construct(
        experiment_type=experiment_type,
        experiment_name="test experiment",
        description="test experiment",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=end_date,
        arms=[Arm(arm_name="overwritten1", arm_description=""), Arm(arm_name="overwritten2", arm_description="")],
    ).model_dump(exclude={"arms"})

    design_spec: BaseFrequentistDesignSpec | MABExperimentSpec | CMABExperimentSpec
    match experiment_type:
        case ExperimentsType.FREQ_PREASSIGNED:
            design_spec = PreassignedFrequentistExperimentSpec(
                **base_kwargs,
                arms=[
                    Arm(arm_name="control", arm_description="Control group"),
                    Arm(arm_name="treatment", arm_description="Treatment group"),
                ],
                metrics=[DesignSpecMetricRequest(field_name="is_onboarded", metric_pct_change=0.1)],
                strata=[Stratum(field_name="gender")],
                filters=filters,
            )
        case ExperimentsType.FREQ_ONLINE:
            design_spec = OnlineFrequentistExperimentSpec(
                **base_kwargs,
                arms=[
                    Arm(arm_name="control", arm_description="Control group"),
                    Arm(arm_name="treatment", arm_description="Treatment group"),
                ],
                metrics=[DesignSpecMetricRequest(field_name="is_onboarded", metric_pct_change=0.1)],
                strata=[Stratum(field_name="gender")],
                filters=filters,
            )
        case ExperimentsType.MAB_ONLINE:
            design_spec = MABExperimentSpec(
                **base_kwargs,
                arms=[
                    ArmBandit(arm_name="control", arm_description="Control group", mu_init=0.0, sigma_init=1.0),
                    ArmBandit(arm_name="treatment", arm_description="Treatment group", mu_init=0.0, sigma_init=1.0),
                ],
                prior_type=PriorTypes.NORMAL,
                reward_type=LikelihoodTypes.NORMAL,
                contexts=None,
            )
        case ExperimentsType.CMAB_ONLINE:
            design_spec = CMABExperimentSpec(
                **base_kwargs,
                arms=[
                    ArmBandit(arm_name="control", arm_description="Control group", mu_init=0.0, sigma_init=1.0),
                    ArmBandit(arm_name="treatment", arm_description="Treatment group", mu_init=0.0, sigma_init=1.0),
                ],
                prior_type=PriorTypes.NORMAL,
                reward_type=LikelihoodTypes.NORMAL,
                contexts=[
                    Context(context_name="c1", context_description="Context 1", value_type=ContextType.REAL_VALUED),
                    Context(context_name="c2", context_description="Context 2", value_type=ContextType.REAL_VALUED),
                ],
            )
        case _:
            raise ValueError(f"Invalid experiment type: {experiment_type}")

    return CreateExperimentRequest.model_construct(
        design_spec=design_spec,
        table_name=table_name,
        primary_key=primary_key,
    )


@pytest.mark.parametrize(
    "key,expected_status,expected_message",
    [
        ("", 400, "request header is required"),
        ("a", 400, "must start with"),
        ("xat_", 403, "invalid or does not have access"),
        ("xat_abc", 403, "invalid or does not have access"),
        ("xata", 403, "invalid or does not have access"),
        (None, 400, "request header is required"),
    ],
)
async def test_list_experiments_with_various_insufficient_headers(
    testing_datasource_with_user,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    key,
    expected_status,
    expected_message,
):
    """Tests that listing experiments tied to a db datasource requires an API key."""
    await create_experiment(testing_datasource_with_user, aclient)
    # Special case the absent header for compatibility with the generated client's argument types.
    if key is None:
        response = eclient.client.get("/v1/experiments", headers={"Datasource-ID": testing_datasource_with_user.ds.id})
        assert response.status_code == expected_status
        assert expected_message in response.json()["message"], response.content
        return

    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.list_experiments(api_key=key, datasource_id=testing_datasource_with_user.ds.id)
    assert exc.value.result.status.value == expected_status
    assert expected_message in exc.value.result.data["message"], exc.value.result.response.content


async def test_list_experiments_with_api_key(
    testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Tests that listing experiments tied to a db datasource with an API key works."""
    new_experiment = await create_experiment(testing_datasource_with_user, aclient)
    experiments = eclient.list_experiments(
        api_key=testing_datasource_with_user.key, datasource_id=testing_datasource_with_user.ds.id
    ).data
    assert len(experiments.items) == 1
    assert experiments.items[0].state == ExperimentState.COMMITTED
    assert new_experiment.design_spec == experiments.items[0].design_spec


async def test_get_experiment(testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient):
    new_experiment = await create_preassigned_experiment(testing_datasource_with_user, aclient)
    response = eclient.get_experiment(
        api_key=testing_datasource_with_user.key, experiment_id=new_experiment.experiment_id
    ).data
    assert response.datasource_id == testing_datasource_with_user.ds.id
    assert response.state == ExperimentState.COMMITTED
    assert isinstance(response.design_spec, PreassignedFrequentistExperimentSpec)
    assert response.design_spec == new_experiment.design_spec


@pytest.mark.parametrize(
    "experiment_type, table_name, primary_key, expected_status, expected_message",
    [
        (ExperimentsType.FREQ_PREASSIGNED, None, None, 422, "table_name and primary_key must be provided"),
        (ExperimentsType.FREQ_PREASSIGNED, "dwh", None, 422, "table_name and primary_key must be provided"),
        (ExperimentsType.FREQ_PREASSIGNED, None, "id", 422, "table_name and primary_key must be provided"),
        (ExperimentsType.FREQ_PREASSIGNED, "dwh", "id", 200, None),
        (ExperimentsType.FREQ_ONLINE, None, None, 422, "table_name and primary_key must be provided"),
        (ExperimentsType.FREQ_ONLINE, "dwh", None, 422, "table_name and primary_key must be provided"),
        (ExperimentsType.FREQ_ONLINE, None, "id", 422, "table_name and primary_key must be provided"),
        (ExperimentsType.FREQ_ONLINE, "dwh", "id", 200, None),
        (ExperimentsType.MAB_ONLINE, "dwh", "id", 422, "table_name and primary_key are not supported"),
        (ExperimentsType.MAB_ONLINE, None, "id", 422, "table_name and primary_key are not supported"),
        (ExperimentsType.CMAB_ONLINE, "dwh", "id", 422, "table_name and primary_key are not supported"),
        (ExperimentsType.CMAB_ONLINE, "dwh", None, 422, "table_name and primary_key are not supported"),
    ],
)
async def test_create_experiment_api_table_name_and_primary_key_presence(
    testing_datasource_with_user,
    aclient: AdminAPIClient,
    experiment_type: ExperimentsType,
    table_name: str | None,
    primary_key: str | None,
    expected_status: HTTPStatus,
    expected_message: str | None,
):
    request = make_unvalidated_create_experiment_request(
        experiment_type=experiment_type,
        table_name=table_name,
        primary_key=primary_key,
    )
    result = aclient.create_experiment(
        datasource_id=testing_datasource_with_user.ds.id,
        body=request,
        raise_if_not_default_status=False,
        desired_n=1,
    )

    assert result.status == expected_status, result.data
    if expected_message is not None:
        assert isinstance(result.data, AdminAPIClientHTTPValidationError)
        print(result.data.detail[0])
        assert expected_message in result.data.detail[0].msg


def test_get_experiment_assignments_not_found(testing_datasource, eclient: ExperimentsAPIClient):
    """Test getting assignments for a non-existent experiment."""
    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.get_experiment_assignments(api_key=testing_datasource.key, experiment_id=tables.experiment_id_factory())
    assert exc.value.result.status == HTTPStatus.NOT_FOUND, exc.value.result.data
    assert exc.value.result.data["detail"] == "Experiment not found or not authorized."


async def test_get_experiment_assignments_wrong_datasource(
    testing_datasource, testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Test getting assignments for an experiment from a different datasource."""
    experiment = await create_experiment(testing_datasource_with_user, aclient)

    # Try to get testing_datasource's experiment from another datasource's key.
    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.get_experiment_assignments(api_key=testing_datasource.key, experiment_id=experiment.experiment_id)
    assert exc.value.result.status == HTTPStatus.NOT_FOUND, exc.value.result.data
    assert exc.value.result.data["detail"] == "Experiment not found or not authorized."


async def test_get_experiment_assignments_success(
    testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    experiment = await create_experiment(testing_datasource_with_user, aclient)
    first_assignment = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=experiment.experiment_id,
        participant_id="participant_1",
    ).data
    second_assignment = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=experiment.experiment_id,
        participant_id="participant_2",
    ).data

    assert first_assignment.assignment is not None
    assert second_assignment.assignment is not None

    parsed = eclient.get_experiment_assignments(
        api_key=testing_datasource_with_user.key,
        experiment_id=experiment.experiment_id,
    ).data
    assert parsed.experiment_id == experiment.experiment_id
    assert parsed.sample_size == 2
    assert parsed.balance_check is None
    assert {assignment.participant_id for assignment in parsed.assignments} == {"participant_1", "participant_2"}
    assert {assignment.arm_name for assignment in parsed.assignments}.issubset({"control", "treatment"})


async def test_get_experiment_assignments_as_csv_success(
    testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    experiment = await create_experiment(testing_datasource_with_user, aclient)
    for i in range(10):
        assignment_response = eclient.get_assignment(
            api_key=testing_datasource_with_user.key,
            experiment_id=experiment.experiment_id,
            participant_id=f"participant_{i}",
        ).data
        assert assignment_response.assignment is not None

    response = eclient.client.get(
        f"/v1/experiments/{experiment.experiment_id}/assignments/csv",
        headers={"X-API-Key": testing_datasource_with_user.key},
    )

    assert response.status_code == HTTPStatus.OK, response.content
    assert response.headers["content-type"].startswith("text/csv")
    assert (
        response.headers["content-disposition"]
        == f'attachment; filename="experiment_{experiment.experiment_id}_assignments.csv"'
    )
    csv_lines = response.text.strip().splitlines()
    assert csv_lines[0] == "participant_id,arm_id,arm_name,created_at,gender"
    assert len(csv_lines) == 11
    assert {line.split(",", 1)[0] for line in csv_lines[1:]} == {f"participant_{i}" for i in range(10)}
    assert all(any(arm.arm_id in line for arm in experiment.design_spec.arms) for line in csv_lines[1:])
    assert all(any(arm.arm_name in line for arm in experiment.design_spec.arms) for line in csv_lines[1:])


async def test_get_assignment_preassigned(
    xngin_session,
    testing_datasource_with_user,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    preassigned_experiment = await create_preassigned_experiment(testing_datasource_with_user, aclient)
    assignment = tables.ArmAssignment(
        experiment_id=preassigned_experiment.experiment_id,
        participant_id="assigned_id",
        participant_type=preassigned_experiment.participant_type_deprecated,
        arm_id=preassigned_experiment.design_spec.arms[0].arm_id,
        strata=[],
    )
    xngin_session.add(assignment)
    await xngin_session.commit()

    parsed = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=preassigned_experiment.experiment_id,
        participant_id="unassigned_id",
    ).data
    assert parsed.experiment_id == preassigned_experiment.experiment_id
    assert parsed.participant_id == "unassigned_id"
    assert parsed.assignment is None

    parsed = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=preassigned_experiment.experiment_id,
        participant_id="assigned_id",
    ).data
    assert parsed.experiment_id == preassigned_experiment.experiment_id
    assert parsed.participant_id == "assigned_id"
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == "control"


async def test_get_assignment_online(
    testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Test endpoint that gets an assignment for a participant via API key."""
    online_experiment = await create_experiment(testing_datasource_with_user, aclient)

    parsed = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    ).data
    assert parsed.experiment_id == online_experiment.experiment_id
    assert parsed.participant_id == "1"
    arms_map = {arm.arm_id: arm.arm_name for arm in online_experiment.design_spec.arms}
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == arms_map[str(parsed.assignment.arm_id)]
    assert parsed.assignment.arm_name == "control"
    assert not parsed.assignment.strata

    # Test that we get the same assignment for the same participant.
    parsed2 = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    ).data
    assert parsed2 == parsed

    assignments = eclient.get_experiment_assignments(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
    ).data
    assert assignments.sample_size == 1
    assert len(assignments.assignments) == 1
    assert assignments.assignments[0].participant_id == "1"
    assert str(assignments.assignments[0].arm_id) == str(parsed.assignment.arm_id)

    experiment = eclient.get_experiment(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
    ).data
    assert experiment.stopped_assignments_at is None
    assert experiment.stopped_assignments_reason is None


async def test_get_assignment_mab_online(
    testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Test endpoint that gets an assignment for a participant via API key."""
    online_experiment = await create_experiment(
        testing_datasource_with_user, aclient, experiment_type=ExperimentsType.MAB_ONLINE
    )

    parsed = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    ).data
    assert parsed.experiment_id == online_experiment.experiment_id
    assert parsed.participant_id == "1"
    arms_map = {arm.arm_id: arm.arm_name for arm in online_experiment.design_spec.arms}
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == arms_map[str(parsed.assignment.arm_id)]
    assert not parsed.assignment.strata
    assert parsed.assignment.observed_at is None
    assert parsed.assignment.outcome is None
    assert parsed.assignment.context_values is None

    # Test that we get the same assignment for the same participant.
    parsed2 = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    ).data
    assert parsed2 == parsed

    assignments = eclient.get_experiment_assignments(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
    ).data
    assert assignments.sample_size == 1
    assert len(assignments.assignments) == 1
    assert assignments.assignments[0].participant_id == "1"
    assert str(assignments.assignments[0].arm_id) == str(parsed.assignment.arm_id)
    assert assignments.assignments[0].observed_at is None
    assert assignments.assignments[0].outcome is None
    assert assignments.assignments[0].context_values is None

    experiment = eclient.get_experiment(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
    ).data
    assert experiment.stopped_assignments_at is None
    assert experiment.stopped_assignments_reason is None


async def test_get_assignment_online_dont_create(
    testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Verify endpoint doesn't create an assignment when create_if_none=False."""
    online_experiment = await create_experiment(testing_datasource_with_user, aclient)

    parsed = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        create_if_none=False,
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    ).data
    assert parsed.experiment_id == online_experiment.experiment_id
    assert parsed.participant_id == "1"
    assert parsed.assignment is None


async def test_get_assignment_online_past_end_date(
    testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Verify endpoint doesn't create an assignment for an online experiment that has ended."""
    online_experiment = await create_experiment(
        testing_datasource_with_user,
        aclient,
        end_date=datetime.now(UTC) - timedelta(days=1),
    )

    parsed = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    ).data
    assert parsed.experiment_id == online_experiment.experiment_id
    assert parsed.participant_id == "1"
    assert parsed.assignment is None

    experiment = eclient.get_experiment(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
    ).data
    assert experiment.stopped_assignments_at is not None
    assert experiment.stopped_assignments_reason == StopAssignmentReason.END_DATE


async def test_get_cmab_experiment_assignment_for_online_participant(
    testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """
    Test getting the assignment for a participant in a CMAB online experiment.
    """
    online_experiment = await create_experiment(
        testing_datasource_with_user, aclient, experiment_type=ExperimentsType.CMAB_ONLINE
    )

    context_inputs = [
        {"context_id": context.context_id, "context_value": 1.0}
        for context in sorted(online_experiment.design_spec.contexts, key=lambda c: c.context_id)
    ]
    parsed = eclient.get_assignment_cmab(
        api_key=testing_datasource_with_user.key,
        body=CMABContextInputRequest(context_inputs=context_inputs),
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    ).data
    assert parsed.experiment_id == online_experiment.experiment_id
    assert parsed.participant_id == "1"
    arms_map = {arm.arm_id: arm.arm_name for arm in online_experiment.design_spec.arms}
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == arms_map[str(parsed.assignment.arm_id)]
    assert not parsed.assignment.strata
    assert parsed.assignment.observed_at is None
    assert parsed.assignment.outcome is None
    assert parsed.assignment.context_values == [1.0, 1.0]

    # Test that we get the same assignment for the same participant.
    parsed2 = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    ).data
    assert parsed2 == parsed

    assignments = eclient.get_experiment_assignments(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
    ).data
    assert assignments.sample_size == 1
    assert len(assignments.assignments) == 1
    assert assignments.assignments[0].participant_id == "1"
    assert str(assignments.assignments[0].arm_id) == str(parsed.assignment.arm_id)
    assert assignments.assignments[0].context_values == [1.0, 1.0]

    experiment = eclient.get_experiment(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
    ).data
    assert experiment.stopped_assignments_at is None
    assert experiment.stopped_assignments_reason is None


async def test_get_cmab_experiment_assignment_for_online_participant_glific_unwrap(
    testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """
    Verifies that a call resembling the webhook request from Glific can use the ?_unwrap=
    parameter to encapsulate an object satisfying the API's request body constraints in a
    message structure that is not fully under the client's control.

    This is the same as test_get_cmab_experiment_assignment_for_online_participant
    but with different HTTP client behavior.
    """
    online_experiment = await create_experiment(
        testing_datasource_with_user, aclient, experiment_type=ExperimentsType.CMAB_ONLINE
    )

    input_data = {
        "context_inputs": [
            {"context_id": context.context_id, "context_value": 1.0}
            for context in sorted(online_experiment.design_spec.contexts, key=lambda c: c.context_id)
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
        f"/v1/experiments/{online_experiment.experiment_id}/assignments/1/assign_cmab"
        "?_unwrap=/variables~1custom/controllable_field",
        headers={"X-API-Key": testing_datasource_with_user.key},
        json=fake_glific_request,
    )
    assert response.status_code == HTTPStatus.OK, response.content
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == online_experiment.experiment_id
    assert parsed.participant_id == "1"
    arms_map = {arm.arm_id: arm.arm_name for arm in online_experiment.design_spec.arms}
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == arms_map[str(parsed.assignment.arm_id)]
    assert not parsed.assignment.strata
    assert parsed.assignment.observed_at is None
    assert parsed.assignment.outcome is None
    assert parsed.assignment.context_values == [1.0, 1.0]


async def test_assign_cmab_wrong_experiment_type(
    testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    online_experiment = await create_experiment(testing_datasource_with_user, aclient)

    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.get_assignment_cmab(
            api_key=testing_datasource_with_user.key,
            body=CMABContextInputRequest(context_inputs=[]),
            experiment_id=online_experiment.experiment_id,
            participant_id="1",
        )
    assert exc.value.result.status == HTTPStatus.UNPROCESSABLE_CONTENT
    assert "is a freq_online experiment, and not a cmab_online experiment" in exc.value.result.data.detail[0].msg


async def test_assign_with_filters_wrong_experiment_type(
    xngin_session,
    testing_datasource_with_user,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    """Test that assign_with_filters endpoint rejects non-FREQ_ONLINE experiments."""
    preassigned_exp = await create_preassigned_experiment(testing_datasource_with_user, aclient)

    # Expect a 422 because we are using the get_assignment_filtered endpoint incorrectly.
    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.get_assignment_filtered(
            api_key=testing_datasource_with_user.key,
            body=OnlineAssignmentWithFiltersRequest(properties=[]),
            experiment_id=preassigned_exp.experiment_id,
            participant_id="participant_1",
        )
    assert exc.value.result.status == HTTPStatus.UNPROCESSABLE_CONTENT
    assert "is a freq_preassigned experiment, and not a freq_online experiment" in exc.value.result.data.detail[0].msg


async def test_assign_with_filters_participant_passes_filters(
    testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Test that participant passing filters gets assigned."""
    experiment = await create_experiment(
        testing_datasource_with_user,
        aclient,
        filters=[Filter(field_name="current_income", relation=Relation.BETWEEN, value=[1000, 5000])],
    )

    parsed = eclient.get_assignment_filtered(
        api_key=testing_datasource_with_user.key,
        body=OnlineAssignmentWithFiltersRequest(properties=[{"field_name": "current_income", "value": 2500}]),
        experiment_id=experiment.experiment_id,
        participant_id="participant_1",
        random_state=42,
    ).data
    assert parsed.experiment_id == experiment.experiment_id
    assert parsed.participant_id == "participant_1"
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name in {"control", "treatment"}


async def test_assign_with_filters_ignores_missing_content_type_header(
    testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    experiment = await create_experiment(
        testing_datasource_with_user,
        aclient,
        filters=[Filter(field_name="current_income", relation=Relation.BETWEEN, value=[1000, 5000])],
    )

    response = eclient.client.post(
        f"/v1/experiments/{experiment.experiment_id}/assignments/participant_1/assign_with_filters?random_state=42",
        headers={"X-API-Key": testing_datasource_with_user.key},
        content=OnlineAssignmentWithFiltersRequest(
            properties=[ParticipantProperty(field_name="current_income", value=2500)]
        ).model_dump_json(),
    )
    assert "content-type" not in response.request.headers
    assert response.status_code == HTTPStatus.OK, response.content

    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == experiment.experiment_id
    assert parsed.participant_id == "participant_1"
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name in {"control", "treatment"}


async def test_get_assignment_preassigned_cache_headers(
    xngin_session,
    testing_datasource_with_user,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    """Test Cache-Control headers for preassigned experiments."""
    preassigned_experiment = await create_preassigned_experiment(testing_datasource_with_user, aclient)
    assignment = tables.ArmAssignment(
        experiment_id=preassigned_experiment.experiment_id,
        participant_id="assigned_id",
        participant_type=preassigned_experiment.participant_type_deprecated,
        arm_id=preassigned_experiment.design_spec.arms[0].arm_id,
        strata=[],
    )
    xngin_session.add(assignment)
    await xngin_session.commit()

    # No assignment = no cache header
    response = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=preassigned_experiment.experiment_id,
        participant_id="unassigned_id",
    )
    assert "Cache-Control" not in response.response.headers

    # With assignment = cache header
    response = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=preassigned_experiment.experiment_id,
        participant_id="assigned_id",
    )
    assert response.response.headers["Cache-Control"] == "private, max-age=3600"


async def test_get_assignment_online_cache_headers(
    testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Test Cache-Control headers for online experiments."""
    online_experiment = await create_experiment(testing_datasource_with_user, aclient)

    # No assignment when create_if_none=false = no cache header
    response = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        create_if_none=False,
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    )
    assert "Cache-Control" not in response.response.headers

    # Default max_age when assignment created
    response = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    )
    assert response.response.headers["Cache-Control"] == "private, max-age=3600"

    # max_age=0 disables caching
    response = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
        max_age=0,
        participant_id="1",
    )
    assert "Cache-Control" not in response.response.headers

    # Custom max_age
    response = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=online_experiment.experiment_id,
        max_age=100,
        participant_id="1",
    )
    assert response.response.headers["Cache-Control"] == "private, max-age=100"

    # Invalid max_age returns 422
    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.get_assignment(
            api_key=testing_datasource_with_user.key,
            experiment_id=online_experiment.experiment_id,
            max_age=-100,
            participant_id="1",
        )
    assert exc.value.result.status == HTTPStatus.UNPROCESSABLE_CONTENT


async def test_get_assignment_mab_cache_headers(
    testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Test Cache-Control headers for MAB experiments (only cached after outcome recorded)."""
    mab_experiment = await create_experiment(
        testing_datasource_with_user, aclient, experiment_type=ExperimentsType.MAB_ONLINE
    )

    # Get assignment - no cache header since no outcome yet
    response = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=mab_experiment.experiment_id,
        participant_id="1",
    )
    assert response.data.assignment is not None
    assert response.data.assignment.outcome is None
    assert "Cache-Control" not in response.response.headers

    # Record outcome
    _ = eclient.update_bandit_arm_with_participant_outcome(
        api_key=testing_datasource_with_user.key,
        body=UpdateBanditArmOutcomeRequest(outcome=1.0),
        experiment_id=mab_experiment.experiment_id,
        participant_id="1",
    )

    # Get assignment again - should have cache header now
    response = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=mab_experiment.experiment_id,
        participant_id="1",
    )
    assert response.data.assignment is not None
    assert response.data.assignment.outcome == 1.0
    assert response.response.headers["Cache-Control"] == "private, max-age=3600"


async def test_get_assignment_cmab_cache_headers(
    testing_datasource_with_user, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Test Cache-Control headers for CMAB experiments (only cached after outcome recorded)."""
    cmab_experiment = await create_experiment(
        testing_datasource_with_user, aclient, experiment_type=ExperimentsType.CMAB_ONLINE
    )

    context_inputs = [
        {"context_id": context.context_id, "context_value": 1.0}
        for context in sorted(cmab_experiment.design_spec.contexts, key=lambda c: c.context_id)
    ]

    # Create assignment via CMAB endpoint
    _ = eclient.get_assignment_cmab(
        api_key=testing_datasource_with_user.key,
        body=CMABContextInputRequest(context_inputs=context_inputs),
        experiment_id=cmab_experiment.experiment_id,
        participant_id="1",
    )

    # Get assignment via GET - no cache header since no outcome yet
    response = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=cmab_experiment.experiment_id,
        participant_id="1",
    )
    assert response.data.assignment is not None
    assert response.data.assignment.outcome is None
    assert "Cache-Control" not in response.response.headers

    # Record outcome
    _ = eclient.update_bandit_arm_with_participant_outcome(
        api_key=testing_datasource_with_user.key,
        body=UpdateBanditArmOutcomeRequest(outcome=1.0),
        experiment_id=cmab_experiment.experiment_id,
        participant_id="1",
    )

    # Get assignment again - should have cache header now
    response = eclient.get_assignment(
        api_key=testing_datasource_with_user.key,
        experiment_id=cmab_experiment.experiment_id,
        participant_id="1",
    )
    assert response.data.assignment is not None
    assert response.data.assignment.outcome == 1.0
    assert response.response.headers["Cache-Control"] == "private, max-age=3600"
