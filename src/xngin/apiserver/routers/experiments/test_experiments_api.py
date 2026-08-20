import csv
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from io import StringIO
from typing import TYPE_CHECKING

import pytest
from pydantic import TypeAdapter

from xngin.apiserver.conftest import DatasourceMetadata
from xngin.apiserver.routers.common_api_types import (
    AnyFrequentistDesignSpec,
    Arm,
    ArmBandit,
    BaseDesignSpec,
    CMABContextInputRequest,
    CMABExperimentSpec,
    Context,
    ContextInput,
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
from xngin.apiserver.routers.common_enums import (
    ContextType,
    ExperimentState,
    Relation,
    StopAssignmentReason,
    UpdateTypeNormal,
)
from xngin.apiserver.routers.experiments.test_experiments_common import make_create_online_bandit_experiment_request
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.admin_api_client import AdminAPIClientHTTPValidationError
from xngin.apiserver.testing.experiments_api_client import ExperimentsAPIClientNotDefaultStatusError
from xngin.stats.bandit_sampling import update_arm

if TYPE_CHECKING:
    from xngin.apiserver.testing.admin_api_client import AdminAPIClient
    from xngin.apiserver.testing.experiments_api_client import ExperimentsAPIClient


async def create_experiment(
    datasource_metadata: DatasourceMetadata,
    aclient: AdminAPIClient,
    *,
    experiment_type: ExperimentsType = ExperimentsType.FREQ_ONLINE,
    table_name: str | None = None,
    primary_key: str | None = None,
    end_date: datetime | None = None,
    filters: list[Filter] | None = None,
    desired_n: int | None = None,
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
        desired_n=desired_n or 1,
    )
    request = CreateExperimentRequest.model_validate(request, from_attributes=True)
    result = aclient.create_experiment(datasource_id=datasource_metadata.datasource_id, body=request)
    created_experiment = result.data
    aclient.commit_experiment(
        datasource_id=datasource_metadata.datasource_id, experiment_id=created_experiment.experiment_id
    )
    config = aclient.get_experiment_for_ui(
        datasource_id=datasource_metadata.datasource_id,
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
    desired_n: int | None = None,
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

    design_spec: AnyFrequentistDesignSpec | MABExperimentSpec | CMABExperimentSpec
    match experiment_type:
        case ExperimentsType.FREQ_PREASSIGNED:
            props: dict = {
                **base_kwargs,
                "arms": [
                    Arm(arm_name="control", arm_description="Control group"),
                    Arm(arm_name="treatment", arm_description="Treatment group"),
                ],
                "metrics": [DesignSpecMetricRequest(field_name="is_onboarded", metric_pct_change=0.1)],
                "strata": [Stratum(field_name="gender")],
                "filters": filters,
                "desired_n": desired_n,
            }
            if table_name is not None:
                props["table_name"] = table_name
            if primary_key is not None:
                props["primary_key"] = primary_key
            design_spec = PreassignedFrequentistExperimentSpec.model_construct(**props)
        case ExperimentsType.FREQ_ONLINE:
            props_online: dict = {
                **base_kwargs,
                "arms": [
                    Arm(arm_name="control", arm_description="Control group"),
                    Arm(arm_name="treatment", arm_description="Treatment group"),
                ],
                "metrics": [DesignSpecMetricRequest(field_name="is_onboarded", metric_pct_change=0.1)],
                "strata": [Stratum(field_name="gender")],
                "filters": filters,
                "desired_n": desired_n,
            }
            if table_name is not None:
                props_online["table_name"] = table_name
            if primary_key is not None:
                props_online["primary_key"] = primary_key
            design_spec = OnlineFrequentistExperimentSpec.model_construct(**props_online)
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

    return CreateExperimentRequest.model_construct(design_spec=design_spec)


@pytest.mark.parametrize(
    ("key", "expected_status", "expected_message"),
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
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    key,
    expected_status,
    expected_message,
):
    """Tests that listing experiments tied to a db datasource requires an API key."""
    await create_experiment(testing_datasource, aclient)
    # Special case the absent header for compatibility with the generated client's argument types.
    if key is None:
        response = eclient.client.get("/v1/experiments", headers={"Datasource-ID": testing_datasource.datasource_id})
        assert response.status_code == expected_status
        assert expected_message in response.json()["message"], response.content
        return

    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.list_experiments(api_key=key, datasource_id=testing_datasource.datasource_id)
    assert exc.value.result.status.value == expected_status
    assert expected_message in exc.value.result.data["message"], exc.value.result.response.content


async def test_list_experiments_with_api_key(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Tests that listing experiments tied to a db datasource with an API key works."""
    new_experiment = await create_experiment(testing_datasource, aclient)
    experiments = eclient.list_experiments(
        api_key=testing_datasource.key, datasource_id=testing_datasource.datasource_id
    ).data
    assert len(experiments.items) == 1
    assert experiments.items[0].state == ExperimentState.COMMITTED
    assert new_experiment.design_spec == experiments.items[0].design_spec


async def test_get_experiment(testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient):
    new_experiment = await create_preassigned_experiment(testing_datasource, aclient)
    response = eclient.get_experiment(api_key=testing_datasource.key, experiment_id=new_experiment.experiment_id).data
    assert response.datasource_id == testing_datasource.datasource_id
    assert response.state == ExperimentState.COMMITTED
    assert isinstance(response.design_spec, PreassignedFrequentistExperimentSpec)
    assert response.design_spec == new_experiment.design_spec


@pytest.mark.parametrize(
    ("experiment_type", "table_name", "primary_key", "expected_status", "expected_in_loc"),
    [
        (ExperimentsType.FREQ_PREASSIGNED, None, None, 422, "table_name"),
        (ExperimentsType.FREQ_PREASSIGNED, "dwh", None, 422, "primary_key"),
        (ExperimentsType.FREQ_PREASSIGNED, None, "id", 422, "table_name"),
        (ExperimentsType.FREQ_PREASSIGNED, "dwh", "id", 200, None),
        (ExperimentsType.FREQ_ONLINE, None, None, 422, "table_name"),
        (ExperimentsType.FREQ_ONLINE, "dwh", None, 422, "primary_key"),
        (ExperimentsType.FREQ_ONLINE, None, "id", 422, "table_name"),
        (ExperimentsType.FREQ_ONLINE, "dwh", "id", 200, None),
    ],
)
# model_construct bypasses validation intentionally, so suppress expected PydanticSerializationUnexpectedValue noise
@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning:pydantic")
async def test_create_experiment_api_table_name_and_primary_key_in_design_spec(
    testing_datasource,
    aclient: AdminAPIClient,
    experiment_type: ExperimentsType,
    table_name: str | None,
    primary_key: str | None,
    expected_status: HTTPStatus,
    expected_in_loc: str | None,
):
    request = make_unvalidated_create_experiment_request(
        experiment_type=experiment_type,
        table_name=table_name,
        primary_key=primary_key,
        desired_n=1,
    )
    result = aclient.create_experiment(
        datasource_id=testing_datasource.datasource_id,
        body=request,
        raise_if_not_default_status=False,
    )

    assert result.status == expected_status, result.data
    if expected_in_loc is not None:
        assert isinstance(result.data, AdminAPIClientHTTPValidationError)
        msgs = [d.msg for d in result.data.detail]
        assert any("Field required" in m for m in msgs), msgs
        locs = [loc for d in result.data.detail for loc in d.loc]
        assert any(expected_in_loc in str(loc) for loc in locs), locs


def test_get_experiment_assignments_not_found(testing_datasource, eclient: ExperimentsAPIClient):
    """Test getting assignments for a non-existent experiment."""
    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.get_experiment_assignments(api_key=testing_datasource.key, experiment_id=tables.experiment_id_factory())
    assert exc.value.result.status == HTTPStatus.NOT_FOUND, exc.value.result.data
    assert exc.value.result.data["detail"] == "Experiment not found or not authorized."


async def test_get_experiment_assignments_wrong_datasource(
    testing_datasource, testing_datasource_other, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Test getting assignments for an experiment from a different datasource."""
    experiment = await create_experiment(testing_datasource_other, aclient)

    # Try to get testing_datasource's experiment from another datasource's key.
    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.get_experiment_assignments(api_key=testing_datasource.key, experiment_id=experiment.experiment_id)
    assert exc.value.result.status == HTTPStatus.NOT_FOUND, exc.value.result.data
    assert exc.value.result.data["detail"] == "Experiment not found or not authorized."


async def test_get_experiment_assignments_success(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    experiment = await create_experiment(testing_datasource, aclient)
    first_assignment = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=experiment.experiment_id,
        participant_id="participant_1",
    ).data
    second_assignment = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=experiment.experiment_id,
        participant_id="participant_2",
    ).data

    assert first_assignment.assignment is not None
    assert second_assignment.assignment is not None

    parsed = eclient.get_experiment_assignments(
        api_key=testing_datasource.key,
        experiment_id=experiment.experiment_id,
    ).data
    assert parsed.experiment_id == experiment.experiment_id
    assert parsed.sample_size == 2
    assert parsed.balance_check is None
    assert {assignment.participant_id for assignment in parsed.assignments} == {"participant_1", "participant_2"}
    assert {assignment.arm_name for assignment in parsed.assignments}.issubset({"control", "treatment"})


async def test_get_experiment_assignments_streams_preassigned_assignments(
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment = await create_experiment(
        testing_datasource,
        aclient,
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        desired_n=2,
    )

    arms_by_id = {arm.arm_id: arm.arm_name for arm in experiment.design_spec.arms}

    data = eclient.get_experiment_assignments(
        api_key=testing_datasource.key,
        experiment_id=experiment.experiment_id,
    ).data
    assert data.experiment_id == experiment.experiment_id
    assert data.sample_size == 2
    assert data.balance_check is None

    assert len(data.assignments) == 2
    for assignment in data.assignments:
        participant_assignment = eclient.get_assignment(
            api_key=testing_datasource.key,
            experiment_id=experiment.experiment_id,
            participant_id=assignment.participant_id,
            create_if_none=False,
        ).data.assignment
        assert participant_assignment is not None
        assert assignment.arm_name == arms_by_id[assignment.arm_id]
        assert assignment.arm_id == participant_assignment.arm_id
        assert assignment.arm_name == participant_assignment.arm_name
        assert assignment.created_at == participant_assignment.created_at
        assert assignment.strata is not None and len(assignment.strata) == 1
        assert assignment.strata[0].field_name == "gender"
        assert assignment.strata[0].strata_value is not None
        assert assignment.observed_at is None
        assert assignment.outcome is None
        assert assignment.context_values is None


async def test_both_get_experiment_assignments_endpoints_have_matching_strata_ordering(
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    request = make_unvalidated_create_experiment_request(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        table_name="dwh",
        primary_key="id",
        desired_n=2,
    )
    request.design_spec = PreassignedFrequentistExperimentSpec(
        **request.design_spec.model_dump(exclude={"strata"}),
        strata=[Stratum(field_name="ethnicity"), Stratum(field_name="gender")],
    )
    created_experiment = aclient.create_experiment(
        datasource_id=testing_datasource.datasource_id,
        body=request,
    ).data
    aclient.commit_experiment(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=created_experiment.experiment_id,
    )

    data = eclient.get_experiment_assignments(
        api_key=testing_datasource.key,
        experiment_id=created_experiment.experiment_id,
    ).data
    csv_response = eclient.client.get(
        f"/v1/experiments/{created_experiment.experiment_id}/assignments/csv",
        headers={"X-API-Key": testing_datasource.key},
    )
    assert csv_response.status_code == HTTPStatus.OK, csv_response.content
    assert "cluster_key" not in csv_response.text.splitlines()[0].split(",")

    csv_rows = {row["participant_id"]: row for row in csv.DictReader(StringIO(csv_response.text))}
    assert set(csv_rows) == {assignment.participant_id for assignment in data.assignments}

    for assignment in data.assignments:
        csv_row = csv_rows[assignment.participant_id]
        assert assignment.strata is not None
        assert [stratum.field_name for stratum in assignment.strata] == ["ethnicity", "gender"]
        assert {stratum.field_name: stratum.strata_value for stratum in assignment.strata} == {
            "ethnicity": csv_row["ethnicity"] or None,
            "gender": csv_row["gender"] or None,
        }

    json_response = eclient.client.get(
        f"/v1/experiments/{created_experiment.experiment_id}/assignments",
        headers={"X-API-Key": testing_datasource.key},
    )
    assert json_response.status_code == HTTPStatus.OK, json_response.content
    assert all("cluster_key" not in assignment for assignment in json_response.json()["assignments"])


async def test_cluster_key_exports_with_preassigned_assignments(
    testing_datasource,
    use_deterministic_random,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    request = CreateExperimentRequest(
        design_spec=PreassignedFrequentistExperimentSpec(
            experiment_type=ExperimentsType.FREQ_PREASSIGNED,
            experiment_name="cluster-key export",
            description="Verify cluster keys are exported with assignments (CSV, bulk JSON, and individual GETs).",
            table_name="clustered_dwh",
            primary_key="participant_id",
            cluster_key="cluster_powerlaw",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime.now(UTC) + timedelta(days=1),
            arms=[Arm(arm_name="control", arm_description="C"), Arm(arm_name="treatment", arm_description="T")],
            metrics=[DesignSpecMetricRequest(field_name="test_score", metric_pct_change=0.1)],
            strata=[],
            filters=[],
            desired_n=24,
            desired_n_clusters=24,
        ),
        webhooks=[],
    )
    datasource_id = testing_datasource.datasource_id
    created_experiment = aclient.create_experiment(datasource_id=datasource_id, body=request, random_state=42).data
    experiment_id = created_experiment.experiment_id
    aclient.commit_experiment(datasource_id=datasource_id, experiment_id=experiment_id)

    eclient_headers = {"X-API-Key": testing_datasource.key}
    json_response = eclient.client.get(f"/v1/experiments/{experiment_id}/assignments", headers=eclient_headers)
    assert json_response.status_code == HTTPStatus.OK, json_response.content
    json_assignments = json_response.json()["assignments"]
    assert all("cluster_key" in assignment for assignment in json_assignments)
    assert all(assignment["cluster_key"] is not None for assignment in json_assignments)
    assert len({assignment["cluster_key"] for assignment in json_assignments}) == 24

    csv_response = eclient.client.get(f"/v1/experiments/{experiment_id}/assignments/csv", headers=eclient_headers)
    assert csv_response.status_code == HTTPStatus.OK, csv_response.content

    # Verify the rows returned via CSV match the JSON export.
    csv_reader = csv.DictReader(StringIO(csv_response.text))
    assert csv_reader.fieldnames == ["participant_id", "cluster_key", "arm_id", "arm_name", "created_at"]
    csv_rows = {row["participant_id"]: row for row in csv_reader}
    assert len(csv_rows) == len(json_assignments)
    for assignment in json_assignments:
        csv_row = csv_rows[assignment["participant_id"]]
        assert csv_row["cluster_key"] == assignment["cluster_key"]
        assert csv_row["arm_id"] == assignment["arm_id"]
        assert csv_row["arm_name"] == assignment["arm_name"]
        json_created_at = datetime.fromisoformat(assignment["created_at"])
        assert csv_row["created_at"] == json_created_at.isoformat(timespec="seconds").replace("+00:00", "Z")

        # And verify the single-assignment GET is consistent with the bulk exports.
        single_assignment = eclient.get_assignment(
            api_key=testing_datasource.key,
            experiment_id=experiment_id,
            participant_id=assignment["participant_id"],
            create_if_none=False,
        ).data.assignment
        assert single_assignment is not None
        assert single_assignment.cluster_key == assignment["cluster_key"]


async def test_get_experiment_assignments_streams_bandit_assignments(
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment = await create_experiment(
        testing_datasource,
        aclient,
        experiment_type=ExperimentsType.MAB_ONLINE,
    )

    observed_at = datetime.now(UTC)
    arms_by_id = {arm.arm_id: arm.arm_name for arm in experiment.design_spec.arms}
    first_assignment = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=experiment.experiment_id,
        participant_id="p1",
    ).data.assignment
    second_assignment = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=experiment.experiment_id,
        participant_id="p2",
    ).data.assignment
    assert first_assignment is not None
    assert second_assignment is not None

    eclient.update_bandit_arm_with_participant_outcome(
        api_key=testing_datasource.key,
        body=UpdateBanditArmOutcomeRequest(outcome=1.5),
        experiment_id=experiment.experiment_id,
        participant_id="p1",
    )
    updated_first_assignment = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=experiment.experiment_id,
        participant_id="p1",
        create_if_none=False,
    ).data.assignment
    updated_second_assignment = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=experiment.experiment_id,
        participant_id="p2",
        create_if_none=False,
    ).data.assignment

    data = eclient.get_experiment_assignments(
        api_key=testing_datasource.key,
        experiment_id=experiment.experiment_id,
    ).data
    assert data.experiment_id == experiment.experiment_id
    assert data.sample_size == 2
    assert data.balance_check is None

    assignments_by_participant_id = {assignment.participant_id: assignment for assignment in data.assignments}
    assert set(assignments_by_participant_id) == {"p1", "p2"}

    p1 = assignments_by_participant_id["p1"]
    assert updated_first_assignment is not None
    assert updated_first_assignment.model_copy(update={"strata": None, "autofailed_outcome": None}) == p1
    assert p1.arm_name == arms_by_id[p1.arm_id]
    assert p1.created_at is not None
    assert p1.observed_at is not None
    assert p1.observed_at >= observed_at.replace(microsecond=0)
    assert p1.outcome == 1.5
    assert p1.context_values is None
    assert p1.strata is None

    p2 = assignments_by_participant_id["p2"]
    assert updated_second_assignment is not None
    assert updated_second_assignment.model_copy(update={"strata": None, "autofailed_outcome": None}) == p2
    assert p2.arm_name == arms_by_id[p2.arm_id]
    assert p2.created_at is not None
    assert p2.observed_at is None
    assert p2.outcome is None
    assert p2.context_values is None
    assert p2.strata is None


async def test_get_experiment_assignments_streams_cmab_context_values(
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment = await create_experiment(
        testing_datasource,
        aclient,
        experiment_type=ExperimentsType.CMAB_ONLINE,
    )

    # Create two draws
    _ = eclient.get_assignment_cmab(
        api_key=testing_datasource.key,
        body=CMABContextInputRequest(
            context_inputs=[
                ContextInput(context_id=context.context_id, context_value=1.0)
                for context in sorted(experiment.design_spec.contexts, key=lambda c: c.context_id)
            ]
        ),
        experiment_id=experiment.experiment_id,
        participant_id="p1",
    ).data.assignment
    _ = eclient.get_assignment_cmab(
        api_key=testing_datasource.key,
        body=CMABContextInputRequest(
            context_inputs=[
                ContextInput(context_id=context.context_id, context_value=2.0)
                for context in sorted(experiment.design_spec.contexts, key=lambda c: c.context_id)
            ]
        ),
        experiment_id=experiment.experiment_id,
        participant_id="p2",
    ).data.assignment

    # One participant has an outcome
    eclient.update_bandit_arm_with_participant_outcome(
        api_key=testing_datasource.key,
        body=UpdateBanditArmOutcomeRequest(outcome=1.5),
        experiment_id=experiment.experiment_id,
        participant_id="p1",
    )

    # Fetch assignments from single-assignment endpoint for later comparison.
    first_assignment = eclient.get_assignment_cmab(
        api_key=testing_datasource.key,
        body=CMABContextInputRequest(context_inputs=None),
        experiment_id=experiment.experiment_id,
        participant_id="p1",
        create_if_none=False,
    ).data.assignment
    second_assignment = eclient.get_assignment_cmab(
        api_key=testing_datasource.key,
        body=CMABContextInputRequest(context_inputs=None),
        experiment_id=experiment.experiment_id,
        participant_id="p2",
        create_if_none=False,
    ).data.assignment
    assert first_assignment is not None
    assert second_assignment is not None

    # Get all assignments
    data = eclient.get_experiment_assignments(
        api_key=testing_datasource.key,
        experiment_id=experiment.experiment_id,
    ).data
    assert data.experiment_id == experiment.experiment_id
    assert data.sample_size == 2
    assert data.balance_check is None

    assignments_by_participant_id = {assignment.participant_id: assignment for assignment in data.assignments}
    assert set(assignments_by_participant_id) == {"p1", "p2"}

    p1 = assignments_by_participant_id["p1"]
    assert p1.created_at is not None
    assert p1.observed_at is not None
    assert p1.outcome == 1.5
    assert p1.strata is None
    assert p1.context_values == [1.0, 1.0]
    assert first_assignment.model_copy(update={"strata": None, "autofailed_outcome": None}) == p1

    p2 = assignments_by_participant_id["p2"]
    assert p2.created_at is not None
    assert p2.observed_at is None
    assert p2.outcome is None
    assert p2.strata is None
    assert p2.context_values == [2.0, 2.0]
    assert second_assignment.model_copy(update={"strata": None}) == p2


async def test_get_experiment_assignments_as_csv_success(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    experiment = await create_experiment(testing_datasource, aclient)
    for i in range(10):
        assignment_response = eclient.get_assignment(
            api_key=testing_datasource.key,
            experiment_id=experiment.experiment_id,
            participant_id=f"participant_{i}",
        ).data
        assert assignment_response.assignment is not None

    response = eclient.client.get(
        f"/v1/experiments/{experiment.experiment_id}/assignments/csv",
        headers={"X-API-Key": testing_datasource.key},
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
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    preassigned_experiment = await create_preassigned_experiment(testing_datasource, aclient)
    assigned = eclient.get_experiment_assignments(
        api_key=testing_datasource.key,
        experiment_id=preassigned_experiment.experiment_id,
    ).data.assignments[0]

    parsed = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=preassigned_experiment.experiment_id,
        participant_id="unassigned_id",
    ).data
    assert parsed.experiment_id == preassigned_experiment.experiment_id
    assert parsed.participant_id == "unassigned_id"
    assert parsed.assignment is None

    parsed = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=preassigned_experiment.experiment_id,
        participant_id=assigned.participant_id,
    ).data
    assert parsed.experiment_id == preassigned_experiment.experiment_id
    assert parsed.participant_id == assigned.participant_id
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == assigned.arm_name


async def test_get_assignment_online(testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient):
    """Test endpoint that gets an assignment for a participant via API key."""
    online_experiment = await create_experiment(testing_datasource, aclient)

    parsed = eclient.get_assignment(
        api_key=testing_datasource.key,
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
        api_key=testing_datasource.key,
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    ).data
    assert parsed2 == parsed

    assignments = eclient.get_experiment_assignments(
        api_key=testing_datasource.key,
        experiment_id=online_experiment.experiment_id,
    ).data
    assert assignments.sample_size == 1
    assert len(assignments.assignments) == 1
    assert assignments.assignments[0].participant_id == "1"
    assert str(assignments.assignments[0].arm_id) == str(parsed.assignment.arm_id)

    experiment = eclient.get_experiment(
        api_key=testing_datasource.key,
        experiment_id=online_experiment.experiment_id,
    ).data
    assert experiment.stopped_assignments_at is None
    assert experiment.stopped_assignments_reason is None


async def test_get_assignment_mab_online(testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient):
    """Test endpoint that gets an assignment for a participant via API key."""
    online_experiment = await create_experiment(testing_datasource, aclient, experiment_type=ExperimentsType.MAB_ONLINE)

    parsed = eclient.get_assignment(
        api_key=testing_datasource.key,
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
        api_key=testing_datasource.key,
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    ).data
    assert parsed2 == parsed

    assignments = eclient.get_experiment_assignments(
        api_key=testing_datasource.key,
        experiment_id=online_experiment.experiment_id,
    ).data
    assert assignments.sample_size == 1
    assert len(assignments.assignments) == 1
    assert assignments.assignments[0].participant_id == "1"
    assert str(assignments.assignments[0].arm_id) == str(parsed.assignment.arm_id)
    assert assignments.assignments[0].strata is None
    assert assignments.assignments[0].observed_at is None
    assert assignments.assignments[0].outcome is None
    assert assignments.assignments[0].context_values is None

    experiment = eclient.get_experiment(
        api_key=testing_datasource.key,
        experiment_id=online_experiment.experiment_id,
    ).data
    assert experiment.stopped_assignments_at is None
    assert experiment.stopped_assignments_reason is None


async def test_get_assignment_online_dont_create(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Verify endpoint doesn't create an assignment when create_if_none=False."""
    online_experiment = await create_experiment(testing_datasource, aclient)

    parsed = eclient.get_assignment(
        api_key=testing_datasource.key,
        create_if_none=False,
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    ).data
    assert parsed.experiment_id == online_experiment.experiment_id
    assert parsed.participant_id == "1"
    assert parsed.assignment is None


async def test_get_assignment_online_past_end_date(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Verify endpoint doesn't create an assignment for an online experiment that has ended."""
    online_experiment = await create_experiment(
        testing_datasource,
        aclient,
        end_date=datetime.now(UTC) - timedelta(days=1),
    )

    parsed = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    ).data
    assert parsed.experiment_id == online_experiment.experiment_id
    assert parsed.participant_id == "1"
    assert parsed.assignment is None

    experiment = eclient.get_experiment(
        api_key=testing_datasource.key,
        experiment_id=online_experiment.experiment_id,
    ).data
    assert experiment.stopped_assignments_at is not None
    assert experiment.stopped_assignments_reason == StopAssignmentReason.END_DATE


async def test_get_cmab_experiment_assignment_for_online_participant(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """
    Test getting the assignment for a participant in a CMAB online experiment.
    """
    online_experiment = await create_experiment(
        testing_datasource, aclient, experiment_type=ExperimentsType.CMAB_ONLINE
    )

    context_inputs = [
        ContextInput(context_id=context.context_id, context_value=1.0)
        for context in sorted(online_experiment.design_spec.contexts, key=lambda c: c.context_id)
    ]
    parsed = eclient.get_assignment_cmab(
        api_key=testing_datasource.key,
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
    parsed2 = eclient.get_assignment_cmab(
        api_key=testing_datasource.key,
        body=CMABContextInputRequest(context_inputs=None),
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    ).data
    assert parsed2 == parsed

    assignments = eclient.get_experiment_assignments(
        api_key=testing_datasource.key,
        experiment_id=online_experiment.experiment_id,
    ).data
    assert assignments.sample_size == 1
    assert len(assignments.assignments) == 1
    assert assignments.assignments[0].participant_id == "1"
    assert str(assignments.assignments[0].arm_id) == str(parsed.assignment.arm_id)
    assert assignments.assignments[0].strata is None
    assert assignments.assignments[0].context_values == [1.0, 1.0]

    experiment = eclient.get_experiment(
        api_key=testing_datasource.key,
        experiment_id=online_experiment.experiment_id,
    ).data
    assert experiment.stopped_assignments_at is None
    assert experiment.stopped_assignments_reason is None


async def test_get_cmab_experiment_assignment_for_online_participant_glific_unwrap(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """
    Verifies that a call resembling the webhook request from Glific can use the ?_unwrap=
    parameter to encapsulate an object satisfying the API's request body constraints in a
    message structure that is not fully under the client's control.

    This is the same as test_get_cmab_experiment_assignment_for_online_participant
    but with different HTTP client behavior.
    """
    online_experiment = await create_experiment(
        testing_datasource, aclient, experiment_type=ExperimentsType.CMAB_ONLINE
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
        headers={"X-API-Key": testing_datasource.key},
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
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    online_experiment = await create_experiment(testing_datasource, aclient)

    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.get_assignment_cmab(
            api_key=testing_datasource.key,
            body=CMABContextInputRequest(context_inputs=[]),
            experiment_id=online_experiment.experiment_id,
            participant_id="1",
        )
    assert exc.value.result.status == HTTPStatus.UNPROCESSABLE_CONTENT
    assert "is a freq_online experiment, and not a cmab_online experiment" in exc.value.result.data.detail[0].msg


async def test_assign_with_filters_wrong_experiment_type(
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    """Test that assign_with_filters endpoint rejects non-FREQ_ONLINE experiments."""
    preassigned_exp = await create_preassigned_experiment(testing_datasource, aclient)

    # Expect a 422 because we are using the get_assignment_filtered endpoint incorrectly.
    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.get_assignment_filtered(
            api_key=testing_datasource.key,
            body=OnlineAssignmentWithFiltersRequest(properties=[]),
            experiment_id=preassigned_exp.experiment_id,
            participant_id="participant_1",
        )
    assert exc.value.result.status == HTTPStatus.UNPROCESSABLE_CONTENT
    assert "is a freq_preassigned experiment, and not a freq_online experiment" in exc.value.result.data.detail[0].msg


async def test_assign_with_filters_participant_passes_filters(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Test that participant passing filters gets assigned."""
    experiment = await create_experiment(
        testing_datasource,
        aclient,
        filters=[Filter(field_name="current_income", relation=Relation.BETWEEN, value=[1000, 5000])],
    )

    parsed = eclient.get_assignment_filtered(
        api_key=testing_datasource.key,
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
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    experiment = await create_experiment(
        testing_datasource,
        aclient,
        filters=[Filter(field_name="current_income", relation=Relation.BETWEEN, value=[1000, 5000])],
    )

    response = eclient.client.post(
        f"/v1/experiments/{experiment.experiment_id}/assignments/participant_1/assign_with_filters?random_state=42",
        headers={"X-API-Key": testing_datasource.key},
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
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    """Test Cache-Control headers for preassigned experiments."""
    preassigned_experiment = await create_preassigned_experiment(testing_datasource, aclient)
    assigned = eclient.get_experiment_assignments(
        api_key=testing_datasource.key,
        experiment_id=preassigned_experiment.experiment_id,
    ).data.assignments[0]

    # No assignment = no cache header
    response = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=preassigned_experiment.experiment_id,
        participant_id="unassigned_id",
    )
    assert "Cache-Control" not in response.response.headers

    # With assignment = cache header
    response = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=preassigned_experiment.experiment_id,
        participant_id=assigned.participant_id,
    )
    assert response.response.headers["Cache-Control"] == "private, max-age=3600"


async def test_get_assignment_online_cache_headers(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Test Cache-Control headers for online experiments."""
    online_experiment = await create_experiment(testing_datasource, aclient)

    # No assignment when create_if_none=false = no cache header
    response = eclient.get_assignment(
        api_key=testing_datasource.key,
        create_if_none=False,
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    )
    assert "Cache-Control" not in response.response.headers

    # Default max_age when assignment created
    response = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=online_experiment.experiment_id,
        participant_id="1",
    )
    assert response.response.headers["Cache-Control"] == "private, max-age=3600"

    # max_age=0 disables caching
    response = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=online_experiment.experiment_id,
        max_age=0,
        participant_id="1",
    )
    assert "Cache-Control" not in response.response.headers

    # Custom max_age
    response = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=online_experiment.experiment_id,
        max_age=100,
        participant_id="1",
    )
    assert response.response.headers["Cache-Control"] == "private, max-age=100"

    # Invalid max_age returns 422
    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.get_assignment(
            api_key=testing_datasource.key,
            experiment_id=online_experiment.experiment_id,
            max_age=-100,
            participant_id="1",
        )
    assert exc.value.result.status == HTTPStatus.UNPROCESSABLE_CONTENT


@pytest.mark.parametrize("experiment_type", [ExperimentsType.MAB_ONLINE, ExperimentsType.CMAB_ONLINE])
async def test_get_assignment_bandit_cache_headers(
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    experiment_type: ExperimentsType,
):
    """Bandit assignments are cached only after an outcome is recorded."""
    experiment = await create_experiment(testing_datasource, aclient, experiment_type=experiment_type)

    if experiment_type == ExperimentsType.CMAB_ONLINE:
        context_inputs = [
            {"context_id": context.context_id, "context_value": 1.0}
            for context in sorted(experiment.design_spec.contexts, key=lambda context: context.context_id)
        ]
        eclient.get_assignment_cmab(
            api_key=testing_datasource.key,
            body=CMABContextInputRequest(context_inputs=context_inputs),
            experiment_id=experiment.experiment_id,
            participant_id="1",
        )

    # Get assignment - no cache header since no outcome yet
    response = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=experiment.experiment_id,
        participant_id="1",
    )
    assert response.data.assignment is not None
    assert response.data.assignment.outcome is None
    assert "Cache-Control" not in response.response.headers

    # Record outcome
    _ = eclient.update_bandit_arm_with_participant_outcome(
        api_key=testing_datasource.key,
        body=UpdateBanditArmOutcomeRequest(outcome=1.0),
        experiment_id=experiment.experiment_id,
        participant_id="1",
    )

    # Get assignment again - should have cache header now
    response = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=experiment.experiment_id,
        participant_id="1",
    )
    assert response.data.assignment is not None
    assert response.data.assignment.outcome == 1.0
    assert response.response.headers["Cache-Control"] == "private, max-age=3600"


@pytest.mark.parametrize(
    ("experiment_type", "prior_type", "reward_type"),
    [
        (ExperimentsType.MAB_ONLINE, PriorTypes.BETA, LikelihoodTypes.BERNOULLI),
        (ExperimentsType.MAB_ONLINE, PriorTypes.NORMAL, LikelihoodTypes.NORMAL),
        (ExperimentsType.MAB_ONLINE, PriorTypes.NORMAL, LikelihoodTypes.BERNOULLI),
        (ExperimentsType.CMAB_ONLINE, PriorTypes.NORMAL, LikelihoodTypes.NORMAL),
        (ExperimentsType.CMAB_ONLINE, PriorTypes.NORMAL, LikelihoodTypes.BERNOULLI),
    ],
)
async def test_update_bandit_arm_with_outcome(
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    experiment_type: ExperimentsType,
    prior_type: PriorTypes,
    reward_type: LikelihoodTypes,
):
    """Record an outcome and verify the updated draw and arm through API responses."""
    arms = [
        ArmBandit(
            arm_name="control",
            arm_description="Control group",
            **(
                {"alpha_init": 1.0, "beta_init": 1.0}
                if prior_type == PriorTypes.BETA
                else {"mu_init": 0.0, "sigma_init": 1.0}
            ),
        ),
        ArmBandit(
            arm_name="treatment",
            arm_description="Treatment group",
            **(
                {"alpha_init": 1.0, "beta_init": 1.0}
                if prior_type == PriorTypes.BETA
                else {"mu_init": 0.0, "sigma_init": 1.0}
            ),
        ),
    ]
    design_spec: MABExperimentSpec | CMABExperimentSpec
    if experiment_type == ExperimentsType.CMAB_ONLINE:
        design_spec = CMABExperimentSpec(
            experiment_type=experiment_type,
            experiment_name="API outcome update",
            description="API outcome update",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime.now(UTC) + timedelta(days=1),
            arms=arms,
            prior_type=prior_type,
            reward_type=reward_type,
            contexts=[
                Context(context_name="c1", context_description="Context 1", value_type=ContextType.REAL_VALUED),
                Context(context_name="c2", context_description="Context 2", value_type=ContextType.REAL_VALUED),
            ],
        )
    else:
        design_spec = MABExperimentSpec(
            experiment_type=experiment_type,
            experiment_name="API outcome update",
            description="API outcome update",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime.now(UTC) + timedelta(days=1),
            arms=arms,
            prior_type=prior_type,
            reward_type=reward_type,
            contexts=None,
        )
    experiment_id = aclient.create_experiment(
        datasource_id=testing_datasource.datasource_id,
        body=CreateExperimentRequest(design_spec=design_spec),
    ).data.experiment_id
    aclient.commit_experiment(datasource_id=testing_datasource.datasource_id, experiment_id=experiment_id)

    participant_id = "test_id"
    committed_design_spec = aclient.get_experiment_for_ui(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=experiment_id,
    ).data.config.design_spec
    assert isinstance(committed_design_spec, MABExperimentSpec | CMABExperimentSpec)
    initial_arms = {arm.arm_id: arm for arm in committed_design_spec.arms}

    if experiment_type == ExperimentsType.CMAB_ONLINE:
        assignment = eclient.get_assignment_cmab(
            api_key=testing_datasource.key,
            body=CMABContextInputRequest(
                context_inputs=(
                    [
                        ContextInput(context_id=context.context_id or "", context_value=1.0)
                        for context in committed_design_spec.contexts or []
                    ]
                    if isinstance(committed_design_spec, CMABExperimentSpec)
                    else []
                )
            ),
            experiment_id=experiment_id,
            participant_id=participant_id,
        ).data.assignment
    else:
        assignment = eclient.get_assignment(
            api_key=testing_datasource.key,
            experiment_id=experiment_id,
            participant_id=participant_id,
        ).data.assignment
    assert assignment is not None

    updated_arm = eclient.update_bandit_arm_with_participant_outcome(
        api_key=testing_datasource.key,
        body=UpdateBanditArmOutcomeRequest(outcome=1.0),
        experiment_id=experiment_id,
        participant_id=participant_id,
    ).data

    updated_design_spec = aclient.get_experiment_for_ui(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=experiment_id,
    ).data.config.design_spec
    assert isinstance(updated_design_spec, MABExperimentSpec | CMABExperimentSpec)
    arms_on_experiment_after = {arm.arm_id: arm for arm in updated_design_spec.arms}

    # Verify arm state has been updated as expected.
    updated_arm_after = arms_on_experiment_after[assignment.arm_id]
    initial_assigned_arm = initial_arms[assignment.arm_id]
    if experiment_type == ExperimentsType.MAB_ONLINE:
        match prior_type:
            case PriorTypes.BETA:
                assert initial_assigned_arm.alpha is not None
                assert initial_assigned_arm.beta is not None
                assert updated_arm_after.alpha == initial_assigned_arm.alpha + 1
                assert updated_arm_after.beta == initial_assigned_arm.beta

                assert updated_arm_after.mu == initial_assigned_arm.mu
                assert updated_arm_after.covariance == initial_assigned_arm.covariance
            case PriorTypes.NORMAL:
                assert updated_arm_after.alpha == initial_assigned_arm.alpha
                assert updated_arm_after.beta == initial_assigned_arm.beta

                assert updated_arm_after.mu == updated_arm.mu
                assert updated_arm_after.covariance == updated_arm.covariance

                if reward_type == LikelihoodTypes.NORMAL:
                    assert updated_arm_after.mu != initial_assigned_arm.mu
                    assert updated_arm_after.covariance != initial_assigned_arm.covariance

                    assert updated_arm_after.mu == pytest.approx([0.5])
                    assert updated_arm_after.covariance is not None
                    assert updated_arm_after.covariance[0] == pytest.approx([0.5])
                else:
                    assert updated_arm_after.mu != initial_assigned_arm.mu
                    assert updated_arm_after.covariance != initial_assigned_arm.covariance
                    # Deferring further assertions: see test_normal_prior_binary_reward_fits_each_outcome_exactly_once
    else:
        assert experiment_type == ExperimentsType.CMAB_ONLINE
        assert prior_type == PriorTypes.NORMAL
        assert updated_arm_after.alpha == initial_assigned_arm.alpha
        assert updated_arm_after.beta == initial_assigned_arm.beta
        assert updated_arm_after.mu == updated_arm.mu
        assert updated_arm_after.covariance == updated_arm.covariance

        if reward_type == LikelihoodTypes.NORMAL:
            assert updated_arm_after.mu == pytest.approx([1 / 3, 1 / 3])
            assert updated_arm_after.covariance is not None
            expected_covariance = [[1 / 3, 0.0], [0.0, 1 / 3]]
            for actual_row, expected_row in zip(
                updated_arm_after.covariance,
                expected_covariance,
                strict=True,
            ):
                assert actual_row == pytest.approx(expected_row)
        else:
            assert updated_arm_after.mu != initial_assigned_arm.mu
            assert updated_arm_after.covariance != initial_assigned_arm.covariance
            # Deferring further assertions: see test_normal_prior_binary_reward_fits_each_outcome_exactly_once

    if experiment_type == ExperimentsType.CMAB_ONLINE:
        updated_assignment = eclient.get_assignment_cmab(
            api_key=testing_datasource.key,
            body=CMABContextInputRequest(context_inputs=None),
            experiment_id=experiment_id,
            participant_id=participant_id,
            create_if_none=False,
        ).data.assignment
    else:
        updated_assignment = eclient.get_assignment(
            api_key=testing_datasource.key,
            experiment_id=experiment_id,
            participant_id=participant_id,
            create_if_none=False,
        ).data.assignment
    assert updated_assignment is not None
    assert updated_assignment.outcome == 1.0
    assert updated_assignment.observed_at is not None
    assert updated_assignment.arm_id == assignment.arm_id
    if experiment_type == ExperimentsType.CMAB_ONLINE:
        assert updated_assignment.context_values == [1.0, 1.0]

    stored_design_spec = aclient.get_experiment_for_ui(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=experiment_id,
    ).data.config.design_spec
    stored_arm = next(arm for arm in stored_design_spec.arms if arm.arm_id == assignment.arm_id)
    assert stored_arm == updated_arm

    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.update_bandit_arm_with_participant_outcome(
            api_key=testing_datasource.key,
            body=UpdateBanditArmOutcomeRequest(outcome=1.0),
            experiment_id=experiment_id,
            participant_id="missing",
        )
    assert exc.value.result.status == HTTPStatus.UNPROCESSABLE_CONTENT

    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.update_bandit_arm_with_participant_outcome(
            api_key=testing_datasource.key,
            body=UpdateBanditArmOutcomeRequest(outcome=1.0),
            experiment_id=experiment_id,
            participant_id=participant_id,
        )
    assert exc.value.result.status == HTTPStatus.UNPROCESSABLE_CONTENT


async def test_create_mab_dwh_bool_target_with_normal_reward_returns_422(testing_datasource, aclient: AdminAPIClient):
    """The API rejects an incompatible Normal reward for a boolean MAB-DWH target."""
    request = make_create_online_bandit_experiment_request(
        experiment_type=ExperimentsType.MAB_ONLINE_DWH,
        prior_type=PriorTypes.NORMAL,
        reward_type=LikelihoodTypes.NORMAL,
        target_field_name="is_onboarded",
    )
    result = aclient.create_experiment(
        datasource_id=testing_datasource.datasource_id,
        body=request,
        random_state=42,
        raise_if_not_default_status=False,
    )
    assert result.status == HTTPStatus.UNPROCESSABLE_CONTENT
    assert "only compatible with reward_type 'binary'" in str(result.data)


async def test_update_bandit_arm_with_outcome_mab_dwh_numeric_target_accepts_any_float(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """A MAB-DWH numeric target accepts arbitrary floats through the API."""
    request = make_create_online_bandit_experiment_request(
        experiment_type=ExperimentsType.MAB_ONLINE_DWH,
        prior_type=PriorTypes.NORMAL,
        reward_type=LikelihoodTypes.NORMAL,
        target_field_name="current_income",
    )
    experiment_id = aclient.create_experiment(
        datasource_id=testing_datasource.datasource_id, body=request, random_state=42
    ).data.experiment_id
    aclient.commit_experiment(datasource_id=testing_datasource.datasource_id, experiment_id=experiment_id)
    eclient.get_assignment(api_key=testing_datasource.key, experiment_id=experiment_id, participant_id="p1")

    eclient.update_bandit_arm_with_participant_outcome(
        api_key=testing_datasource.key,
        body=UpdateBanditArmOutcomeRequest(outcome=42.7),
        experiment_id=experiment_id,
        participant_id="p1",
    )
    assignment = eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=experiment_id,
        participant_id="p1",
        create_if_none=False,
    ).data.assignment
    assert assignment is not None
    assert assignment.outcome == 42.7
    assert assignment.observed_at is not None


@pytest.mark.parametrize("experiment_type", [ExperimentsType.FREQ_ONLINE, ExperimentsType.FREQ_PREASSIGNED])
async def test_update_bandit_arm_with_freq_experiments_returns_422(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient, experiment_type: ExperimentsType
):
    """Frequentist experiments reject bandit outcome updates through the API."""
    experiment = await create_experiment(testing_datasource, aclient, experiment_type=experiment_type)
    if experiment_type == ExperimentsType.FREQ_ONLINE:
        assignment = eclient.get_assignment(
            api_key=testing_datasource.key,
            experiment_id=experiment.experiment_id,
            participant_id="p1",
        ).data.assignment
    else:
        assignments = eclient.get_experiment_assignments(
            api_key=testing_datasource.key, experiment_id=experiment.experiment_id
        ).data.assignments
        assignment = assignments[0] if assignments else None
    assert assignment is not None

    with pytest.raises(ExperimentsAPIClientNotDefaultStatusError) as exc:
        eclient.update_bandit_arm_with_participant_outcome(
            api_key=testing_datasource.key,
            body=UpdateBanditArmOutcomeRequest(outcome=42.7),
            experiment_id=experiment.experiment_id,
            participant_id=assignment.participant_id,
        )
    assert exc.value.result.status == HTTPStatus.UNPROCESSABLE_CONTENT
    assert "Cannot dynamically update arms for frequentist experiments" in str(exc.value.result.data)


@pytest.mark.skip("EVE-171")
async def test_normal_prior_binary_reward_fits_each_outcome_exactly_once(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """The endpoint folds one recorded outcome into a Normal/Bernoulli posterior exactly once.

    Normal/Bernoulli is the only update_arm branch that consumes every entry in its outcomes
    argument; the Beta/Bernoulli and Normal/Normal branches use only outcomes[0]. This makes it the
    branch that detects if the endpoint accidentally supplies duplicate or previously absorbed
    outcomes instead of the intended singleton. Draw rows continue to retain the complete outcome
    and context history, so a future batch implementation can deliberately recompute from the
    original prior without weakening this incremental-update invariant.
    """
    # A single arm dimension keeps the posterior easy to read. Arms start at
    # mu=[mu_init] and covariance=diag([sigma_init]) (storage_format_converters.py:532).
    initial_mu = [0.0]
    initial_covariance = [[1.0]]
    design_spec = MABExperimentSpec(
        experiment_type=ExperimentsType.MAB_ONLINE,
        experiment_name="normal prior binary reward",
        description="normal prior binary reward",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC) + timedelta(days=1),
        arms=[
            ArmBandit(arm_name="control", arm_description="", mu_init=initial_mu[0], sigma_init=1.0),
            ArmBandit(arm_name="treatment", arm_description="", mu_init=initial_mu[0], sigma_init=1.0),
        ],
        prior_type=PriorTypes.NORMAL,
        reward_type=LikelihoodTypes.BERNOULLI,
        contexts=None,
    )
    experiment_id = aclient.create_experiment(
        datasource_id=testing_datasource.datasource_id,
        body=CreateExperimentRequest(design_spec=design_spec),
    ).data.experiment_id
    aclient.commit_experiment(datasource_id=testing_datasource.datasource_id, experiment_id=experiment_id)

    eclient.get_assignment(
        api_key=testing_datasource.key,
        experiment_id=experiment_id,
        participant_id="1",
    )
    updated_arm = eclient.update_bandit_arm_with_participant_outcome(
        api_key=testing_datasource.key,
        body=UpdateBanditArmOutcomeRequest(outcome=1.0),
        experiment_id=experiment_id,
        participant_id="1",
    ).data

    # Oracle: the same model fitted with the one outcome that was actually recorded. Using
    # update_arm itself keeps the expectation free of any reimplementation of the math.
    expected = update_arm(
        experiment=tables.Experiment(
            experiment_type=ExperimentsType.MAB_ONLINE.value,
            prior_type=PriorTypes.NORMAL.value,
            reward_type=LikelihoodTypes.BERNOULLI.value,
        ),
        arm_to_update=tables.Arm(mu=initial_mu, covariance=initial_covariance),
        outcomes=[1.0],
        context=None,
    )
    assert isinstance(expected, UpdateTypeNormal)

    assert updated_arm.mu == pytest.approx(expected.mu)


async def test_update_bandit_arm_with_outcome_rejects_non_binary_outcome_for_binary_reward(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """An experiment whose reward is binary accepts only 0 or 1."""
    experiment_id = aclient.create_experiment(
        datasource_id=testing_datasource.datasource_id,
        body=CreateExperimentRequest(
            design_spec=MABExperimentSpec(
                experiment_type=ExperimentsType.MAB_ONLINE,
                experiment_name="binary reward",
                description="binary reward",
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime.now(UTC) + timedelta(days=1),
                arms=[
                    ArmBandit(arm_name="control", arm_description="", alpha_init=1, beta_init=1),
                    ArmBandit(arm_name="treatment", arm_description="", alpha_init=1, beta_init=1),
                ],
                prior_type=PriorTypes.BETA,
                reward_type=LikelihoodTypes.BERNOULLI,
                contexts=None,
            )
        ),
    ).data.experiment_id
    aclient.commit_experiment(datasource_id=testing_datasource.datasource_id, experiment_id=experiment_id)
    eclient.get_assignment(api_key=testing_datasource.key, experiment_id=experiment_id, participant_id="1")

    response = eclient.client.post(
        f"/v1/experiments/{experiment_id}/assignments/1/outcome",
        headers={"X-API-Key": testing_datasource.key},
        json={"outcome": 0.5},
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_CONTENT, response.content
    assert "Must be 0 or 1" in response.text


async def test_update_bandit_arm_with_outcome_rejects_non_numeric_outcome(
    testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """Non-numeric outcome in the request body is rejected"""
    mab_experiment = await create_experiment(testing_datasource, aclient, experiment_type=ExperimentsType.MAB_ONLINE)

    response = eclient.client.post(
        f"/v1/experiments/{mab_experiment.experiment_id}/assignments/1/outcome",
        headers={"X-API-Key": testing_datasource.key},
        json={"outcome": "not-a-float"},
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_CONTENT, response.content
