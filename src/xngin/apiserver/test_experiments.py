import dataclasses
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from deepdiff import DeepDiff
from fastapi import HTTPException
from fastapi.testclient import TestClient
from numpy.random import RandomState, MT19937
from sqlalchemy import Boolean, Column, MetaData, String, Table, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable
from xngin.apiserver import conftest, constants
from xngin.apiserver.models import tables
from xngin.apiserver.routers.stateless_api_types import (
    Arm,
    BalanceCheck,
    OnlineExperimentSpec,
    PreassignedExperimentSpec,
    DesignSpecMetric,
    DesignSpecMetricRequest,
    MetricPowerAnalysis,
    MetricPowerAnalysisMessage,
    MetricPowerAnalysisMessageType,
    MetricType,
    PowerResponse,
    Stratum,
)
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.main import app
from xngin.apiserver.models.enums import ExperimentState
from xngin.apiserver.models.tables import (
    ArmAssignment,
    ArmTable,
    Datasource,
    Experiment,
    experiment_id_factory,
    arm_id_factory,
)
from xngin.apiserver.routers.experiments import (
    ExperimentsAssignmentError,
    abandon_experiment_impl,
    commit_experiment_impl,
    create_experiment_impl,
    experiment_assignments_to_csv_generator,
    get_assign_summary,
    get_existing_assignment_for_participant,
    get_experiment_assignments_impl,
    list_experiments_impl,
    create_assignment_for_participant,
)
from xngin.apiserver.routers.experiments_api_types import (
    CreateExperimentRequest,
    CreateExperimentResponse,
    GetExperimentAssignmentsResponse,
    GetParticipantAssignmentResponse,
    ListExperimentsResponse,
)

conftest.setup(app)
client = TestClient(app)
client.base_url = str(client.base_url) + constants.API_PREFIX_V1


@pytest.fixture(name="db_session")
def fixture_db_session():
    session = next(app.dependency_overrides[xngin_db_session]())
    yield session


@pytest.fixture(autouse=True, scope="function")
def fixture_teardown(db_session: Session):
    # setup here
    yield
    # teardown here
    # Rollback any pending transactions that may have been hanging due to an exception.
    db_session.rollback()
    # Clean up objects created in each test by truncating tables and leveraging cascade.
    db_session.query(Datasource).delete()
    db_session.commit()
    db_session.close()


def make_create_preassigned_experiment_request(
    with_uuids: bool = True,
) -> CreateExperimentRequest:
    experiment_id = experiment_id_factory() if with_uuids else None
    arm1_id = arm_id_factory() if with_uuids else None
    arm2_id = arm_id_factory() if with_uuids else None
    # Attach UTC tz, but use dates_equal() to compare to respect db storage support
    start_date = datetime(2025, 1, 1, tzinfo=UTC)
    end_date = datetime(2025, 2, 1, tzinfo=UTC)
    # Construct request body
    return CreateExperimentRequest(
        design_spec=PreassignedExperimentSpec(
            participant_type="test_participant_type",
            experiment_id=experiment_id,
            experiment_name="Test Experiment",
            description="Test experiment description",
            arms=[
                Arm(
                    arm_id=arm1_id, arm_name="control", arm_description="Control group"
                ),
                Arm(
                    arm_id=arm2_id,
                    arm_name="treatment",
                    arm_description="Treatment group",
                ),
            ],
            start_date=start_date,
            end_date=end_date,
            filters=[],
            strata=[Stratum(field_name="gender")],
            metrics=[
                DesignSpecMetricRequest(
                    field_name="is_onboarded",
                    metric_pct_change=0.1,
                )
            ],
            power=0.8,
            alpha=0.05,
            fstat_thresh=0.2,
        ),
    )


# Insert an experiment with a valid state.
def make_insertable_experiment(state: ExperimentState, datasource_id="testing"):
    request = make_create_preassigned_experiment_request()
    balance_check = BalanceCheck(
        f_statistic=0.088004147,
        numerator_df=2,
        denominator_df=97,
        p_value=0.91583011,
        balance_ok=True,
    )
    return Experiment(
        id=request.design_spec.experiment_id,
        datasource_id=datasource_id,
        experiment_type="preassigned",
        participant_type=request.design_spec.participant_type,
        name=request.design_spec.experiment_name,
        description=request.design_spec.description,
        state=state,
        start_date=request.design_spec.start_date,
        end_date=request.design_spec.end_date,
        design_spec=request.design_spec.model_dump(mode="json"),
        power_analyses=PowerResponse(
            analyses=[
                MetricPowerAnalysis(
                    metric_spec=DesignSpecMetric(
                        field_name="is_onboarded",
                        metric_type=MetricType.BINARY,
                        metric_baseline=0.5,
                        metric_pct_change=0.1,
                        available_nonnull_n=1000,
                        available_n=1200,
                    ),
                    target_n=800,
                    sufficient_n=True,
                    msg=MetricPowerAnalysisMessage(
                        type=MetricPowerAnalysisMessageType.SUFFICIENT,
                        msg="Sample size is sufficient to detect the target effect",
                        source_msg="Sample size of {available_n} is sufficient to detect {target} effect",
                        values={"available_n": 1200, "target": 0.1},
                    ),
                )
            ]
        ).model_dump(),
    ).set_balance_check(balance_check)


def make_insertable_online_experiment(
    state=ExperimentState.COMMITTED, datasource_id="testing"
):
    experiment_id = experiment_id_factory()
    arm1_id = arm_id_factory()
    arm2_id = arm_id_factory()
    arm1 = Arm(arm_id=arm1_id, arm_name="control", arm_description="Control")
    arm2 = Arm(arm_id=arm2_id, arm_name="treatment", arm_description="Treatment")
    # Attach UTC tz, but use dates_equal() to compare to respect db storage support
    start_date = datetime(2025, 1, 1, tzinfo=UTC)
    end_date = datetime(2025, 2, 1, tzinfo=UTC)
    design_spec = OnlineExperimentSpec(
        participant_type="test_participant_type",
        experiment_id=experiment_id,
        experiment_name="Test Experiment",
        description="Test experiment description",
        arms=[arm1, arm2],
        start_date=start_date,
        end_date=end_date,
        filters=[],
        strata=[Stratum(field_name="gender")],
        metrics=[
            DesignSpecMetricRequest(
                field_name="is_onboarded",
                metric_pct_change=0.1,
            )
        ],
        power=0.8,
        alpha=0.05,
        fstat_thresh=0.2,
    )
    return Experiment(
        id=experiment_id,
        datasource_id=datasource_id,
        experiment_type="online",
        participant_type=design_spec.participant_type,
        name=design_spec.experiment_name,
        description=design_spec.description,
        state=state,
        start_date=design_spec.start_date,
        end_date=design_spec.end_date,
        design_spec=design_spec.model_dump(mode="json"),
    )


def make_arms_from_experiment(experiment: Experiment, organization_id: str):
    return [
        ArmTable(
            id=arm["arm_id"],
            experiment_id=experiment.id,
            name=arm["arm_name"],
            description=arm["arm_description"],
            organization_id=organization_id,
        )
        for arm in experiment.design_spec["arms"]
    ]


@dataclass
class MockRow:
    """Simulate the bits of a sqlalchemy Row that we need here."""

    participant_id: str
    gender: str
    is_onboarded: bool
    region: str = "North"  # Default value for backward compatibility

    def _asdict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@pytest.fixture
def sample_table():
    """Create a mock SQLAlchemy table that works with make_create_preassigned_experiment_request()"""
    metadata_obj = MetaData()
    return Table(
        "participants",
        metadata_obj,
        Column("participant_id", String, primary_key=True),
        Column("gender", String),
        Column("is_onboarded", Boolean),
    )


def make_sample_data(n=100):
    """Create mock participant data that works with our sample_table"""
    rs = RandomState(MT19937())
    rs.seed(42)
    return [
        MockRow(
            participant_id=f"p{i}",
            gender=rs.choice(["M", "F"]),
            is_onboarded=bool(rs.choice([True, False], p=[0.5, 0.5])),
        )
        for i in range(n)
    ]


def test_create_experiment_impl_for_preassigned(
    db_session, testing_datasource, sample_table, use_deterministic_random
):
    """Test implementation of creating a preassigned experiment."""
    participants = make_sample_data(n=100)
    request = make_create_preassigned_experiment_request(with_uuids=False)
    # Add a partial mock PowerResponse just to verify storage
    request.power_analyses = PowerResponse(
        analyses=[
            MetricPowerAnalysis(
                metric_spec=DesignSpecMetric(
                    field_name="is_onboarded", metric_type=MetricType.BINARY
                )
            )
        ]
    )

    # Test!
    response = create_experiment_impl(
        request=request.model_copy(
            deep=True
        ),  # we'll use the original request for assertions
        datasource_id=testing_datasource.ds.id,
        participant_unique_id_field="participant_id",
        dwh_sa_table=sample_table,
        dwh_participants=participants,
        random_state=42,
        xngin_session=db_session,
        stratify_on_metrics=True,
    )

    # Verify response
    assert response.datasource_id == testing_datasource.ds.id
    assert response.state == ExperimentState.ASSIGNED
    # Verify design_spec
    assert response.design_spec.experiment_id is not None
    assert response.design_spec.arms[0].arm_id is not None
    assert response.design_spec.arms[1].arm_id is not None
    assert response.design_spec.experiment_name == request.design_spec.experiment_name
    assert response.design_spec.description == request.design_spec.description
    assert response.design_spec.start_date == request.design_spec.start_date
    assert response.design_spec.end_date == request.design_spec.end_date
    # although we stratify on target metrics as well in this test, note that the
    # original strata are not augmented with the metric names.
    assert response.design_spec.strata == [Stratum(field_name="gender")]
    assert response.power_analyses == request.power_analyses
    # Verify assign_summary
    assert response.assign_summary.sample_size == len(participants)
    assert response.assign_summary.balance_check is not None
    assert response.assign_summary.balance_check.balance_ok is True

    # Verify database state using the ids in the returned DesignSpec.
    experiment: Experiment = db_session.scalars(
        select(Experiment).where(Experiment.id == response.design_spec.experiment_id)
    ).one()
    assert experiment.experiment_type == "preassigned"
    assert experiment.participant_type == request.design_spec.participant_type
    assert experiment.name == request.design_spec.experiment_name
    assert experiment.description == request.design_spec.description
    assert experiment.state == ExperimentState.ASSIGNED
    assert experiment.datasource_id == testing_datasource.ds.id
    # This comparison is dependent on whether the db can store tz or not (sqlite does not).
    assert conftest.dates_equal(experiment.start_date, request.design_spec.start_date)
    assert conftest.dates_equal(experiment.end_date, request.design_spec.end_date)
    # Verify design_spec was stored correctly
    stored_design_spec = experiment.get_design_spec()
    assert stored_design_spec == response.design_spec
    stored_power_analyses = experiment.get_power_analyses()
    assert stored_power_analyses == response.power_analyses
    # Verify assignments were created
    assignments = db_session.scalars(
        select(ArmAssignment).where(ArmAssignment.experiment_id == experiment.id)
    ).all()
    assert len(assignments) == len(participants)
    # Verify all participant IDs in the db are the participants in the request
    assignment_participant_ids = {a.participant_id for a in assignments}
    assert assignment_participant_ids == {p.participant_id for p in participants}
    assert len(assignment_participant_ids) == len(participants)

    # Verify arms were created in database
    arms = db_session.scalars(
        select(ArmTable).where(ArmTable.experiment_id == experiment.id)
    ).all()
    assert len(arms) == 2
    arm_ids = {arm.id for arm in arms}
    expected_arm_ids = {arm.arm_id for arm in response.design_spec.arms}
    assert arm_ids == expected_arm_ids

    # Check one assignment to see if it looks roughly right
    sample_assignment = assignments[0]
    assert sample_assignment.participant_type == "test_participant_type"
    assert sample_assignment.experiment_id == experiment.id
    assert sample_assignment.arm_id in (arm.arm_id for arm in response.design_spec.arms)
    # Verify strata information
    assert (
        len(sample_assignment.strata) == 2
    )  # our metric by default and the original strata
    assert sample_assignment.strata[0]["field_name"] == "gender"
    assert sample_assignment.strata[1]["field_name"] == "is_onboarded"

    # Check for approximate balance in arm assignments
    arm1_id = response.design_spec.arms[0].arm_id
    arm2_id = response.design_spec.arms[1].arm_id
    num_control = sum(1 for a in assignments if a.arm_id == arm1_id)
    num_treat = sum(1 for a in assignments if a.arm_id == arm2_id)
    assert abs(num_control - num_treat) <= 1


def test_create_experiment_impl_for_online(
    db_session, testing_datasource, sample_table, use_deterministic_random
):
    """Test implementation of creating an online experiment."""
    # Create online experiment request, modifying the experiment type from the fixture
    request = make_create_preassigned_experiment_request(with_uuids=False)
    # Convert the experiment type to online
    request.design_spec.experiment_type = "online"

    response = create_experiment_impl(
        request=request.model_copy(deep=True),
        datasource_id=testing_datasource.ds.id,
        participant_unique_id_field="participant_id",
        dwh_sa_table=sample_table,
        dwh_participants=[],  # No pre-assigned participants in online experiments
        random_state=42,
        xngin_session=db_session,
        stratify_on_metrics=True,
    )

    # Verify response
    assert response.datasource_id == testing_datasource.ds.id
    assert response.state == ExperimentState.ASSIGNED

    # Verify design_spec
    assert response.design_spec.experiment_id is not None
    assert response.design_spec.arms[0].arm_id is not None
    assert response.design_spec.arms[1].arm_id is not None
    assert response.design_spec.experiment_name == request.design_spec.experiment_name
    assert response.design_spec.description == request.design_spec.description
    assert response.design_spec.start_date == request.design_spec.start_date
    assert response.design_spec.end_date == request.design_spec.end_date
    assert response.design_spec.strata == [Stratum(field_name="gender")]
    assert (
        response.power_analyses is None
    )  # Online experiments don't have power analyses by default

    # Verify assign_summary for online experiment
    assert response.assign_summary.sample_size == 0
    assert response.assign_summary.balance_check is None
    assert response.assign_summary.arm_sizes is not None
    assert all(arm_size.size == 0 for arm_size in response.assign_summary.arm_sizes)

    # Verify database state
    experiment: Experiment = db_session.scalars(
        select(Experiment).where(Experiment.id == response.design_spec.experiment_id)
    ).one()
    assert experiment.experiment_type == "online"
    assert experiment.participant_type == request.design_spec.participant_type
    assert experiment.name == request.design_spec.experiment_name
    assert experiment.description == request.design_spec.description
    # Online experiments still go through a review step before being committed
    assert experiment.state == ExperimentState.ASSIGNED
    assert experiment.datasource_id == testing_datasource.ds.id
    assert conftest.dates_equal(experiment.start_date, request.design_spec.start_date)
    assert conftest.dates_equal(experiment.end_date, request.design_spec.end_date)

    # Verify design_spec was stored correctly
    stored_design_spec = experiment.get_design_spec()
    assert stored_design_spec == response.design_spec
    # Verify no power_analyses for online experiments
    assert experiment.power_analyses is None

    # Verify arms were created in database
    arms = db_session.scalars(
        select(ArmTable).where(ArmTable.experiment_id == experiment.id)
    ).all()
    assert len(arms) == 2
    arm_ids = {arm.id for arm in arms}
    expected_arm_ids = {arm.arm_id for arm in response.design_spec.arms}
    assert arm_ids == expected_arm_ids

    # Verify that no assignments were created for online experiment
    assignments = db_session.scalars(
        select(ArmAssignment).where(ArmAssignment.experiment_id == experiment.id)
    ).all()
    assert len(assignments) == 0


def test_create_experiment_impl_overwrites_uuids(
    db_session, testing_datasource, sample_table, use_deterministic_random
):
    """
    Test that the function overwrites requests with preset UUIDs
    (which would otherwise be caught in the route handler).
    """
    participants = make_sample_data(n=100)
    request = make_create_preassigned_experiment_request(with_uuids=True)
    original_experiment_id = request.design_spec.experiment_id
    original_arm_ids = [arm.arm_id for arm in request.design_spec.arms]

    # Call the function under test
    response = create_experiment_impl(
        request=request,
        datasource_id=testing_datasource.ds.id,
        participant_unique_id_field="participant_id",
        dwh_sa_table=sample_table,
        dwh_participants=participants,
        random_state=42,
        xngin_session=db_session,
        stratify_on_metrics=True,
    )

    # Verify that new UUIDs were generated
    assert response.design_spec.experiment_id != original_experiment_id
    new_arm_ids = [arm.arm_id for arm in response.design_spec.arms]
    assert set(new_arm_ids) != set(original_arm_ids)

    # Verify database state
    experiment = db_session.scalars(
        select(Experiment).where(Experiment.id == response.design_spec.experiment_id)
    ).one()
    assert experiment.state == ExperimentState.ASSIGNED
    # Verify assignments were created with the new UUIDs
    assignments = db_session.scalars(
        select(ArmAssignment).where(ArmAssignment.experiment_id == experiment.id)
    ).all()
    # Verify all assignments use the new arm IDs
    assignment_arm_ids = {a.arm_id for a in assignments}
    assert assignment_arm_ids == set(new_arm_ids)


def test_create_experiment_impl_no_metric_stratification(
    db_session, testing_datasource, sample_table, use_deterministic_random
):
    """Test implementation of creating an experiment without stratifying on metrics."""
    participants = make_sample_data(n=100)
    request = make_create_preassigned_experiment_request(with_uuids=False)

    # Test with stratify_on_metrics=False
    response = create_experiment_impl(
        request=request.model_copy(deep=True),
        datasource_id=testing_datasource.ds.id,
        participant_unique_id_field="participant_id",
        dwh_sa_table=sample_table,
        dwh_participants=participants,
        random_state=42,
        xngin_session=db_session,
        stratify_on_metrics=False,
    )

    # Verify basic response
    assert response.datasource_id == testing_datasource.ds.id
    assert response.state == ExperimentState.ASSIGNED
    assert response.design_spec.experiment_id is not None
    assert response.design_spec.arms[0].arm_id is not None
    # Same as in the stratify_on_metrics=True test.
    # Only the output assignments will also store a snapshot of the metric values as strata.
    assert response.design_spec.strata == [Stratum(field_name="gender")]

    # Verify database state
    experiment = db_session.scalars(
        select(Experiment).where(Experiment.id == response.design_spec.experiment_id)
    ).one()
    # Verify assignments were created
    assignments = db_session.scalars(
        select(ArmAssignment).where(ArmAssignment.experiment_id == experiment.id)
    ).all()
    assert len(assignments) == len(participants)
    # Check strata information only has gender, not is_onboarded
    sample_assignment = assignments[0]
    assert len(sample_assignment.strata) == 1
    assert sample_assignment.strata[0]["field_name"] == "gender"
    assert not any(s["field_name"] == "is_onboarded" for s in sample_assignment.strata)

    # Check for approximate balance in arm assignments
    arm1_id = response.design_spec.arms[0].arm_id
    arm2_id = response.design_spec.arms[1].arm_id
    num_control = sum(1 for a in assignments if a.arm_id == arm1_id)
    num_treat = sum(1 for a in assignments if a.arm_id == arm2_id)
    assert abs(num_control - num_treat) <= 1


def test_create_experiment_impl_invalid_design_spec(db_session):
    """Test creating an experiment and saving assignments to the database."""
    request = make_create_preassigned_experiment_request(with_uuids=True)

    response = client.post(
        "/experiments/with-assignment",
        params={"chosen_n": 100},
        headers={constants.HEADER_CONFIG_ID: "testing"},
        content=request.model_dump_json(),
    )
    assert response.status_code == 422, request
    assert "UUIDs must not be set" in response.json()["message"]


def test_create_experiment_with_assignment_sl(
    db_session, sample_table, use_deterministic_random
):
    """Test creating an experiment and saving assignments to the database."""
    # First create a datasource to maintain proper referential integrity, but with a local config so we know we can read our dwh data.
    ds_metadata = conftest.make_datasource_metadata(db_session, datasource_id="testing")
    request = make_create_preassigned_experiment_request(with_uuids=False)

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


@pytest.mark.parametrize(
    "method_under_test,initial_state,expected_state,expected_status,expected_detail",
    [
        # Success case
        (
            commit_experiment_impl,
            ExperimentState.ASSIGNED,
            ExperimentState.COMMITTED,
            204,
            None,
        ),
        # No-op
        (
            commit_experiment_impl,
            ExperimentState.COMMITTED,
            ExperimentState.COMMITTED,
            304,
            None,
        ),
        # Failure cases
        (
            commit_experiment_impl,
            ExperimentState.DESIGNING,
            ExperimentState.DESIGNING,
            400,
            "Invalid state: designing",
        ),
        (
            commit_experiment_impl,
            ExperimentState.ABORTED,
            ExperimentState.ABORTED,
            400,
            "Invalid state: aborted",
        ),
        # Success cases
        (
            abandon_experiment_impl,
            ExperimentState.DESIGNING,
            ExperimentState.ABANDONED,
            204,
            None,
        ),
        (
            abandon_experiment_impl,
            ExperimentState.ASSIGNED,
            ExperimentState.ABANDONED,
            204,
            None,
        ),
        # No-op
        (
            abandon_experiment_impl,
            ExperimentState.ABANDONED,
            ExperimentState.ABANDONED,
            304,
            None,
        ),
        # Failure case
        (
            abandon_experiment_impl,
            ExperimentState.COMMITTED,
            ExperimentState.COMMITTED,
            400,
            "Invalid state: committed",
        ),
    ],
)
def test_state_setting_experiment_impl(
    db_session,
    testing_datasource,
    method_under_test,
    initial_state,
    expected_state,
    expected_status,
    expected_detail,
):
    # Initialize our state with an existing experiment who's state we want to modify.
    experiment = make_insertable_experiment(initial_state, testing_datasource.ds.id)
    db_session.add(experiment)
    db_session.commit()

    try:
        response = method_under_test(db_session, experiment)
    except HTTPException as e:
        assert e.status_code == expected_status
        assert e.detail == expected_detail
    else:
        assert response.status_code == expected_status
        assert experiment.state == expected_state


def test_list_experiments_sl_without_api_key(db_session, testing_datasource):
    """Tests that listing experiments tied to a db datasource requires an API key.

    TODO: This indirectly tests that the datasource_dependency (i.e. sheets-based config) enforces
    authentication, although is not used by any client. Likely we will deprecate this method of auth
    used by routes in experiments.py to keep just the pure stateless and admin.py APIs."""
    experiment = make_insertable_experiment(
        ExperimentState.ASSIGNED, testing_datasource.ds.id
    )
    db_session.add(experiment)
    db_session.commit()

    response = client.get(
        "/experiments",
        headers={constants.HEADER_CONFIG_ID: testing_datasource.ds.id},
    )
    assert response.status_code == 403
    assert response.json()["message"] == "API key missing or invalid."


def test_list_experiments_sl_with_api_key(db_session, testing_datasource):
    """Tests that listing experiments tied to a db datasource with an API key works.

    TODO: deprecate/remove when we can officially move off sheets-based configuration.
    """

    expected_experiment = make_insertable_experiment(
        ExperimentState.ASSIGNED, testing_datasource.ds.id
    )
    db_session.add(expected_experiment)
    db_session.commit()

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
    diff = DeepDiff(
        expected_experiment.get_design_spec(), experiments.items[0].design_spec
    )
    assert not diff, f"Objects differ:\n{diff.pretty()}"


def test_list_experiments_impl(db_session, testing_datasource):
    """Test that we only get experiments in a valid state for the specified datasource."""
    experiment1 = make_insertable_experiment(
        ExperimentState.ASSIGNED, testing_datasource.ds.id
    )
    experiment2 = make_insertable_experiment(
        ExperimentState.COMMITTED, testing_datasource.ds.id
    )
    experiment3 = make_insertable_experiment(
        ExperimentState.DESIGNING, testing_datasource.ds.id
    )
    experiment4 = make_insertable_experiment(
        ExperimentState.ABORTED, testing_datasource.ds.id
    )
    # One more experiment associated with a *different* datasource.
    experiment5_metadata = conftest.make_datasource_metadata(db_session)
    experiment5 = make_insertable_experiment(
        ExperimentState.ASSIGNED, datasource_id=experiment5_metadata.ds.id
    )
    # Set the created_at time to test ordering
    experiment1.created_at = datetime.now(UTC) - timedelta(days=1)
    experiment2.created_at = datetime.now(UTC)
    experiment3.created_at = datetime.now(UTC) + timedelta(days=1)
    db_session.add_all([
        experiment1,
        experiment2,
        experiment3,
        experiment4,
        experiment5,
    ])
    db_session.commit()

    experiments = list_experiments_impl(db_session, testing_datasource.ds.id)

    # experiment5 excluded due to datasource mismatch
    assert len(experiments.items) == 3
    actual1 = experiments.items[2]  # experiment1 is last as it's oldest
    actual2 = experiments.items[1]
    actual3 = experiments.items[0]
    assert actual1.state == ExperimentState.ASSIGNED
    diff = DeepDiff(actual1.design_spec, experiment1.get_design_spec())
    assert not diff, f"Objects differ:\n{diff.pretty()}"
    assert actual2.state == ExperimentState.COMMITTED
    diff = DeepDiff(actual2.design_spec, experiment2.get_design_spec())
    assert not diff, f"Objects differ:\n{diff.pretty()}"
    assert actual3.state == ExperimentState.DESIGNING
    diff = DeepDiff(actual3.design_spec, experiment3.get_design_spec())
    assert not diff, f"Objects differ:\n{diff.pretty()}"


def test_get_experiment(db_session, testing_datasource):
    """TODO: deprecate in favor of admin.py version when ready."""
    new_experiment = make_insertable_experiment(
        ExperimentState.DESIGNING, testing_datasource.ds.id
    )
    db_session.add(new_experiment)
    db_session.commit()

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
    expected = PreassignedExperimentSpec.model_validate(new_experiment.design_spec)
    diff = DeepDiff(actual, expected)
    assert not diff, f"Objects differ:\n{diff.pretty()}"


def test_get_experiment_assignments_impl(db_session, testing_datasource):
    # First insert an experiment with assignments
    experiment = make_insertable_experiment(
        ExperimentState.COMMITTED, testing_datasource.ds.id
    )
    experiment_id = experiment.id
    db_session.add(experiment)
    db_arms = make_arms_from_experiment(
        experiment, testing_datasource.ds.organization_id
    )
    db_session.add_all(db_arms)

    arm1_id = experiment.design_spec["arms"][0]["arm_id"]
    arm2_id = experiment.design_spec["arms"][1]["arm_id"]
    arm_assignments = [
        ArmAssignment(
            experiment_id=experiment_id,
            participant_type="test_participant_type",
            participant_id="p1",
            arm_id=arm1_id,
            strata=[{"field_name": "gender", "strata_value": "F"}],
        ),
        ArmAssignment(
            experiment_id=experiment_id,
            participant_type="test_participant_type",
            participant_id="p2",
            arm_id=arm2_id,
            strata=[{"field_name": "gender", "strata_value": "M"}],
        ),
    ]
    db_session.add_all(arm_assignments)
    db_session.commit()

    data: GetExperimentAssignmentsResponse = get_experiment_assignments_impl(experiment)

    # Check the response structure
    assert data.experiment_id == experiment.id
    assert data.sample_size == get_assign_summary(db_session, experiment).sample_size
    assert data.balance_check == experiment.get_balance_check()

    # Check assignments
    assignments = data.assignments
    assert len(assignments) == 2

    # Verify first assignment
    assert assignments[0].participant_id == "p1"
    assert str(assignments[0].arm_id) == arm1_id
    assert assignments[0].arm_name == "control"
    assert assignments[0].strata is not None and len(assignments[0].strata) == 1
    assert assignments[0].strata[0].field_name == "gender"
    assert assignments[0].strata[0].strata_value == "F"

    # Verify second assignment
    assert assignments[1].participant_id == "p2"
    assert str(assignments[1].arm_id) == arm2_id
    assert assignments[1].arm_name == "treatment"
    assert assignments[1].strata is not None and len(assignments[1].strata) == 1
    assert assignments[1].strata[0].field_name == "gender"
    assert assignments[1].strata[0].strata_value == "M"


def test_get_experiment_assignments_not_found():
    """Test getting assignments for a non-existent experiment.

    TODO: deprecate this in favor of an admin.py version when ready.
    """
    response = client.get(
        f"/experiments/{experiment_id_factory()}/assignments",
        headers={constants.HEADER_CONFIG_ID: "testing"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment not found"


def test_get_experiment_assignments_wrong_datasource(db_session, testing_datasource):
    """Test getting assignments for an experiment from a different datasource.

    TODO: deprecate this in favor of an admin.py version when ready.
    """
    # Create experiment in one datasource
    experiment = make_insertable_experiment(
        ExperimentState.COMMITTED, testing_datasource.ds.id
    )
    db_session.add(experiment)
    db_session.commit()

    # Try to get it from another datasource
    response = client.get(
        f"/experiments/{experiment.id!s}/assignments",
        headers={constants.HEADER_CONFIG_ID: "testing-inline-schema"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment not found"


def make_experiment_with_assignments(db_session, datasource: Datasource):
    """Helper function for the tests below."""
    # First insert an experiment with assignments
    experiment = make_insertable_experiment(ExperimentState.COMMITTED, datasource.id)
    db_session.add(experiment)
    arms = make_arms_from_experiment(experiment, datasource.organization_id)
    db_session.add_all(arms)
    arm1_id = experiment.design_spec["arms"][0]["arm_id"]
    arm2_id = experiment.design_spec["arms"][1]["arm_id"]
    assignments = [
        ArmAssignment(
            experiment_id=experiment.id,
            participant_type="test_participant_type",
            participant_id="p1",
            arm_id=arm1_id,
            strata=[
                {"field_name": "gender", "strata_value": "F"},
                {"field_name": "score", "strata_value": "1.1"},
            ],
        ),
        ArmAssignment(
            experiment_id=experiment.id,
            participant_type="test_participant_type",
            participant_id="p2",
            arm_id=arm2_id,
            strata=[
                {"field_name": "gender", "strata_value": "M"},
                {"field_name": "score", "strata_value": "esc,aped"},
            ],
        ),
    ]
    db_session.add_all(assignments)
    db_session.commit()
    return experiment


def test_experiment_assignments_to_csv_generator(db_session, testing_datasource):
    experiment = make_experiment_with_assignments(db_session, testing_datasource.ds)

    (arm1_id, arm2_id) = experiment.get_arm_ids()
    (arm1_name, arm2_name) = experiment.get_arm_names()
    batches = list(experiment_assignments_to_csv_generator(experiment)())
    assert len(batches) == 1
    rows = batches[0].splitlines(keepends=True)
    assert rows[0] == "participant_id,arm_id,arm_name,gender,score\r\n"
    assert rows[1] == f"p1,{arm1_id},{arm1_name},F,1.1\r\n"
    assert rows[2] == f'p2,{arm2_id},{arm2_name},M,"esc,aped"\r\n'


def test_get_experiment_assignments_as_csv(db_session, testing_datasource):
    experiment = make_experiment_with_assignments(db_session, testing_datasource.ds)

    response = client.get(
        f"/experiments/{experiment.id!s}/assignments/csv",
        headers={
            constants.HEADER_CONFIG_ID: testing_datasource.ds.id,
            constants.HEADER_API_KEY: testing_datasource.key,
        },
    )
    assert response.status_code == 200
    assert (
        f"experiment_{experiment.id}_assignments.csv"
        in response.headers["Content-Disposition"]
    )
    # Now verify the contents
    rows = response.text.splitlines(keepends=True)
    arm_name_to_id = {a.name: a.id for a in experiment.arms}
    assert len(rows) == 3
    assert rows[0] == "participant_id,arm_id,arm_name,gender,score\r\n"
    assert rows[1] == f"p1,{arm_name_to_id['control']},control,F,1.1\r\n"
    assert rows[2] == f'p2,{arm_name_to_id["treatment"]},treatment,M,"esc,aped"\r\n'


def test_get_assignment_for_preassigned_participant_with_apikey(
    db_session, testing_datasource
):
    preassigned_experiment = make_insertable_experiment(
        ExperimentState.COMMITTED, testing_datasource.ds.id
    )
    db_session.add(preassigned_experiment)
    arms = make_arms_from_experiment(
        preassigned_experiment, testing_datasource.ds.organization_id
    )
    db_session.add_all(arms)
    assignment = tables.ArmAssignment(
        experiment_id=preassigned_experiment.id,
        participant_id="assigned_id",
        participant_type=preassigned_experiment.participant_type,
        arm_id=arms[0].id,
        strata=[],
    )
    db_session.add(assignment)
    db_session.commit()

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
    db_session, testing_datasource
):
    """Test endpoint that gets an assignment for a participant via API key."""
    online_experiment = make_insertable_online_experiment(
        ExperimentState.COMMITTED, testing_datasource.ds.id
    )
    db_session.add(online_experiment)
    arms = make_arms_from_experiment(
        online_experiment, testing_datasource.ds.organization_id
    )
    db_session.add_all(arms)
    db_session.commit()

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
    assignment = db_session.scalars(
        select(tables.ArmAssignment).where(
            tables.ArmAssignment.experiment_id == online_experiment.id
        )
    ).one()
    assert assignment.participant_id == "1"
    assert assignment.arm_id == str(parsed.assignment.arm_id)


def test_get_existing_assignment_for_participant(db_session, testing_datasource):
    experiment = make_experiment_with_assignments(db_session, testing_datasource.ds)
    expected_assignment = experiment.arm_assignments[0]

    assignment = get_existing_assignment_for_participant(
        db_session, experiment.id, expected_assignment.participant_id
    )
    assert assignment is not None
    assert assignment.participant_id == expected_assignment.participant_id
    assert str(assignment.arm_id) == expected_assignment.arm_id

    assignment = get_existing_assignment_for_participant(
        db_session, experiment.id, "new_id"
    )
    assert assignment is None


def test_make_assignment_for_participant_errors(db_session, testing_datasource):
    experiment = make_insertable_experiment(
        ExperimentState.ASSIGNED, testing_datasource.ds.id
    )
    with pytest.raises(
        ExperimentsAssignmentError, match="Invalid experiment state: assigned"
    ):
        create_assignment_for_participant(db_session, experiment, "p1", None)

    experiment = make_insertable_experiment(
        ExperimentState.COMMITTED, testing_datasource.ds.id
    )
    with pytest.raises(ExperimentsAssignmentError, match="Experiment has no arms"):
        create_assignment_for_participant(db_session, experiment, "p1", None)


def test_make_assignment_for_participant(db_session, testing_datasource):
    preassigned_experiment = make_insertable_experiment(
        ExperimentState.COMMITTED, testing_datasource.ds.id
    )
    db_session.add(preassigned_experiment)
    arms = make_arms_from_experiment(
        preassigned_experiment, testing_datasource.ds.organization_id
    )
    db_session.add_all(arms)
    db_session.commit()
    # Assert that we won't create new assignments for preassigned experiments
    expect_none = create_assignment_for_participant(
        db_session, preassigned_experiment, "new_id", None
    )
    assert expect_none is None

    online_experiment = make_insertable_online_experiment(
        ExperimentState.COMMITTED, testing_datasource.ds.id
    )
    db_session.add(online_experiment)
    arms = make_arms_from_experiment(
        online_experiment, testing_datasource.ds.organization_id
    )
    db_session.add_all(arms)
    db_session.commit()
    # Assert that we do create new assignments for online experiments
    assignment = create_assignment_for_participant(
        db_session, online_experiment, "new_id", None
    )
    assert assignment is not None
    assert assignment.participant_id == "new_id"
    arms = {arm.id: arm.name for arm in arms}
    assert assignment.arm_name == arms[str(assignment.arm_id)]
    assert not assignment.strata

    # But that if we try to create an assignment for a participant that already has one, it triggers an error.
    with pytest.raises(
        ExperimentsAssignmentError, match="Failed to assign participant"
    ):
        create_assignment_for_participant(db_session, online_experiment, "new_id", None)


def test_experiment_sql():
    pg_sql = str(
        CreateTable(ArmAssignment.__table__).compile(dialect=postgresql.dialect())
    )
    assert "arm_id VARCHAR(36) NOT NULL," in pg_sql
    assert "strata JSONB NOT NULL," in pg_sql
