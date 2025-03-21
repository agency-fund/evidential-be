import uuid
from datetime import datetime, timedelta, UTC
import dataclasses
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
import pytest
import numpy as np
from deepdiff import DeepDiff
from fastapi.testclient import TestClient
from pydantic_core import to_jsonable_python
from sqlalchemy import select, Boolean, Column, MetaData, String, Table
from sqlalchemy.dialects import sqlite, postgresql
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from xngin.apiserver import conftest, constants
from xngin.apiserver.api_types import (
    Arm,
    AudienceSpec,
    BalanceCheck,
    DesignSpec,
    DesignSpecMetric,
    DesignSpecMetricRequest,
    MetricAnalysis,
    MetricAnalysisMessage,
    MetricAnalysisMessageType,
    MetricType,
    PowerResponse,
)
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.main import app
from xngin.apiserver.models.enums import ExperimentState
from xngin.apiserver.models.tables import ArmAssignment, Experiment
from xngin.apiserver.routers.experiments import (
    abandon_experiment_impl,
    commit_experiment_impl,
    experiment_assignments_to_csv_generator,
    create_experiment_with_assignment_impl,
)
from xngin.apiserver.routers.experiments_api_types import (
    CreateExperimentRequest,
    AssignSummary,
    ExperimentConfig,
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
    db_session.query(Experiment).delete()
    db_session.query(ArmAssignment).delete()
    db_session.commit()
    db_session.close()


def make_create_experiment_request(with_uuids: bool = True) -> CreateExperimentRequest:
    experiment_id = str(uuid.uuid4()) if with_uuids else None
    arm1_id = str(uuid.uuid4()) if with_uuids else None
    arm2_id = str(uuid.uuid4()) if with_uuids else None
    # Use timestamps without timezone to be database agnostic
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 2, 1)
    # Construct request body
    return CreateExperimentRequest(
        design_spec=DesignSpec(
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
            strata_field_names=["gender"],
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
        audience_spec=AudienceSpec(
            participant_type="test_participant_type",
            filters=[],
        ),
    )


# Insert an experiment with a valid state.
def make_insertable_experiment(state: ExperimentState, datasource_id="testing"):
    request = make_create_experiment_request()
    return Experiment(
        id=str(request.design_spec.experiment_id),
        datasource_id=datasource_id,
        state=state,
        start_date=request.design_spec.start_date,
        end_date=request.design_spec.end_date,
        design_spec=to_jsonable_python(request.design_spec),
        audience_spec=request.audience_spec.model_dump(),
        power_analyses=PowerResponse(
            analyses=[
                MetricAnalysis(
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
                    msg=MetricAnalysisMessage(
                        type=MetricAnalysisMessageType.SUFFICIENT,
                        msg="Sample size is sufficient to detect the target effect",
                        source_msg="Sample size of {available_n} is sufficient to detect {target} effect",
                        values={"available_n": 1200, "target": 0.1},
                    ),
                )
            ]
        ).model_dump(),
        assign_summary=AssignSummary(
            sample_size=100,
            balance_check=BalanceCheck(
                f_statistic=0.088004147,
                numerator_df=2,
                denominator_df=97,
                p_value=0.91583011,
                balance_ok=True,
            ),
        ).model_dump(),
    )


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
    """Create a mock SQLAlchemy table that works with make_create_experiment_request()"""
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
    np.random.seed(42)
    return [
        MockRow(
            participant_id=f"p{i}",
            gender=np.random.choice(["M", "F"]),
            is_onboarded=bool(np.random.choice([True, False], p=[0.5, 0.5])),
        )
        for i in range(n)
    ]


def test_create_experiment_with_assignment_impl(
    db_session: Session, sample_table, use_deterministic_random
):
    """Test implementation of creating an experiment with assignment."""
    participants = make_sample_data(n=100)
    request = make_create_experiment_request(with_uuids=False)
    # Add a partial mock PowerResponse just to verify storage
    request.power_analyses = PowerResponse(
        analyses=[
            MetricAnalysis(
                metric_spec=DesignSpecMetric(
                    field_name="is_onboarded", metric_type=MetricType.BINARY
                )
            )
        ]
    )

    # Test!
    response = create_experiment_with_assignment_impl(
        request=request.model_copy(
            deep=True
        ),  # we'll use the original request for assertions
        datasource_id="testing",
        participant_unique_id_field="participant_id",
        dwh_sa_table=sample_table,
        dwh_participants=participants,
        random_state=42,
        xngin_session=db_session,
        stratify_on_metrics=True,
    )

    # Verify response
    assert response.datasource_id == "testing"
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
    # original strata_field_names are not augmented with the metric names.
    assert response.design_spec.strata_field_names == ["gender"]
    assert response.audience_spec == request.audience_spec
    assert response.power_analyses == request.power_analyses
    # Verify assign_summary
    assert response.assign_summary.sample_size == len(participants)
    assert response.assign_summary.balance_check.balance_ok is True

    # Verify database state using the ids in the returned DesignSpec.
    experiment = db_session.scalars(
        select(Experiment).where(
            Experiment.id == str(response.design_spec.experiment_id)
        )
    ).one()
    assert experiment.state == ExperimentState.ASSIGNED
    assert experiment.datasource_id == "testing"
    assert experiment.start_date == request.design_spec.start_date
    assert experiment.end_date == request.design_spec.end_date
    # Verify design_spec and audience_spec were stored correctly
    stored_design_spec = experiment.get_design_spec()
    assert stored_design_spec == response.design_spec
    stored_audience_spec = experiment.get_audience_spec()
    assert stored_audience_spec == response.audience_spec
    stored_power_analyses = experiment.get_power_analyses()
    assert stored_power_analyses == response.power_analyses
    # Verify assignments were created
    assignments = db_session.scalars(
        select(ArmAssignment).where(ArmAssignment.experiment_id == str(experiment.id))
    ).all()
    assert len(assignments) == len(participants)
    # Verify all participant IDs in the db are the participants in the request
    assignment_participant_ids = {a.participant_id for a in assignments}
    assert assignment_participant_ids == {p.participant_id for p in participants}
    assert len(assignment_participant_ids) == len(participants)

    # Check one assignment to see if it looks roughly right
    sample_assignment = assignments[0]
    assert sample_assignment.participant_type == "test_participant_type"
    assert sample_assignment.experiment_id == experiment.id
    assert sample_assignment.arm_id in (
        str(arm.arm_id) for arm in response.design_spec.arms
    )
    # Verify strata information
    assert (
        len(sample_assignment.strata) == 2
    )  # our metric by default and strata_field_names
    assert sample_assignment.strata[0]["field_name"] == "gender"
    assert sample_assignment.strata[1]["field_name"] == "is_onboarded"

    # Check for approximate balance in arm assignments
    arm1_id = response.design_spec.arms[0].arm_id
    arm2_id = response.design_spec.arms[1].arm_id
    num_control = sum(1 for a in assignments if a.arm_id == arm1_id)
    num_treat = sum(1 for a in assignments if a.arm_id == arm2_id)
    assert abs(num_control - num_treat) <= 1


def test_create_experiment_with_assignment_impl_overwrites_uuids(
    db_session: Session, sample_table, use_deterministic_random
):
    """
    Test that the function overwrites requests with preset UUIDs
    (which would otherwise be caught in the route handler).
    """
    participants = make_sample_data(n=100)
    request = make_create_experiment_request(with_uuids=True)
    original_experiment_id = str(request.design_spec.experiment_id)
    original_arm_ids = [str(arm.arm_id) for arm in request.design_spec.arms]

    # Call the function under test
    response = create_experiment_with_assignment_impl(
        request=request,
        datasource_id="testing",
        participant_unique_id_field="participant_id",
        dwh_sa_table=sample_table,
        dwh_participants=participants,
        random_state=42,
        xngin_session=db_session,
        stratify_on_metrics=True,
    )

    # Verify that new UUIDs were generated
    assert response.design_spec.experiment_id != original_experiment_id
    new_arm_ids = [str(arm.arm_id) for arm in response.design_spec.arms]
    assert set(new_arm_ids) != set(original_arm_ids)

    # Verify database state
    experiment = db_session.scalars(
        select(Experiment).where(
            Experiment.id == str(response.design_spec.experiment_id)
        )
    ).one()
    assert experiment.state == ExperimentState.ASSIGNED
    # Verify assignments were created with the new UUIDs
    assignments = db_session.scalars(
        select(ArmAssignment).where(ArmAssignment.experiment_id == str(experiment.id))
    ).all()
    # Verify all assignments use the new arm IDs
    assignment_arm_ids = {str(a.arm_id) for a in assignments}
    assert assignment_arm_ids == set(new_arm_ids)


def test_create_experiment_with_assignment_impl_no_metric_stratification(
    db_session: Session, sample_table, use_deterministic_random
):
    """Test implementation of creating an experiment without stratifying on metrics."""
    participants = make_sample_data(n=100)
    request = make_create_experiment_request(with_uuids=False)

    # Test with stratify_on_metrics=False
    response = create_experiment_with_assignment_impl(
        request=request.model_copy(deep=True),
        datasource_id="testing",
        participant_unique_id_field="participant_id",
        dwh_sa_table=sample_table,
        dwh_participants=participants,
        random_state=42,
        xngin_session=db_session,
        stratify_on_metrics=False,
    )

    # Verify basic response
    assert response.datasource_id == "testing"
    assert response.state == ExperimentState.ASSIGNED
    assert response.design_spec.experiment_id is not None
    assert response.design_spec.arms[0].arm_id is not None
    # Same as in the stratify_on_metrics=True test. Only the output assignments will also store a snapshot of the metric values as strata.
    assert response.design_spec.strata_field_names == ["gender"]

    # Verify database state
    experiment = db_session.scalars(
        select(Experiment).where(
            Experiment.id == str(response.design_spec.experiment_id)
        )
    ).one()
    # Verify assignments were created
    assignments = db_session.scalars(
        select(ArmAssignment).where(ArmAssignment.experiment_id == str(experiment.id))
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


def test_create_experiment_with_assignment_invalid_design_spec(db_session: Session):
    """Test creating an experiment and saving assignments to the database."""
    request = make_create_experiment_request(with_uuids=True)

    response = client.post(
        "/experiments/with-assignment",
        params={"chosen_n": 100, "random_state": 42},
        headers={constants.HEADER_CONFIG_ID: "testing"},
        content=request.model_dump_json(),
    )
    assert response.status_code == 422, request
    assert "UUIDs must not be set" in response.json()["message"]


def test_create_experiment_with_assignment(
    db_session: Session, use_deterministic_random
):
    """Test creating an experiment and saving assignments to the database."""
    request = make_create_experiment_request(with_uuids=False)

    response = client.post(
        "/experiments/with-assignment",
        params={"chosen_n": 100, "random_state": 42},
        headers={constants.HEADER_CONFIG_ID: "testing"},
        content=request.model_dump_json(),
    )

    # Verify basic response
    assert response.status_code == 200, request
    experiment_config = response.json()
    assert experiment_config["design_spec"]["experiment_id"] is not None
    assert experiment_config["design_spec"]["arms"][0]["arm_id"] is not None
    assert experiment_config["design_spec"]["arms"][1]["arm_id"] is not None
    assert experiment_config["datasource_id"] == "testing"
    assert experiment_config["state"] == ExperimentState.ASSIGNED
    assign_summary = experiment_config["assign_summary"]
    assert assign_summary["sample_size"] == 100
    assert assign_summary["balance_check"]["balance_ok"] is False, assign_summary[
        "balance_check"
    ]
    # Check if the representations are equivalent
    config = ExperimentConfig.model_validate(experiment_config)
    # scrub the uuids from the config for comparison
    actual_design_spec = config.design_spec.model_copy(deep=True)
    actual_design_spec.experiment_id = None
    actual_design_spec.arms[0].arm_id = None
    actual_design_spec.arms[1].arm_id = None
    assert actual_design_spec == request.design_spec
    assert config.audience_spec == request.audience_spec
    assert config.power_analyses == request.power_analyses

    experiment_id = config.design_spec.experiment_id
    (arm1_id, arm2_id) = [str(arm.arm_id) for arm in config.design_spec.arms]
    # Verify database state using the ids in the returned DesignSpec.
    experiment = db_session.scalars(
        select(Experiment).where(Experiment.id == str(experiment_id))
    ).one()
    assert experiment.state == ExperimentState.ASSIGNED
    assert experiment.datasource_id == "testing"
    assert experiment.start_date == request.design_spec.start_date
    assert experiment.end_date == request.design_spec.end_date
    # Verify assignments were created
    assignments = db_session.scalars(
        select(ArmAssignment).where(ArmAssignment.experiment_id == str(experiment_id))
    ).all()
    assert len(assignments) == 100, {
        e.name: getattr(experiment, e.name) for e in Experiment.__table__.columns
    }

    # Check one assignment to see if it looks roughly right
    sample_assignment: ArmAssignment = assignments[0]
    assert sample_assignment.participant_type == "test_participant_type"
    assert sample_assignment.experiment_id == str(experiment_id)
    assert sample_assignment.arm_id in (arm1_id, arm2_id)
    for stratum in sample_assignment.strata:
        assert stratum["field_name"] in {"gender", "is_onboarded"}

    # Check for approximate balance in arm assignment
    num_control = sum(1 for a in assignments if a.arm_id == arm1_id)
    num_treat = sum(1 for a in assignments if a.arm_id == arm2_id)
    assert abs(num_control - num_treat) <= 5  # Allow some wiggle room


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
            403,
            "Invalid state: designing",
        ),
        (
            commit_experiment_impl,
            ExperimentState.ABORTED,
            ExperimentState.ABORTED,
            403,
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
            403,
            "Invalid state: committed",
        ),
    ],
)
def test_state_setting_experiment_impl(
    db_session,
    method_under_test,
    initial_state,
    expected_state,
    expected_status,
    expected_detail,
):
    # Initialize our state with an existing experiment who's state we want to modify.
    experiment = make_insertable_experiment(initial_state)
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


def test_list_experiments(db_session: Session):
    experiment1 = make_insertable_experiment(
        ExperimentState.ASSIGNED, datasource_id="testing"
    )
    experiment2 = make_insertable_experiment(
        ExperimentState.COMMITTED, datasource_id="testing"
    )
    experiment3 = make_insertable_experiment(
        ExperimentState.ABORTED, datasource_id="testing-inline-schema"
    )
    # Set the created_at time to test ordering
    experiment1.created_at = datetime.now(UTC) - timedelta(days=1)
    experiment2.created_at = datetime.now(UTC)
    experiment3.created_at = datetime.now(UTC) + timedelta(days=1)
    db_session.add_all([experiment1, experiment2, experiment3])
    db_session.commit()

    response = client.get(
        "/experiments",
        headers={constants.HEADER_CONFIG_ID: "testing"},
    )

    assert response.status_code == 200, response.content

    experiments = ListExperimentsResponse.model_validate(response.json())
    # experiment3 excluded due to datasource mismatch
    assert len(experiments.items) == 2
    actual1 = experiments.items[1]  # experiment1 is the second item as it's older
    actual2 = experiments.items[0]
    assert actual1.state == ExperimentState.ASSIGNED
    diff = DeepDiff(to_jsonable_python(actual1.design_spec), experiment1.design_spec)
    assert not diff, f"Objects differ:\n{diff.pretty()}"
    assert actual2.state == ExperimentState.COMMITTED
    diff = DeepDiff(to_jsonable_python(actual2.design_spec), experiment2.design_spec)
    assert not diff, f"Objects differ:\n{diff.pretty()}"


def test_get_experiment(db_session: Session):
    new_experiment = make_insertable_experiment(ExperimentState.DESIGNING)
    db_session.add(new_experiment)
    db_session.commit()

    response = client.get(
        f"/experiments/{new_experiment.id!s}",
        headers={constants.HEADER_CONFIG_ID: "testing"},
    )

    assert response.status_code == 200, response.content

    experiment_json = response.json()
    assert experiment_json["datasource_id"] == new_experiment.datasource_id
    assert experiment_json["state"] == new_experiment.state
    actual = DesignSpec.model_validate(experiment_json["design_spec"])
    expected = DesignSpec.model_validate(new_experiment.design_spec)
    diff = DeepDiff(actual, expected)
    assert not diff, f"Objects differ:\n{diff.pretty()}"


def test_get_experiment_assignments(db_session: Session):
    # First insert an experiment with assignments
    experiment = make_insertable_experiment(ExperimentState.COMMITTED)
    experiment_id = str(experiment.id)
    db_session.add(experiment)

    arm1_id = str(experiment.design_spec["arms"][0]["arm_id"])
    arm2_id = str(experiment.design_spec["arms"][1]["arm_id"])
    assignments = [
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
    db_session.add_all(assignments)
    db_session.commit()

    response = client.get(
        f"/experiments/{experiment_id}/assignments",
        headers={constants.HEADER_CONFIG_ID: "testing"},
    )

    # Verify response
    assert response.status_code == 200
    data = response.json()

    # Check the response structure
    assert data["experiment_id"] == experiment_id
    assert data["sample_size"] == experiment.assign_summary["sample_size"]
    assert data["balance_check"] == experiment.assign_summary["balance_check"]

    # Check assignments
    assignments = data["assignments"]
    assert len(assignments) == 2

    # Verify first assignment
    assert assignments[0]["participant_id"] == "p1"
    assert assignments[0]["arm_id"] == arm1_id
    assert assignments[0]["arm_name"] == "control"
    assert len(assignments[0]["strata"]) == 1
    assert assignments[0]["strata"][0]["field_name"] == "gender"
    assert assignments[0]["strata"][0]["strata_value"] == "F"

    # Verify second assignment
    assert assignments[1]["participant_id"] == "p2"
    assert assignments[1]["arm_id"] == arm2_id
    assert assignments[1]["arm_name"] == "treatment"
    assert len(assignments[1]["strata"]) == 1
    assert assignments[1]["strata"][0]["field_name"] == "gender"
    assert assignments[1]["strata"][0]["strata_value"] == "M"


def test_get_experiment_assignments_not_found():
    """Test getting assignments for a non-existent experiment."""
    response = client.get(
        f"/experiments/{uuid.uuid4()}/assignments",
        headers={constants.HEADER_CONFIG_ID: "testing"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment not found"


def test_get_experiment_assignments_wrong_datasource(db_session: Session):
    # Create experiment in one datasource
    experiment = make_insertable_experiment(ExperimentState.COMMITTED)
    db_session.add(experiment)
    db_session.commit()

    # Try to get it from another datasource
    response = client.get(
        f"/experiments/{experiment.id!s}/assignments",
        headers={constants.HEADER_CONFIG_ID: "testing-inline-schema"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment not found"


def make_experiment_with_assignments(db_session):
    # First insert an experiment with assignments
    experiment = make_insertable_experiment(ExperimentState.COMMITTED)
    db_session.add(experiment)

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


def test_experiment_assignments_to_csv_generator(db_session: Session):
    experiment = make_experiment_with_assignments(db_session)

    (arm1_id, arm2_id) = experiment.get_arm_ids()
    (arm1_name, arm2_name) = experiment.get_arm_names()
    batches = list(experiment_assignments_to_csv_generator(experiment)())
    assert len(batches) == 1
    rows = batches[0].splitlines(keepends=True)
    assert rows[0] == "participant_id,arm_id,arm_name,gender,score\r\n"
    assert rows[1] == f"p1,{arm1_id},{arm1_name},F,1.1\r\n"
    assert rows[2] == f'p2,{arm2_id},{arm2_name},M,"esc,aped"\r\n'


def test_get_experiment_assignments_as_csv(db_session: Session):
    experiment = make_experiment_with_assignments(db_session)

    response = client.get(
        f"/experiments/{experiment.id!s}/assignments/csv",
        headers={constants.HEADER_CONFIG_ID: "testing"},
    )
    assert response.status_code == 200
    assert (
        f"experiment_{experiment.id}_assignments.csv"
        in response.headers["Content-Disposition"]
    )
    # Now verify the contents
    rows = response.text.splitlines(keepends=True)
    (arm1_id, arm2_id) = experiment.get_arm_ids()
    (arm1_name, arm2_name) = experiment.get_arm_names()
    assert len(rows) == 3
    assert rows[0] == "participant_id,arm_id,arm_name,gender,score\r\n"
    assert rows[1] == f"p1,{arm1_id},{arm1_name},F,1.1\r\n"
    assert rows[2] == f'p2,{arm2_id},{arm2_name},M,"esc,aped"\r\n'


def test_experiment_sql():
    pg_sql = str(
        CreateTable(ArmAssignment.__table__).compile(dialect=postgresql.dialect())
    )
    assert "arm_id UUID NOT NULL" in pg_sql
    assert "strata JSONB NOT NULL" in pg_sql
    sqlite_sql = str(
        CreateTable(ArmAssignment.__table__).compile(dialect=sqlite.dialect())
    )
    assert "arm_id CHAR(32) NOT NULL" in sqlite_sql
    assert "strata JSON NOT NULL" in sqlite_sql
