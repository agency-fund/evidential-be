import uuid
from datetime import datetime, timedelta, UTC

import pytest
from deepdiff import DeepDiff
from fastapi.testclient import TestClient
from pydantic_core import to_jsonable_python
from sqlalchemy import select
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


def make_create_experiment_request(with_uuids: bool = True) -> CreateExperimentRequest:
    # TODO: generate ids in server
    experiment_id = uuid.uuid4() if with_uuids else None
    arm1_id = uuid.uuid4() if with_uuids else None
    arm2_id = uuid.uuid4() if with_uuids else None
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
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC) + timedelta(days=30),
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
def make_insert_experiment(state: ExperimentState, datasource_id="testing"):
    request = make_create_experiment_request()
    return Experiment(
        id=request.design_spec.experiment_id,
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


def test_create_experiment_with_assignment_invalid_design_spec(
    db_session: Session,
):
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
    db_session: Session,
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
    assert assign_summary["balance_check"]["balance_ok"] is True, assign_summary[
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
    (arm1_id, arm2_id) = [arm.arm_id for arm in config.design_spec.arms]
    # Verify database state using the ids in the returned DesignSpec.
    experiment = db_session.scalars(
        select(Experiment).where(Experiment.id == experiment_id)
    ).one()
    assert experiment.state == ExperimentState.ASSIGNED
    assert experiment.datasource_id == "testing"
    assert experiment.start_date == request.design_spec.start_date
    assert experiment.end_date == request.design_spec.end_date

    assignments = db_session.scalars(
        select(ArmAssignment).where(ArmAssignment.experiment_id == experiment_id)
    ).all()
    assert len(assignments) == 100, {
        e.name: getattr(experiment, e.name) for e in Experiment.__table__.columns
    }

    # Check one assignment to see if it looks roughly right
    sample_assignment: ArmAssignment = assignments[0]
    assert sample_assignment.participant_type == "test_participant_type"
    assert sample_assignment.experiment_id == experiment_id
    assert sample_assignment.arm_id in (arm1_id, arm2_id)
    for stratum in sample_assignment.strata:
        assert stratum["field_name"] in {"gender", "is_onboarded"}

    # Check for approximate balance in arm assignment
    num_control = sum(1 for a in assignments if a.arm_id == arm1_id)
    num_treat = sum(1 for a in assignments if a.arm_id == arm2_id)
    assert abs(num_control - num_treat) <= 5  # Allow some wiggle room


@pytest.mark.parametrize(
    "endpoint,initial_state,expected_status,expected_detail",
    [
        ("commit", ExperimentState.ASSIGNED, 204, None),  # Success case
        ("commit", ExperimentState.COMMITTED, 304, None),  # No-op
        ("commit", ExperimentState.DESIGNING, 403, "Invalid state: designing"),
        ("commit", ExperimentState.ABORTED, 403, "Invalid state: aborted"),
        ("abandon", ExperimentState.DESIGNING, 204, None),  # Success case
        ("abandon", ExperimentState.ASSIGNED, 204, None),  # Success case
        ("abandon", ExperimentState.ABANDONED, 304, None),  # No-op
        ("abandon", ExperimentState.COMMITTED, 403, "Invalid state: committed"),
    ],
)
def test_experiment_state_setting(
    db_session: Session, endpoint, initial_state, expected_status, expected_detail
):
    experiment = make_insert_experiment(initial_state)
    db_session.add(experiment)
    db_session.commit()

    response = client.post(
        f"/experiments/{experiment.id!s}/{endpoint}",
        headers={constants.HEADER_CONFIG_ID: "testing"},
    )

    # Verify
    assert response.status_code == expected_status
    # If success case, verify state was updated
    if expected_status == 204:
        expected_state = (
            ExperimentState.ABANDONED
            if endpoint == "abandon"
            else ExperimentState.COMMITTED
        )
        db_session.refresh(experiment)
        assert experiment.state == expected_state
    # If failure case, verify the error message
    if expected_detail:
        assert response.json()["detail"] == expected_detail


def test_list_experiments(db_session: Session):
    experiment1 = make_insert_experiment(
        ExperimentState.ASSIGNED, datasource_id="testing"
    )
    experiment2 = make_insert_experiment(
        ExperimentState.COMMITTED, datasource_id="testing"
    )
    experiment3 = make_insert_experiment(
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
    new_experiment = make_insert_experiment(ExperimentState.DESIGNING)
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
    experiment = make_insert_experiment(ExperimentState.COMMITTED)
    db_session.add(experiment)

    arm1_id = uuid.UUID(experiment.design_spec["arms"][0]["arm_id"])
    arm2_id = uuid.UUID(experiment.design_spec["arms"][1]["arm_id"])
    assignments = [
        ArmAssignment(
            experiment_id=experiment.id,
            participant_type="test_participant_type",
            participant_id="p1",
            arm_id=arm1_id,
            strata=[{"field_name": "gender", "strata_value": "F"}],
        ),
        ArmAssignment(
            experiment_id=experiment.id,
            participant_type="test_participant_type",
            participant_id="p2",
            arm_id=arm2_id,
            strata=[{"field_name": "gender", "strata_value": "M"}],
        ),
    ]
    db_session.add_all(assignments)
    db_session.commit()

    response = client.get(
        f"/experiments/{experiment.id!s}/assignments",
        headers={constants.HEADER_CONFIG_ID: "testing"},
    )

    # Verify response
    assert response.status_code == 200
    data = response.json()

    # Check the response structure
    assert data["experiment_id"] == str(experiment.id)
    assert data["sample_size"] == experiment.assign_summary["sample_size"]
    assert data["balance_check"] == experiment.assign_summary["balance_check"]

    # Check assignments
    assignments = data["assignments"]
    assert len(assignments) == 2

    # Verify first assignment
    assert assignments[0]["participant_id"] == "p1"
    assert assignments[0]["arm_id"] == str(arm1_id)
    assert assignments[0]["arm_name"] == "control"
    assert len(assignments[0]["strata"]) == 1
    assert assignments[0]["strata"][0]["field_name"] == "gender"
    assert assignments[0]["strata"][0]["strata_value"] == "F"

    # Verify second assignment
    assert assignments[1]["participant_id"] == "p2"
    assert assignments[1]["arm_id"] == str(arm2_id)
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
    experiment = make_insert_experiment(ExperimentState.COMMITTED)
    db_session.add(experiment)
    db_session.commit()

    # Try to get it from another datasource
    response = client.get(
        f"/experiments/{experiment.id!s}/assignments",
        headers={constants.HEADER_CONFIG_ID: "testing-inline-schema"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment not found"
