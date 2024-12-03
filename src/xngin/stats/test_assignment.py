import pytest
import pandas as pd
import numpy as np
from xngin.stats.assignment import assign_treatment
from xngin.apiserver.api_types import ExperimentParticipant


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 1000
    data = {
        "id": range(n),
        "age": np.round(np.random.normal(30, 5, n), 0),
        "income": np.round(np.float64(np.random.lognormal(10, 1, n)), 0),
        "gender": np.random.choice(["M", "F"], n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
    }
    return pd.DataFrame(data)


def test_assign_treatment(sample_data):
    result = assign_treatment(
        data=sample_data,
        stratum_cols=["gender", "region"],
        metric_cols=["age", "income"],
        id_col="id",
        arm_names=["control", "treatment"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        description="Test experiment",
        random_state=42,
    )

    assert result.f_statistic is not None
    assert result.p_value is not None
    assert result.balance_ok is not None
    assert str(result.experiment_id) == "b767716b-f388-4cd9-a18a-08c4916ce26f"
    assert result.description == "Test experiment"
    assert result.sample_size == len(sample_data)
    assert result.id_col == "id"
    assert isinstance(result.assignments, list)
    # TODO(roboton): Fix this test
    # assert 'treat' in result.assignments.columns


def test_assign_treatment_multiple_arms(sample_data):
    result = assign_treatment(
        data=sample_data,
        stratum_cols=["gender", "region"],
        metric_cols=["age", "income"],
        id_col="id",
        arm_names=["control", "treatment_a", "treatment_b"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        description="Test multi-arm experiment",
    )

    # Check that assignments is a list
    assert isinstance(result.assignments, list)
    # Check that the list contains ExperimentParticipant objects
    assert all(
        isinstance(participant, ExperimentParticipant)
        for participant in result.assignments
    )
    # Check that the treatment assignments are valid (not None or NaN)
    assert all(
        participant.treatment_assignment is not None
        for participant in result.assignments
    )
    # Check that the treatment assignments are valid
    assert (
        len(set(participant.treatment_assignment for participant in result.assignments))
        == 3
    )
    assert result.sample_size == len(sample_data)
    assert result.id_col == "id"


def test_assign_treatment_reproducibility(sample_data):
    result1 = assign_treatment(
        data=sample_data,
        stratum_cols=["gender", "region"],
        metric_cols=["age", "income"],
        id_col="id",
        arm_names=["control", "treatment"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        description="Test reproducibility",
        random_state=42,
    )

    result2 = assign_treatment(
        data=sample_data,
        stratum_cols=["gender", "region"],
        metric_cols=["age", "income"],
        id_col="id",
        arm_names=["control", "treatment"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        description="Test reproducibility",
        random_state=42,
    )

    # Check that both results have the same assignments
    assert len(result1.assignments) == len(result2.assignments)
    assert result1.id_col == result2.id_col

    # Check that the treatment assignments are the same
    for p1, p2 in zip(result1.assignments, result2.assignments, strict=False):
        assert p1.treatment_assignment == p2.treatment_assignment
        assert (
            p1.participant_id == p2.participant_id
        )  # Assuming id is a unique identifier for participants
        assert p1.strata == p2.strata  # Check if strata are equal


def test_assign_treatment_with_missing_values(sample_data):
    # Add some missing values
    sample_data.loc[sample_data.index[:100], "income"] = np.nan

    result = assign_treatment(
        data=sample_data,
        stratum_cols=["gender", "region"],
        metric_cols=["age", "income"],
        id_col="id",
        arm_names=["control", "treatment"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        description="Test with missing values",
    )

    assert result.sample_size == len(sample_data)
    assert result.id_col == "id"
    # Check that treatment assignments are not None or NaN
    assert all(
        participant.treatment_assignment is not None
        for participant in result.assignments
    )
