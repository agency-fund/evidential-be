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
        "skewed": np.random.permutation(
            np.concatenate((np.repeat([1], 900), np.repeat([0], 100)))
        ),
    }
    return pd.DataFrame(data)


def test_assign_treatment(sample_data):
    result = assign_treatment(
        data=sample_data,
        stratum_cols=["gender", "region"],
        id_col="id",
        arm_names=["control", "treatment"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        description="Test experiment",
        random_state=42,
    )

    assert result.f_statistic == pytest.approx(0.006156735)
    assert result.p_value == pytest.approx(0.99992466)
    assert result.balance_ok
    assert str(result.experiment_id) == "b767716b-f388-4cd9-a18a-08c4916ce26f"
    assert result.description == "Test experiment"
    assert result.sample_size == len(sample_data)
    assert result.sample_size == result.numerator_df + result.denominator_df + 1
    assert result.id_col == "id"
    assert isinstance(result.assignments, list)
    assert len(set([x.participant_id for x in result.assignments])) == len(
        result.assignments
    )
    assert all(len(participant.strata) == 2 for participant in result.assignments)
    assert all(
        participant.treatment_assignment in ["control", "treatment"]
        for participant in result.assignments
    )
    assert result.assignments[0].treatment_assignment == "control"
    assert result.assignments[1].treatment_assignment == "control"
    assert result.assignments[2].treatment_assignment == "treatment"
    assert result.assignments[3].treatment_assignment == "control"
    assert result.assignments[4].treatment_assignment == "treatment"
    assert result.assignments[5].treatment_assignment == "control"
    assert result.assignments[6].treatment_assignment == "treatment"
    assert result.assignments[7].treatment_assignment == "treatment"
    assert result.assignments[8].treatment_assignment == "treatment"
    assert result.assignments[9].treatment_assignment == "treatment"


def test_assign_treatment_multiple_arms(sample_data):
    result = assign_treatment(
        data=sample_data,
        stratum_cols=["gender", "region"],
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
        id_col="id",
        arm_names=["control", "treatment"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        description="Test reproducibility",
        random_state=42,
    )

    result2 = assign_treatment(
        data=sample_data,
        stratum_cols=["gender", "region"],
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


def test_assign_treatment_with_obj_columns_inferred(sample_data):
    # Extend samples with values simulating incorrect type inference as objects.
    n = len(sample_data)
    sample_data = sample_data.assign(
        object1=[2**32] * (n - 1) + [2],
        # If not converted, causes SyntaxError due to floating point numbers
        # converted to dummy variables with bad column names
        object2=np.concatenate(([None] * (n - 100), np.random.uniform(size=100))),
        # If not converted, will cause a recursion error
        object3=np.random.uniform(size=n),
    ).astype({"object1": "O", "object2": "O", "object3": "O"})

    result = assign_treatment(
        data=sample_data,
        stratum_cols=["gender", "region", "object2", "object3"],
        id_col="id",
        arm_names=["control", "treatment"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        description="Test with numeric types mistakenly typed as objects",
    )

    assert result.sample_size == len(sample_data)
    assert result.id_col == "id"
    assert pd.isna(result.p_value) is False
    assert pd.isna(result.f_statistic) is False
    # Check that treatment assignments are not None or NaN
    assert all(
        participant.treatment_assignment is not None
        for participant in result.assignments
    )


def test_assign_treatment_with_integers_as_floats_for_unique_id(sample_data):
    def assign(data):
        return assign_treatment(
            data=data,
            stratum_cols=["gender", "region"],
            id_col="id",
            arm_names=["control", "treatment"],
            experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
            description="Test integers_as_floats_for_unique_id",
            random_state=42,
        )

    # When it's a float that can be converted to an int, we're still ok
    sample_data["id"] = sample_data["id"].apply(float)
    result = assign(sample_data)
    assert result.f_statistic == pytest.approx(0.006156735)
    assert result.p_value == pytest.approx(0.99992466)
    json = result.model_dump()
    assert json["assignments"][0]["participant_id"] == "0"
    assert json["assignments"][1]["participant_id"] == "1"

    # But if it's too large we raise an exception
    with pytest.raises(ValueError):
        sample_data.loc[0, "id"] = float(2**53) + 2.0
        result = assign(sample_data)

    # Similarly if it's negative, raise an exception
    with pytest.raises(ValueError):
        sample_data.loc[0, "id"] = -1.0
        result = assign(sample_data)
