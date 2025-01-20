from dataclasses import dataclass
import dataclasses
from decimal import Decimal
from typing import Any
import pytest
import pandas as pd
import numpy as np
from sqlalchemy import DECIMAL, Column, Float, Integer, MetaData, String, Table
from xngin.stats.assignment import assign_treatment
from xngin.apiserver.api_types import Assignment


@dataclass
class Row:
    """Simulate the bits of a sqlalchemy Row that we need here."""

    id: int
    age: float
    income: float
    gender: str
    region: str
    skewed: int
    income_dec: Decimal
    is_male: bool

    def _asdict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@pytest.fixture
def sample_table():
    metadata_obj = MetaData()
    return Table(
        "table_name",
        metadata_obj,
        Column("id", String, primary_key=True, autoincrement=False),
        Column("age", Integer),
        Column("income", Float),
        Column("gender", String),
        Column("region", String),
        Column("skewed", Float),
        Column("income_dec", DECIMAL),
    )


@pytest.fixture
def sample_data():
    return pd.DataFrame(make_sample_data_dict())


def make_sample_data_dict(n=1000):
    np.random.seed(42)
    data = {
        "id": range(n),
        "age": np.round(np.random.normal(30, 5, n), 0),
        "income": np.round(np.float64(np.random.lognormal(10, 1, n)), 0),
        "gender": np.random.choice(["M", "F"], n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "skewed": np.random.permutation(
            np.concatenate((np.repeat([1], n * 0.9), np.repeat([0], n * 0.1)))
        ),
    }
    data["income_dec"] = [Decimal(i).quantize(Decimal("1")) for i in data["income"]]
    data["is_male"] = [g == "M" for g in data["gender"]]
    return data


def test_assign_treatment(sample_table, sample_data):
    rows = [Row(**row) for row in sample_data.to_dict("records")]

    result = assign_treatment(
        sa_table=sample_table,
        data=rows,
        stratum_cols=["gender", "region"],
        id_col="id",
        arm_names=["control", "treatment"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        random_state=42,
    )

    assert result.balance_check.f_statistic == pytest.approx(0.006156735)
    assert result.balance_check.p_value == pytest.approx(0.99992466)
    assert result.balance_check.balance_ok
    assert str(result.experiment_id) == "b767716b-f388-4cd9-a18a-08c4916ce26f"
    assert result.sample_size == len(sample_data)
    assert (
        result.sample_size
        == result.balance_check.numerator_df + result.balance_check.denominator_df + 1
    )
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


def test_assign_treatment_multiple_arms(sample_table, sample_data):
    rows = [Row(**row) for row in sample_data.to_dict("records")]

    result = assign_treatment(
        sa_table=sample_table,
        data=rows,
        stratum_cols=["gender", "region"],
        id_col="id",
        arm_names=["control", "treatment_a", "treatment_b"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
    )

    # Check that assignments is a list
    assert isinstance(result.assignments, list)
    # Check that the list contains ExperimentParticipant objects
    assert all(
        isinstance(participant, Assignment) for participant in result.assignments
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


def test_assign_treatment_reproducibility(sample_table, sample_data):
    rows = [Row(**row) for row in sample_data.to_dict("records")]

    result1 = assign_treatment(
        sa_table=sample_table,
        data=rows,
        stratum_cols=["gender", "region"],
        id_col="id",
        arm_names=["control", "treatment"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        random_state=42,
    )

    result2 = assign_treatment(
        sa_table=sample_table,
        data=rows,
        stratum_cols=["gender", "region"],
        id_col="id",
        arm_names=["control", "treatment"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
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


def test_assign_treatment_with_missing_values(sample_table, sample_data):
    # Add some missing values
    sample_data.loc[sample_data.index[:100], "income"] = np.nan
    rows = [Row(**row) for row in sample_data.to_dict("records")]

    result = assign_treatment(
        sa_table=sample_table,
        data=rows,
        stratum_cols=["gender", "region"],
        id_col="id",
        arm_names=["control", "treatment"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
    )

    assert result.sample_size == len(sample_data)
    assert result.id_col == "id"
    # Check that treatment assignments are not None or NaN
    assert all(
        participant.treatment_assignment is not None
        for participant in result.assignments
    )


def test_assign_treatment_with_obj_columns_inferred(sample_table, sample_data):
    # Extend samples with values simulating incorrect type inference as objects.
    @dataclass
    class ExtendedRow(Row):
        object1: Any
        object2: Any
        object3: Any

    n = len(sample_data)
    # Test numeric types mistakenly typed as objects
    sample_data = sample_data.assign(
        object1=[2**32] * (n - 1) + [2],
        # If not converted, causes SyntaxError due to floating point numbers
        # converted to dummy variables with bad column names
        object2=np.concatenate([[None] * (n - 100), np.random.uniform(size=100)]),
        # If not converted, will cause a recursion error
        object3=np.random.uniform(size=n),
    ).astype({"object1": "O", "object2": "O", "object3": "O"})
    rows = [ExtendedRow(**row) for row in sample_data.to_dict("records")]

    result = assign_treatment(
        sa_table=sample_table,
        data=rows,
        stratum_cols=["gender", "region", "object2", "object3"],
        id_col="id",
        arm_names=["control", "treatment"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
    )

    assert result.sample_size == len(sample_data)
    assert result.id_col == "id"
    assert pd.isna(result.balance_check.p_value) is False
    assert pd.isna(result.balance_check.f_statistic) is False
    # Check that treatment assignments are not None or NaN
    assert all(
        participant.treatment_assignment is not None
        for participant in result.assignments
    )


MAX_SAFE_INTEGER = (1 << 53) - 1  # 9007199254740991


def test_assign_treatment_with_integers_as_floats_for_unique_id(
    sample_table, sample_data
):
    def assign(data):
        rows = [Row(**row) for row in data.to_dict("records")]
        return assign_treatment(
            sa_table=sample_table,
            data=rows,
            stratum_cols=["gender", "region"],
            id_col="id",
            arm_names=["control", "treatment"],
            experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
            random_state=42,
        )

    # We should be able to handle Decimals (e.g. from psycopg2 with redshift numerics).
    sample_data["id"] = sample_data["id"].apply(Decimal)
    result = assign(sample_data)
    assert result.balance_check.f_statistic == pytest.approx(0.006156735)
    assert result.balance_check.p_value == pytest.approx(0.99992466)
    json = result.model_dump()
    assert json["assignments"][0]["participant_id"] == "0"
    assert json["assignments"][1]["participant_id"] == "1"

    # We should be able to support bigger than signed int64s
    sample_data.loc[0, "id"] = Decimal(MAX_SAFE_INTEGER + 2)
    result = assign(sample_data)
    # Ensure the id string is rendered properly. (index=891 since participant ids are ordered lexicogrpahically)
    json_str = result.model_dump_json()
    assert '"participant_id":"9007199254740993"' in json_str

    # We should be able to support very big negatives as well
    sample_data.loc[0, "id"] = -MAX_SAFE_INTEGER - 2
    result = assign(sample_data)
    json = result.model_dump()
    assert json["assignments"][0]["participant_id"] == "-9007199254740993"


def test_decimal_and_bool_strata_are_rendered_correctly(sample_table, sample_data):
    rows = [Row(**row) for row in sample_data.to_dict("records")]

    result = assign_treatment(
        sa_table=sample_table,
        data=rows,
        stratum_cols=["income_dec", "is_male"],
        id_col="id",
        arm_names=["control", "treatment"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        random_state=42,
    )

    for p in result.assignments:
        json = p.model_dump()
        # we rounded the Decimal to an int, so shouldn't see the decimal point
        assert "." not in json["strata"][0]["strata_value"], json
        assert json["strata"][1]["strata_value"] in ("True", "False"), json


def test_with_nans_that_would_break_stochatreat_without_preprocessing(sample_table):
    local_data = make_sample_data_dict(20)
    # Replace entries with NaN such that the grouping into strata causes stochatreat to raise a
    # ValueError as it internally uses df.groupby(..., dropna=True), causing the count of
    # synthetic rows created to be off.
    local_data["age"] = [None, 2] + [1, 2] * 9
    local_data = pd.DataFrame(local_data)
    rows = [Row(**row) for row in local_data.to_dict("records")]
    result = assign_treatment(
        sa_table=sample_table,
        data=rows,
        stratum_cols=["age"],
        id_col="id",
        arm_names=["control", "treatment"],
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        random_state=42,
    )
    # But we still expect success since internally we'll preprocess the data to handle NaNs.
    assert result.balance_check.f_statistic > 0
    assert result.balance_check.p_value > 0
    assert result.balance_check.balance_ok is False
    assert result.sample_size == len(local_data)
    assert (
        result.sample_size
        == result.balance_check.numerator_df + result.balance_check.denominator_df + 1
    )
    assert all(len(participant.strata) == 1 for participant in result.assignments)
