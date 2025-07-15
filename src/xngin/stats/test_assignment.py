import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.random import MT19937, RandomState
from sqlalchemy import DECIMAL, Boolean, Column, Float, Integer, MetaData, String, Table

from xngin.apiserver.models import tables
from xngin.apiserver.routers.assignment_adapters import (
    assign_treatment,
    simple_random_assignment,
)
from xngin.apiserver.routers.common_api_types import Arm, Assignment, Strata


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
    single_value: int

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
        Column("is_male", Boolean),
        Column("single_value", Integer),
    )


def make_sample_data_dict(n=1000):
    rs = RandomState(MT19937())
    rs.seed(42)
    data = {
        "id": range(n),
        "age": np.round(rs.normal(30, 5, n), 0),
        "income": np.round(np.float64(rs.lognormal(10, 1, n)), 0),
        "gender": rs.choice(["M", "F"], n),
        "region": rs.choice(["North", "South", "East", "West"], n),
        "skewed": rs.permutation(
            np.concatenate((np.repeat([1], int(n * 0.9)), np.repeat([0], int(n * 0.1))))
        ),
    }
    data["income_dec"] = [Decimal(i).quantize(Decimal(1)) for i in data["income"]]
    data["is_male"] = [g == "M" for g in data["gender"]]
    data["single_value"] = [1] * n
    return data


@pytest.fixture(name="sample_data")
def fixture_sample_data():
    """Helper that turns a python dict into a pandas DataFrame."""
    return pd.DataFrame(make_sample_data_dict())


@pytest.fixture(name="sample_rows")
def fixture_sample_rows(sample_data):
    """Helper that turns a pandas DataFrame into a list of SQLAlchemy-like Row objects."""
    return [Row(**row) for row in sample_data.to_dict("records")]


def make_arms(names: list[str]):
    return [Arm(arm_id=tables.arm_id_factory(), arm_name=name) for name in names]


def test_assign_treatment(sample_table, sample_rows):
    arms = make_arms(["control", "treatment"])
    result = assign_treatment(
        sa_table=sample_table,
        data=sample_rows,
        stratum_cols=["region", "gender"],
        id_col="id",
        arms=arms,
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        random_state=42,
        stratum_id_name="stratum_id",
    )

    assert result.balance_check is not None
    assert result.balance_check.f_statistic == pytest.approx(0.006156735)
    assert result.balance_check.p_value == pytest.approx(0.99992466)
    assert result.balance_check.balance_ok
    assert result.experiment_id == "b767716b-f388-4cd9-a18a-08c4916ce26f"
    assert result.sample_size == len(sample_rows)
    assert (
        result.sample_size
        == result.balance_check.numerator_df + result.balance_check.denominator_df + 1
    )
    assert result.unique_id_field == "id"
    assert isinstance(result.assignments, list)
    assert len(set([x.participant_id for x in result.assignments])) == len(
        result.assignments
    )
    for participant in result.assignments:
        assert participant.strata is not None
        assert len(participant.strata) == 3
    assert all(
        participant.arm_name in {"control", "treatment"}
        for participant in result.assignments
    )
    assert result.assignments[0].arm_name == "control"
    assert result.assignments[1].arm_name == "control"
    assert result.assignments[2].arm_name == "treatment"
    assert result.assignments[3].arm_name == "control"
    assert result.assignments[4].arm_name == "treatment"
    assert result.assignments[5].arm_name == "control"
    assert result.assignments[6].arm_name == "treatment"
    assert result.assignments[7].arm_name == "treatment"
    assert result.assignments[8].arm_name == "treatment"
    assert result.assignments[9].arm_name == "treatment"
    # Verify strata sorted in order
    assert result.assignments[0].strata is not None
    assert [s.field_name for s in result.assignments[0].strata] == [
        "gender",
        "region",
        "stratum_id",
    ]
    # There should only be 8 distinct stratum_ids (gender x region)
    strata_ids = [s.strata[2].strata_value for s in result.assignments if s.strata is not None]
    assert set(strata_ids) == {str(x) for x in range(8)}
    # Count occurrences of each unique strata tuple
    strata_counts: defaultdict[tuple, int] = defaultdict(int)
    for participant in result.assignments:
        assert participant.strata is not None
        stratum = tuple(s.strata_value for s in participant.strata)
        strata_counts[stratum] += 1
    assert len(strata_counts) == 8, strata_counts
    # Verify that each stratum has at least one participant
    assert all(count > 0 for count in strata_counts.values())


def test_assign_treatment_multiple_arms(sample_table, sample_rows):
    arms = make_arms(["control", "treatment_a", "treatment_b"])
    result = assign_treatment(
        sa_table=sample_table,
        data=sample_rows,
        stratum_cols=["gender", "region"],
        id_col="id",
        arms=arms,
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        random_state=42,
    )

    # Check that assignments is a list
    assert isinstance(result.assignments, list)
    # Check that the list contains ExperimentParticipant objects
    assert all(
        isinstance(participant, Assignment) for participant in result.assignments
    )
    # Check that the treatment assignments are valid (not None or NaN)
    assert all(participant.arm_name is not None for participant in result.assignments)
    # Check that the treatment assignments are valid
    assert len(set(participant.arm_name for participant in result.assignments)) == 3
    assert result.sample_size == len(sample_rows)
    assert result.unique_id_field == "id"


def test_assign_treatment_reproducibility(sample_table, sample_rows):
    result1 = assign_treatment(
        sa_table=sample_table,
        data=sample_rows,
        stratum_cols=["gender", "region"],
        id_col="id",
        arms=make_arms(["control", "treatment"]),
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        random_state=42,
    )

    result2 = assign_treatment(
        sa_table=sample_table,
        data=sample_rows,
        stratum_cols=["gender", "region"],
        id_col="id",
        arms=make_arms(["control", "treatment"]),
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        random_state=42,
    )

    # Check that both results have the same assignments
    assert len(result1.assignments) == len(result2.assignments)
    assert result1.unique_id_field == result2.unique_id_field

    # Check that the treatment assignments are the same
    for p1, p2 in zip(result1.assignments, result2.assignments, strict=False):
        assert p1.arm_name == p2.arm_name
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
        arms=make_arms(["control", "treatment"]),
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
    )

    assert result.sample_size == len(sample_data)
    assert result.unique_id_field == "id"
    # Check that treatment assignments are not None or NaN
    assert all(participant.arm_name is not None for participant in result.assignments)


def test_assign_treatment_with_obj_columns_inferred(sample_table, sample_data):
    # Extend samples with values simulating incorrect type inference as objects.
    @dataclass
    class ExtendedRow(Row):
        object1: Any
        object2: Any
        object3: Any

    n = len(sample_data)
    # Test numeric types mistakenly typed as objects
    rng = np.random.default_rng()
    sample_data = sample_data.assign(
        object1=[2**32] * (n - 1) + [2],
        # If not converted, causes SyntaxError due to floating point numbers
        # converted to dummy variables with bad column names
        object2=np.concatenate([[None] * (n - 100), rng.uniform(size=100)]),
        # If not converted, will cause a recursion error
        object3=rng.uniform(size=n),
    ).astype({"object1": "O", "object2": "O", "object3": "O"})
    rows = [ExtendedRow(**row) for row in sample_data.to_dict("records")]

    result = assign_treatment(
        sa_table=sample_table,
        data=rows,
        stratum_cols=["gender", "region", "object2", "object3"],
        id_col="id",
        arms=make_arms(["control", "treatment"]),
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
    )

    assert result.sample_size == len(sample_data)
    assert result.unique_id_field == "id"
    assert result.balance_check is not None
    assert pd.isna(result.balance_check.p_value) is False
    assert pd.isna(result.balance_check.f_statistic) is False
    # Check that treatment assignments are not None or NaN
    assert all(participant.arm_name is not None for participant in result.assignments)


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
            arms=make_arms(["control", "treatment"]),
            experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
            random_state=42,
        )

    # We should be able to handle Decimals (e.g. from psycopg2 with redshift numerics).
    sample_data["id"] = sample_data["id"].apply(Decimal)
    result = assign(sample_data)
    assert result.balance_check is not None
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


def test_decimal_and_bool_strata_are_rendered_correctly(sample_table, sample_rows):
    result = assign_treatment(
        sa_table=sample_table,
        data=sample_rows,
        stratum_cols=["income_dec", "is_male"],
        id_col="id",
        arms=make_arms(["control", "treatment"]),
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        random_state=42,
    )

    for p in result.assignments:
        json = p.model_dump()
        # we rounded the Decimal to an int, so shouldn't see the decimal point
        assert "." not in json["strata"][0]["strata_value"], json
        assert json["strata"][1]["strata_value"] in {"True", "False"}, json


def test_with_nans_that_would_break_stochatreat_without_preprocessing(sample_table):
    local_data = make_sample_data_dict(20)
    # Replace entries with NaN such that the grouping into strata causes stochatreat to raise a
    # ValueError as it internally uses df.groupby(..., dropna=True), causing the count of
    # synthetic rows created to be off.
    local_data["gender"] = [None, "F"] + ["M", "F"] * 9
    local_data = pd.DataFrame(local_data)
    rows = [Row(**row) for row in local_data.to_dict("records")]
    result = assign_treatment(
        sa_table=sample_table,
        data=rows,
        stratum_cols=["gender"],
        id_col="id",
        arms=make_arms(["control", "treatment"]),
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        random_state=42,
    )
    # But we still expect success since internally we'll preprocess the data to handle NaNs.
    assert result.balance_check is not None
    assert result.balance_check.f_statistic > 0
    assert result.balance_check.p_value > 0
    assert result.balance_check.balance_ok is True
    assert result.sample_size == len(local_data)
    assert (
        result.sample_size
        == result.balance_check.numerator_df + result.balance_check.denominator_df + 1
    )
    assert all(
        participant.strata is not None and len(participant.strata) == 1
        for participant in result.assignments
    )


def test_simple_random_assignment(sample_rows):
    n = len(sample_rows)
    assignments = simple_random_assignment(
        pd.DataFrame(sample_rows), make_arms(["A", "B"]), random_state=42
    )
    assert len(assignments) == n
    assert assignments.count(0) == n // 2
    assert assignments.count(1) == n // 2

    assignments = simple_random_assignment(
        pd.DataFrame(sample_rows), make_arms(["A", "B", "C"]), random_state=42
    )
    assert len(assignments) == n
    assert assignments.count(0) in {n // 3, n // 3 + 1}
    assert assignments.count(1) in {n // 3, n // 3 + 1}
    assert assignments.count(2) in {n // 3, n // 3 + 1}


def test_assign_treatment_with_no_stratification(sample_table, sample_rows):
    # random_state=None since the counts of each arm should always be equal regardless.
    result = assign_treatment(
        sa_table=sample_table,
        data=sample_rows,
        stratum_cols=[],
        id_col="id",
        arms=make_arms(["control", "treatment"]),
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        random_state=None,
    )
    assert result.balance_check is None
    arm_counts: defaultdict[str, int] = defaultdict(int)
    # There should be no strata
    for participant in result.assignments:
        arm_counts[participant.arm_name] += 1
        assert participant.strata is None
    # The number of assignments per arm should be equal
    assert arm_counts["control"] == arm_counts["treatment"]
    assert arm_counts["control"] == len(result.assignments) // 2


def test_assign_treatment_with_no_valid_strata(sample_table, sample_rows):
    result = assign_treatment(
        sa_table=sample_table,
        data=sample_rows,
        stratum_cols=["single_value"],
        id_col="id",
        arms=make_arms(["control", "treatment"]),
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        random_state=42,
        stratum_id_name="stratum_id",
    )
    assert result.balance_check is None
    # In this case, we still output the strata column, even though it's all the same value.
    arm_counts: defaultdict[str, int] = defaultdict(int)
    for participant in result.assignments:
        arm_counts[participant.arm_name] += 1
        assert participant.strata == [
            Strata(field_name="single_value", strata_value="1")
        ]
    # And since we used our simple random assignment, the arm lengths should be equal.
    assert arm_counts["control"] == arm_counts["treatment"]
    assert arm_counts["control"] == len(result.assignments) // 2
