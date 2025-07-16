"""Test assignment adapter conversion functions."""

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
    _make_balance_check,  # noqa: PLC2701
    assign_treatment,
)
from xngin.apiserver.routers.common_api_types import (
    Arm,
    Assignment,
    BalanceCheck,
    Strata,
)
from xngin.stats.balance import BalanceResult


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
            np.concatenate((np.repeat([1], int(n * 0.9)), np.repeat([0], n - int(n * 0.9))))
        ),
        "single_value": [1] * n,
    }
    data["income_dec"] = [Decimal(i).quantize(Decimal(1)) for i in data["income"]]
    data["is_male"] = [g == "M" for g in data["gender"]]
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


# Tests for the conversion function
def test_make_balance_check():
    """Test conversion from BalanceResult to BalanceCheck."""
    # Test with None input
    assert _make_balance_check(None, 0.5) is None

    # Test with actual BalanceResult
    balance_result = BalanceResult(
        f_statistic=1.234567890123456,
        f_pvalue=0.876543210987654,
        model_summary="test summary",
        is_balanced=True,
        numerator_df=5.0,
        denominator_df=100.0,
    )
    balance_check = _make_balance_check(balance_result, 0.5)

    assert isinstance(balance_check, BalanceCheck)
    assert balance_check.f_statistic == pytest.approx(1.234567890, abs=1e-9)
    assert balance_check.p_value == pytest.approx(0.876543211, abs=1e-9)
    assert balance_check.balance_ok is True
    assert balance_check.numerator_df == 5
    assert balance_check.denominator_df == 100


def test_make_balance_check_not_balanced():
    """Test conversion when balance is not OK."""
    balance_result = BalanceResult(
        f_statistic=2.5,
        f_pvalue=0.3,  # Less than threshold
        model_summary="test summary",
        is_balanced=False,
        numerator_df=3.0,
        denominator_df=50.0,
    )

    balance_check = _make_balance_check(balance_result, 0.5)

    assert balance_check.balance_ok is False
    assert balance_check.f_statistic == 2.5
    assert balance_check.p_value == 0.3
    assert balance_check.numerator_df == 3
    assert balance_check.denominator_df == 50


def test_assign_adapter_creates_proper_assign_response(sample_table, sample_rows):
    """Test that assign_treatment creates a proper AssignResponse with all required fields."""
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

    # Test AssignResponse structure
    assert result.balance_check is not None
    assert result.balance_check.f_statistic == pytest.approx(0.006156735)
    assert result.balance_check.p_value == pytest.approx(0.99992466)
    assert result.balance_check.balance_ok
    assert result.experiment_id == "b767716b-f388-4cd9-a18a-08c4916ce26f"
    assert result.sample_size == len(sample_rows)
    assert result.unique_id_field == "id"
    assert isinstance(result.assignments, list)
    assert len(set([x.participant_id for x in result.assignments])) == len(result.assignments)


def test_assign_adapter_multiple_arms(sample_table, sample_rows):
    """Test assignment with multiple arms."""
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

    # Check that assignments is a list of Assignment objects
    assert isinstance(result.assignments, list)
    assert all(isinstance(participant, Assignment) for participant in result.assignments)
    # Check that the treatment assignments are valid (not None or NaN)
    assert all(participant.arm_name is not None for participant in result.assignments)
    # Check that the treatment assignments are valid
    assert len(set(participant.arm_name for participant in result.assignments)) == 3
    assert result.sample_size == len(sample_rows)
    assert result.unique_id_field == "id"


MAX_SAFE_INTEGER = (1 << 53) - 1  # 9007199254740991


def test_assign_adapter_with_large_integers_as_participant_ids(sample_table, sample_data):
    """Test assignment with large integer participant IDs."""

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


def test_assign_adapter_renders_decimal_and_bool_strata_correctly(sample_table, sample_rows):
    """Test that the adapter correctly renders decimal and bool strata as strings."""
    arms = make_arms(["control", "treatment"])
    result = assign_treatment(
        sa_table=sample_table,
        data=sample_rows,
        stratum_cols=["income_dec", "is_male"],
        id_col="id",
        arms=arms,
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        random_state=42,
    )
    for p in result.assignments:
        json = p.model_dump()
        # we rounded the Decimal to an int, so shouldn't see the decimal point
        assert "." not in json["strata"][0]["strata_value"], json
        assert json["strata"][1]["strata_value"] in {"True", "False"}, json


def test_assign_adapter_with_no_stratification(sample_table, sample_rows):
    """Test assignment with no stratification columns."""
    result = assign_treatment(
        sa_table=sample_table,
        data=sample_rows,
        stratum_cols=[],
        id_col="id",
        arms=make_arms(["control", "treatment"]),
        experiment_id="b767716b-f388-4cd9-a18a-08c4916ce26f",
        random_state=None,  # counts in each arm should always be equal regardless.
    )
    assert result.balance_check is None
    arm_counts: defaultdict[str, int] = defaultdict(int)
    # There should be no strata in the output
    for participant in result.assignments:
        arm_counts[participant.arm_name] += 1
        assert participant.strata is None
    # The number of assignments per arm should be equal
    assert arm_counts["control"] == arm_counts["treatment"]
    assert arm_counts["control"] == len(result.assignments) // 2


def test_assign_adapter_with_no_valid_strata(sample_table, sample_rows):
    """Test assignment when strata columns have no valid stratification values."""
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
    # In this case, we still output the strata column, even though it's all the same value
    arm_counts: defaultdict[str, int] = defaultdict(int)
    for participant in result.assignments:
        arm_counts[participant.arm_name] += 1
        assert participant.strata == [Strata(field_name="single_value", strata_value="1")]
    # And since we used simple random assignment, the arm lengths should be equal
    assert arm_counts["control"] == arm_counts["treatment"]
    assert arm_counts["control"] == len(result.assignments) // 2
