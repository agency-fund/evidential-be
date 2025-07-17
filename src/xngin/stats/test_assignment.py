"""Tests for core assignment statistics functionality."""

import numpy as np
import pandas as pd
import pytest
from numpy.random import MT19937, RandomState
from stochatreat import stochatreat

from xngin.stats.assignment import (
    assign_treatment_and_check_balance,
    simple_random_assignment,
)
from xngin.stats.balance import BalanceResult


def make_sample_data_dict(n=1000):
    """Create sample data dictionary for testing core stats functionality."""
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
    data["is_male"] = [g == "M" for g in data["gender"]]
    return data


@pytest.fixture(name="sample_df")
def fixture_sample_df():
    """Helper that turns a python dict into a pandas DataFrame."""
    return pd.DataFrame(make_sample_data_dict())


def test_assign_treatment_with_stratification(sample_df):
    """Test core assignment logic with stratification."""
    treatment_ids, stratum_ids, balance_result, orig_stratum_cols = assign_treatment_and_check_balance(
        df=sample_df,
        decimal_columns=[],
        stratum_cols=["region", "gender"],
        id_col="id",
        n_arms=2,
        random_state=42,
    )

    # Check lengths and valid assignments
    assert len(treatment_ids) == len(sample_df)
    assert set(treatment_ids) == {0, 1}
    assert len(stratum_ids) == len(sample_df)
    # 2 genders, 4 regions => 8 strata
    assert set(stratum_ids) == set(range(8))
    assert set(orig_stratum_cols) == {"region", "gender"}

    # Check balance result structure
    assert isinstance(balance_result, BalanceResult)
    assert balance_result.f_statistic > 0
    assert balance_result.f_pvalue > 0
    assert (
        len(treatment_ids)
        == balance_result.numerator_df + balance_result.denominator_df + 1
    )


def test_assign_treatment_multiple_arms(sample_df):
    """Test assignment with multiple arms."""
    treatment_ids, stratum_ids, balance_result, orig_stratum_cols = assign_treatment_and_check_balance(
        df=sample_df,
        decimal_columns=[],
        stratum_cols=["gender", "region"],
        id_col="id",
        n_arms=3,
        random_state=42,
    )

    # Check that all three arms are represented
    assert set(treatment_ids) == {0, 1, 2}
    assert len(treatment_ids) == len(sample_df)
    assert len(stratum_ids) == len(sample_df)
    assert set(orig_stratum_cols) == {"gender", "region"}
    assert balance_result is not None
    assert isinstance(balance_result, BalanceResult)


def test_assign_treatment_reproducibility(sample_df):
    """Test that assignment is reproducible with same random state."""
    result1 = assign_treatment_and_check_balance(
        df=sample_df,
        decimal_columns=[],
        stratum_cols=["gender", "region"],
        id_col="id",
        n_arms=2,
        random_state=42,
    )

    result2 = assign_treatment_and_check_balance(
        df=sample_df,
        decimal_columns=[],
        stratum_cols=["gender", "region"],
        id_col="id",
        n_arms=2,
        random_state=42,
    )

    # Check that results are identical
    assert result1[0] == result2[0]  # treatment_ids
    assert result1[1] == result2[1]  # stratum_ids
    assert result1[3] == result2[3]  # orig_stratum_cols
    # Balance results should have same values
    assert result1[2].f_statistic == result2[2].f_statistic
    assert result1[2].f_pvalue == result2[2].f_pvalue


def test_assign_treatment_with_missing_values(sample_df):
    """Test assignment handling of missing values."""
    # Add some missing values
    sample_df.loc[sample_df.index[:100], "income"] = np.nan

    treatment_ids, stratum_ids, balance_result, _ = assign_treatment_and_check_balance(
        df=sample_df,
        decimal_columns=[],
        stratum_cols=["gender", "region"],
        id_col="id",
        n_arms=2,
        random_state=42,
    )

    assert len(treatment_ids) == len(sample_df)
    assert len(stratum_ids) == len(sample_df)
    assert balance_result is not None
    # Check that treatment assignments are not None or NaN
    assert set(treatment_ids) == {0, 1}


def test_assign_treatment_with_no_stratification(sample_df):
    """Test assignment with no stratification columns."""
    treatment_ids, stratum_ids, balance_result, orig_stratum_cols = assign_treatment_and_check_balance(
        df=sample_df,
        decimal_columns=[],
        stratum_cols=[],
        id_col="id",
        n_arms=2,
        random_state=42,
    )

    # Should use simple random assignment
    assert len(treatment_ids) == len(sample_df)
    assert stratum_ids is None
    assert balance_result is None
    assert orig_stratum_cols == []
    # Arm lengths should be equal
    assert set(treatment_ids) == {0, 1}
    assert treatment_ids.count(0) == treatment_ids.count(1)


def test_assign_treatment_with_no_valid_strata(sample_df):
    """Test assignment when strata columns have no valid stratification values."""
    treatment_ids, stratum_ids, balance_result, orig_stratum_cols = assign_treatment_and_check_balance(
        df=sample_df,
        decimal_columns=[],
        stratum_cols=["single_value"],
        id_col="id",
        n_arms=2,
        random_state=42,
    )

    # Should fall back to simple random assignment
    assert len(treatment_ids) == len(sample_df)
    assert stratum_ids is None
    assert balance_result is None
    assert orig_stratum_cols == ["single_value"]
    # Arm lengths should be equal
    assert set(treatment_ids) == {0, 1}
    assert treatment_ids.count(0) == treatment_ids.count(1)


def test_assign_treatment_with_problematic_values():
    """Test assignment with None/NaN that could break stochatreat due to grouping issues."""
    # Entries with None such that the grouping into strata causes stochatreat to raise a
    # ValueError as it internally uses df.groupby(..., dropna=True), causing the count of
    # synthetic rows created to be off.
    df = pd.DataFrame(make_sample_data_dict(20))
    df.loc[0, "gender"] = None
    with pytest.raises(ValueError):
        stochatreat(data=df, idx_col="id", stratum_cols=["gender"], treats=2)
    df.loc[0, "gender"] = np.nan
    with pytest.raises(ValueError):
        stochatreat(data=df, idx_col="id", stratum_cols=["gender"], treats=2)

    treatment_ids, _, balance_result, _ = assign_treatment_and_check_balance(
        df=df,
        decimal_columns=[],
        stratum_cols=["gender"],
        id_col="id",
        n_arms=2,
    )
    # But we still expect success since internally we'll preprocess the data to handle NaNs.
    assert balance_result is not None
    assert balance_result.f_statistic > 0
    assert balance_result.f_pvalue > 0
    assert len(treatment_ids) == len(df)
    assert (
        len(treatment_ids)
        == balance_result.numerator_df + balance_result.denominator_df + 1
    )


def test_simple_random_assignment(sample_df):
    """Test simple random assignment function."""
    assignments = simple_random_assignment(sample_df, n_arms=2, random_state=42)
    assert len(assignments) == len(sample_df)
    assert assignments.count(0) == len(sample_df) // 2
    assert assignments.count(1) == len(sample_df) // 2
    assert set(assignments) == {0, 1}


def test_simple_random_assignment_multiple_arms(sample_df):
    """Test simple random assignment with multiple arms, sample size not divisible by 3."""
    assignments = simple_random_assignment(sample_df, n_arms=3, random_state=42)
    assert len(assignments) == len(sample_df)
    assert assignments.count(0) in {len(sample_df) // 3, len(sample_df) // 3 + 1}
    assert assignments.count(1) in {len(sample_df) // 3, len(sample_df) // 3 + 1}
    assert assignments.count(2) in {len(sample_df) // 3, len(sample_df) // 3 + 1}
    assert set(assignments) == {0, 1, 2}


def test_simple_random_assignment_reproducibility(sample_df):
    """Test that simple random assignment is reproducible."""
    assignments1 = simple_random_assignment(sample_df, n_arms=2, random_state=42)
    assignments2 = simple_random_assignment(sample_df, n_arms=2, random_state=42)
    assert assignments1 == assignments2


def test_simple_random_assignment_different_seeds(sample_df):
    """Test that simple random assignment gives different results with different seeds."""
    assignments1 = simple_random_assignment(sample_df, n_arms=2, random_state=42)
    assignments2 = simple_random_assignment(sample_df, n_arms=2, random_state=123)
    # Should be different with different seeds
    assert assignments1 != assignments2
    # But should have same length and arm counts
    assert len(assignments1) == len(assignments2)
    assert assignments1.count(0) == assignments2.count(0)
    assert assignments1.count(1) == assignments2.count(1)
