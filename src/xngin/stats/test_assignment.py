"""Tests for core assignment statistics functionality."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from stochatreat import stochatreat

from xngin.stats.assignment import (
    assign_treatment_and_check_balance,
    simple_random_assignment,
)
from xngin.stats.balance import BalanceResult


def make_sample_data_dict(n=1000):
    """Create sample data dictionary for testing core stats functionality."""
    rs = np.random.default_rng(42)
    data = {
        "id": range(n),
        "age": np.round(rs.normal(30, 5, n), 0),
        "income": np.round(np.float64(rs.lognormal(10, 1, n)), 0),
        "gender": rs.choice(["M", "F"], n),
        "region": rs.choice(["North", "South", "East", "West"], n),
        "skewed": rs.permutation(
            np.concatenate((
                np.repeat([1], int(n * 0.9)),
                np.repeat([0], n - int(n * 0.9)),
            ))
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
    result = assign_treatment_and_check_balance(
        df=sample_df,
        stratum_cols=["region", "gender"],
        id_col="id",
        n_arms=2,
        random_state=42,
    )

    # Check lengths and valid assignments
    assert len(result.treatment_ids) == len(sample_df)
    assert set(result.treatment_ids) == {0, 1}
    assert result.stratum_ids
    assert len(result.stratum_ids) == len(sample_df)
    # 2 genders, 4 regions => 8 strata
    assert set(result.stratum_ids) == set(range(8))
    assert set(result.orig_stratum_cols) == {"region", "gender"}

    # Check balance result structure
    assert isinstance(result.balance_result, BalanceResult)
    assert result.balance_result.f_statistic > 0
    assert result.balance_result.f_pvalue > 0
    assert len(result.treatment_ids) == result.balance_result.numerator_df + result.balance_result.denominator_df + 1


def test_assign_treatment_multiple_arms(sample_df):
    """Test assignment with multiple arms."""
    result = assign_treatment_and_check_balance(
        df=sample_df,
        stratum_cols=["gender", "region"],
        id_col="id",
        n_arms=3,
        random_state=42,
    )

    # Check that all three arms are represented
    assert set(result.treatment_ids) == {0, 1, 2}
    assert len(result.treatment_ids) == len(sample_df)
    assert result.stratum_ids
    assert len(result.stratum_ids) == len(sample_df)
    assert set(result.orig_stratum_cols) == {"gender", "region"}
    assert result.balance_result is not None
    assert isinstance(result.balance_result, BalanceResult)


def test_assign_treatment_reproducibility(sample_df):
    """Test that assignment is reproducible with same random state."""
    result1 = assign_treatment_and_check_balance(
        df=sample_df,
        stratum_cols=["gender", "region"],
        id_col="id",
        n_arms=2,
        random_state=42,
    )

    result2 = assign_treatment_and_check_balance(
        df=sample_df,
        stratum_cols=["gender", "region"],
        id_col="id",
        n_arms=2,
        random_state=42,
    )

    # Check that results are identical
    assert result1.treatment_ids == result2.treatment_ids
    assert result1.stratum_ids == result2.stratum_ids
    assert result1.orig_stratum_cols == result2.orig_stratum_cols
    # Balance results should have same values
    assert result1.balance_result
    assert result2.balance_result
    assert result1.balance_result.f_statistic == result2.balance_result.f_statistic
    assert result1.balance_result.f_pvalue == result2.balance_result.f_pvalue


def test_assign_treatment_with_missing_values(sample_df):
    """Test assignment handling of missing values."""
    # Add some missing values
    sample_df.loc[sample_df.index[:100], "income"] = np.nan

    result = assign_treatment_and_check_balance(
        df=sample_df,
        stratum_cols=["gender", "region"],
        id_col="id",
        n_arms=2,
        random_state=42,
    )

    assert len(result.treatment_ids) == len(sample_df)
    assert result.stratum_ids
    assert len(result.stratum_ids) == len(sample_df)
    assert result.balance_result is not None
    # Check that treatment assignments are not None or NaN
    assert set(result.treatment_ids) == {0, 1}


def test_assign_treatment_with_infer_objects():
    """Test that we infer objects correctly, resulting in a proper number of strata."""
    n = 300
    df = (
        pd.DataFrame()
        .assign(
            id=np.arange(n),
            # nullable, non-unique numeric => 3 levels (NaN, 0, 1)
            col1=pd.Series([None, 1, 2] * (n // 3), dtype="object"),
            # nullable, unique numeric => 5 levels since default quantiles=4, but only 1 NaN to stratify on
            # The NaN will be treated as a separate category with one member, becoming a "high leverage" point.
            col2=[np.nan, *list(np.arange(n - 1))],
            # non-unique numeric => 2 levels
            col3=[1.0, 2.0] * (n // 2),
        )
        .astype("O")
    )  # turn all columns into objects to test inference
    # Improper inference of objects would raise a SyntaxError in the regression of
    # balance.py::check_balance_of_preprocessed_df due to creating dummies out of float64s.
    result = assign_treatment_and_check_balance(
        df=df,
        stratum_cols=["col1", "col2", "col3"],
        id_col="id",
        n_arms=2,
        random_state=42,
    )
    assert set(result.treatment_ids) == {0, 1}
    # Improper inference of objects would result in a different number of strata.
    assert result.stratum_ids
    assert set(result.stratum_ids) == set(range(25))  # = 2*4*3 + 1 for the None in col1
    assert result.balance_result
    # Note: HC1 robust standard errors doesn't downweight by leverage, so p-value is quite low, i.e.
    # it's not a great estimator either in this scenario of singleton categories.
    assert result.balance_result.f_pvalue < 0.2
    assert result.orig_stratum_cols == ["col1", "col2", "col3"]


def test_assign_treatment_decimal_strata_columns_may_cause_problems(sample_df):
    """Test that unconverted Decimal strata columns does NOT raise an error due to implicit categorical treatment.

    Note: when cardinality is high generating many dummy variables, it can error if the dummies
    cause the design matrix to be rank deficient. (Also possible for any sparse categorical column.)
    """
    # Ok since we treat it as a categorical (for better or worse):
    sample_df["decimal"] = sample_df["income"].apply(lambda x: Decimal(round(x, -5)))
    # Number of distinct decimals due to rounding: 5
    # Would cause rank deficiency error without rounding if we used HC3 robust standard errors.
    assign_treatment_and_check_balance(df=sample_df, stratum_cols=["decimal"], id_col="id", n_arms=2)


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

    result = assign_treatment_and_check_balance(
        df=df,
        stratum_cols=["gender"],
        id_col="id",
        n_arms=2,
    )
    # But we still expect success since internally we'll preprocess the data to handle NaNs.
    assert result.balance_result is not None
    assert result.balance_result.f_statistic > 0
    assert result.balance_result.f_pvalue > 0
    assert len(result.treatment_ids) == len(df)
    assert len(result.treatment_ids) == result.balance_result.numerator_df + result.balance_result.denominator_df + 1


def test_assign_treatment_with_no_stratification(sample_df):
    """Test assignment with no stratification columns."""
    result = assign_treatment_and_check_balance(
        df=sample_df,
        stratum_cols=[],
        id_col="id",
        n_arms=2,
        random_state=42,
    )

    # Should use simple random assignment
    assert len(result.treatment_ids) == len(sample_df)
    assert result.stratum_ids is None
    assert result.balance_result is None
    assert result.orig_stratum_cols == []
    # Arm lengths should be equal
    assert set(result.treatment_ids) == {0, 1}
    assert result.treatment_ids.count(0) == result.treatment_ids.count(1)


def test_assign_treatment_with_no_valid_strata(sample_df):
    """Test assignment when strata columns have no valid stratification values."""
    result = assign_treatment_and_check_balance(
        df=sample_df,
        stratum_cols=["single_value"],
        id_col="id",
        n_arms=2,
        random_state=42,
    )

    # Should fall back to simple random assignment
    assert len(result.treatment_ids) == len(sample_df)
    assert result.stratum_ids is None
    assert result.balance_result is None
    assert result.orig_stratum_cols == ["single_value"]
    # Arm lengths should be equal
    assert set(result.treatment_ids) == {0, 1}
    assert result.treatment_ids.count(0) == result.treatment_ids.count(1)


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


def test_simple_random_assignment_with_different_seeds(sample_df):
    """Test that simple random assignment gives different results with different seeds."""
    assignments1 = simple_random_assignment(sample_df, n_arms=2, random_state=42)
    # Test random_state reproducibility
    assignments2 = simple_random_assignment(sample_df, n_arms=2, random_state=42)
    assert assignments1 == assignments2
    # Should be different with different seeds
    assignments3 = simple_random_assignment(sample_df, n_arms=2, random_state=123)
    assert assignments1 != assignments3
    # But should have same length and arm counts
    assert len(assignments1) == len(assignments3)
    assert assignments1.count(0) == assignments3.count(0)
    assert assignments1.count(1) == assignments3.count(1)


def test_assign_treatment_with_bigints_as_participant_ids(sample_df):
    """Test assignment with large integer participant IDs (int64) are handled correctly.

    stochatreat's logic may silently upcast certain dtypes to float64s, which can cause precision
    loss, resulting in incorrect/duplicate ids getting treatment assignments.
    """
    sample_df["id"] = sample_df["id"].astype("int64")
    max_safe_int = (1 << 53) - 1
    # This value is 9007199254740993 but if cast to float64 would round to 9007199254740992
    sample_df.loc[1, "id"] = max_safe_int + 2
    # Next value would get rounded to 103241243500726320 if a float64
    sample_df.loc[2, "id"] = 103241243500726324
    result = assign_treatment_and_check_balance(
        df=sample_df,
        stratum_cols=["gender", "region"],
        id_col="id",
        n_arms=2,
        random_state=42,
    )

    # If stochatreat silently upcasted int64 to float64, we'd lose assignments for both the bigints
    # above, as they would not join back with any of the original ids in the df.
    assert len(result.treatment_ids) == len(sample_df)
