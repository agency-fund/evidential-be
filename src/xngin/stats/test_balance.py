import pytest
import pandas as pd
import numpy as np
from stochatreat import stochatreat
from xngin.stats.balance import (
    BalanceResult,
    check_balance_of_preprocessed_df,
    preprocess_for_balance_and_stratification,
    restore_original_numeric_columns,
)
from xngin.stats.stats_errors import StatsBalanceError


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 1000
    data = {
        "treat": np.random.binomial(1, 0.5, n),
        "age": np.random.normal(30, 5, n),
        "income": np.random.lognormal(10, 1, n),
        "gender": np.random.binomial(1, 0.5, n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
    }
    return pd.DataFrame(data)


def check_balance(
    data: pd.DataFrame,
    treatment_col: str = "treat",
    exclude_cols: list[str] | None = None,
    alpha: float = 0.5,
    quantiles: int = 4,
    missing_string="__NULL__",
) -> BalanceResult:
    """
    Helper wrapper around our components: preprocess_for_balance_and_stratification(),
    restore_original_numeric_columns(), and check_balance_of_preprocessed_df().

    Originally called after stratified random assignment, we've split this into a preprocessing step
    shared by the random assignment, then steps for the actual balance check. Retained for
    convenience doing unittests of the above.

    Args:
        data: DataFrame containing treatment assignments and covariates
        treatment_col: Name of treatment assignment column
        exclude_cols: List of columns to exclude from balance check
        alpha: Significance level for balance test
        quantiles: Number of quantiles to bucket numeric columns with NAs
        missing_string: value used internally for replacing NAs in non-numeric columns

    Returns:
        BalanceResult object containing test results
    """
    if exclude_cols is None:
        exclude_cols = [treatment_col]
    else:
        exclude_cols.append(treatment_col)

    df_cleaned, exclude_cols_set, numeric_notnull_set = (
        preprocess_for_balance_and_stratification(
            data=data,
            exclude_cols=exclude_cols,
            quantiles=quantiles,
            missing_string=missing_string,
        )
    )
    df_cleaned = restore_original_numeric_columns(
        df_orig=data,
        df_cleaned=df_cleaned,
        numeric_notnull_set=numeric_notnull_set,
    )
    return check_balance_of_preprocessed_df(
        data=df_cleaned,
        treatment_col=treatment_col,
        exclude_col_set=exclude_cols_set,
        alpha=alpha,
    )


def test_check_balance(sample_data):
    result = check_balance(sample_data)

    assert result.f_statistic is not None
    assert result.f_pvalue is not None
    assert result.is_balanced
    assert result.model_summary is not None


def test_check_balance_with_missing_values(sample_data):
    # Add some missing values
    sample_data.loc[sample_data.index[:100], "income"] = np.nan

    result = check_balance(sample_data)

    assert result.f_statistic is not None
    assert result.f_pvalue is not None
    assert result.is_balanced


def test_check_balance_with_excluded_cols(sample_data):
    result = check_balance(data=sample_data, exclude_cols=["income", "region"])

    assert result.f_statistic is not None
    assert result.f_pvalue is not None
    assert isinstance(result.is_balanced, bool)


def test_check_balance_invalid_treatment(sample_data):
    invalid_data = sample_data.drop("treat", axis=1)

    with pytest.raises(KeyError):
        check_balance(invalid_data)


def test_check_balance_with_single_value_columns(sample_data):
    sample_data["constant_one"] = [1] * len(sample_data)
    sample_data["constant_none"] = [None] * len(sample_data)

    result = check_balance(sample_data)

    assert result.f_statistic is not None
    assert result.f_pvalue is not None
    assert result.is_balanced
    assert result.model_summary is not None


def test_check_balance_with_column_exclusion_from_dummy_var_generation():
    """
    If pd.qcut() used labels, this triggers the ValueError:
      Bin labels must be one fewer than the number of bin edges
    """
    data = {
        "treat": [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        "int64": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
        "index": [None, *range(0, 19)],
        "sindex": [str(i) for i in range(0, 20)],
    }
    df = pd.DataFrame(data)
    result = check_balance(df, exclude_cols=["index", "sindex"])

    assert result.numerator_df == 1
    assert result.denominator_df == 18
    assert result.is_balanced is True
    assert result.model_summary is not None


def test_check_balance_with_skewed_column_doesnt_raise_valueerror():
    """
    If pd.qcut() used labels, this triggers the ValueError:
      Bin labels must be one fewer than the number of bin edges
    """
    data = {
        "treat": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1] * 2,
        "skews": [0, 0, 0, 0, 0, 0, 4, 4, np.nan, np.nan] * 2,
    }
    df = pd.DataFrame(data)
    result = check_balance(df)

    assert result.f_statistic is not None
    assert result.f_pvalue is not None
    assert result.is_balanced is False
    assert result.model_summary is not None


def test_check_balance_with_mostly_nulls_categorical():
    """
    Dataset purposely design to result in predictors > # observations *if we
    failed* to handle NAs in non-numeric columns (which otherwise would
    induce dropped rows in the ols).
    """
    data = {
        "treat": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 2,
        "int64": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2,
        "float": np.random.uniform(size=20),
        "nulls": [None] * 16 + ["a", "b"] * 2,
    }
    df = pd.DataFrame(data)
    result = check_balance(df)

    assert pd.isna(result.f_pvalue) is False, result.model_summary
    assert pd.isna(result.f_statistic) is False, result.model_summary
    assert result.denominator_df > 0
    assert isinstance(result.is_balanced, bool)
    assert result.model_summary is not None


def test_preprocessing():
    """
    Capture basic behavior of different data types. (Booleans in another test.)
    """
    data = {
        # Numeric
        "ints": range(0, 10),
        # Numeric w/ missing values
        "ints_na": [*range(0, 9), None],
        # Non-numeric
        "strs": ["a", "b"] * 5,
        # Non-numeric with missing values
        "strs_na": ["a", "b", "c", "d", "e", "f", "g", "h", "h", None],
    }
    missing_string = "_TEST_"
    df, exclude, numeric_notnull_set = preprocess_for_balance_and_stratification(
        pd.DataFrame(data), quantiles=4, missing_string=missing_string
    )

    assert exclude == set()
    assert numeric_notnull_set == {"ints"}
    # Our ints were converted into quantiles, and our missing string is a unique value:
    assert df["ints"].nunique() == 4
    assert df["ints_na"].nunique() == 5
    assert df["ints_na"][9] == missing_string
    # non-numerics also have missing values converted:
    assert df["strs"].nunique() == 2
    assert df["strs_na"].nunique() == 9
    assert df["strs_na"][9] == missing_string
    # Same number of columns since we no longer create dummy vars here
    assert len(df.columns) == 4


def test_preprocessing_booleans():
    """
    Verify handling booleans avoids conversion to quantiles, as otherwise a skewed distribution
    could result in every value collapsing into one quantile.
    """
    data = {
        # If treated as a numeric and quantiled, this list would get mapped to only one interval:
        # (-0.001, 1.0], but we skip quantiling for booleans.
        "bools": [True] * 8 + [False] * 2,
        # However, a raw list of booleans + None is stored as dtype='object', so we handle this as a
        # categorical.
        "bools_na": [True] * 8 + [False] * 1 + [None],
    }
    df, exclude, numeric_notnull_set = preprocess_for_balance_and_stratification(
        pd.DataFrame(data)
    )
    assert exclude == set()
    assert numeric_notnull_set == set()
    assert df["bools"].nunique() == 2
    assert df["bools_na"].nunique() == 3
    assert df["bools"].dtype == "bool"
    assert df["bools_na"].dtype == "object"


def test_preprocessing_with_exclusions():
    """
    Verify we exclude certain columns from preprocessing for various reasons.
    """
    data = pd.DataFrame({
        # This is explicitly excluded by caller
        "skip": [2, 2, 3, 3],
        # These are excluded to to all being the same value.
        "same_int": [1.0] * 4,
        "same_int_na": [1, 1, None, None],
        "same_str": ["a"] * 4,
        # Only uniq_obj is excluded since our check for all uniques is for non-numerics
        "uniq_int": range(0, 4),
        "uniq_obj": pd.Series(range(0, 4), dtype="object"),
        # uniq_obj_na is excluded when nones are ignored
        "uniq_obj_na": ["a", "b", None, None],
        # Excluded since NA is dropped when testing for all identical values.
        "same_value_na": [1, 1, None, None],
    })
    df, exclude, numeric_notnull_set = preprocess_for_balance_and_stratification(
        data, exclude_cols=["skip"]
    )

    assert exclude == {
        "skip",
        "same_int",
        "same_int_na",
        "same_str",
        "uniq_obj",
        "uniq_obj_na",
        "same_value_na",
    }
    assert numeric_notnull_set == {"uniq_int"}
    assert set(df.columns) == set(data.columns)
    # test that we didn't preprocess any columns since either they were skipped or didn't need it,
    # i.e. uniq_int
    assert df.compare(data).empty

    # Lastly check that if we did try to assign but have no variability, we raise an error.
    with pytest.raises(StatsBalanceError) as excinfo:
        df["treat"] = [0, 1, 0, 1]  # assignments needed for balance check
        exclude_all = exclude | {"uniq_int"}
        check_balance_of_preprocessed_df(df, exclude_col_set=exclude_all)
    assert "No usable fields for performing a balance check found." in str(
        excinfo.value
    )


def test_preprocessing_numerics_as_categories():
    # Motivation: stochatreat doesn't preprocess numerics, so would end up creating 1 strata per
    # unique value, effectively devolving into simple random sampling, which we don't want:
    data = pd.DataFrame({"id": range(0, 100), "a": range(0, 100)})
    status = stochatreat(data=data, idx_col="id", stratum_cols=["a"], treats=2)
    assert status["stratum_id"].max() == 99

    # So ensure that we construct quantiles for all numerics:
    data = pd.DataFrame({
        "ints": range(0, 100),
        "ints_with_na": [*range(0, 99), None],
        "floats": np.random.normal(30, 5, 100),
    })
    df, exclude, numeric_notnull_set = preprocess_for_balance_and_stratification(data)

    assert exclude == set()
    assert numeric_notnull_set == {"ints", "floats"}
    assert set(df["ints"]) == {0, 1, 2, 3}
    assert set(df["ints_with_na"]) == {0, 1, 2, 3, "__NULL__"}
    assert set(df["floats"]) == {0, 1, 2, 3}
