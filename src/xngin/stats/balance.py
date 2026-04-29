from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api.types import is_any_real_numeric_dtype

from xngin.stats.stats_errors import StatsBalanceError


@dataclass(slots=True)
class BalanceResult:
    """Results from balance check."""

    f_statistic: float
    f_pvalue: float
    numerator_df: float
    denominator_df: float
    # More values for debugging
    model_params: list[float]
    model_param_std_errors: list[float]
    model_summary: str


def preprocess_for_balance_and_stratification(
    data: pd.DataFrame,
    exclude_cols: list[str] | None = None,
    quantiles: int = 4,
    missing_string="__NULL__",
):
    """
    Preprocess data to quantize numerics and replace NaNs with missing_string.

    Args:
        data: DataFrame containing treatment assignments and covariates
        exclude_cols: List of columns caller knows to exclude from balance check
        quantiles: Number of quantiles to bucket numeric columns
        missing_string: value used internally for replacing NAs in non-numeric columns

    Returns tuple:
        df_analysis: processed df
        exclude_set: Column names to exclude from stratification and balance checks.
        numeric_notnull_set: Column names of numerics with no NaNs in the original data, to use with
            restore_original_numeric_columns() if you wish to replace these columns with the
            original values before balance checks.
    """
    # Create copy of data for analysis
    df_analysis = data.copy()

    exclude_set = set() if exclude_cols is None else set(exclude_cols)

    # Exclude columns from the check that contain only the same value or unique non-numeric values.
    # Note: pd.qcut() will return NaN for all objects if the non-None values are identical and we
    # drop duplicate bin edges; pd.get_dummies() will drop a column of all NaN, so drop here.
    single_value_cols = []
    unique_non_numeric_cols = []
    working_list = []
    for col in sorted(set(df_analysis.columns) - exclude_set):
        unique_count = df_analysis[col].nunique(dropna=True)
        if unique_count <= 1:
            single_value_cols.append(col)
        elif not is_any_real_numeric_dtype(df_analysis[col]) and df_analysis[col].dropna().is_unique:
            unique_non_numeric_cols.append(col)
        else:
            working_list.append(col)

    # Also update our exclude_set to return.
    exclude_set = exclude_set.union(single_value_cols, unique_non_numeric_cols)

    # Handle numeric columns (can include NaNs) by converting to quartiles. Excludes booleans.
    numeric_columns = [c for c in working_list if is_any_real_numeric_dtype(df_analysis[c])]
    for col in numeric_columns:
        labels = pd.qcut(
            df_analysis[col],
            q=quantiles,
            duplicates="drop",
            # No labels as dropping edges will misalign labels and trigger a ValueError.
            # Integer indicators starting at 0 will be returned instead.
            labels=False,
        )
        # Since there are NaNs, labels will be dtype=float64. To avoid bugs later due to dummy var
        # naming, we want integer categories, so first cast to nullable ints then category.
        df_analysis[col] = pd.Series(labels).astype("Int8").astype("category")

    # Next backfill NaNs. Since we converted numerics to categoricals, we can treat them the same as
    # the original non-numeric columns.
    column_index = df_analysis[working_list].columns
    isnull_columns = column_index[df_analysis[working_list].isnull().any()]
    for col in isnull_columns:
        if df_analysis[col].dtype == "category":
            df_analysis[col] = df_analysis[col].cat.add_categories(missing_string)
        df_analysis.fillna({col: missing_string}, inplace=True)

    numeric_notnull_set = set(numeric_columns) - set(isnull_columns)
    return df_analysis, exclude_set, numeric_notnull_set


def restore_original_numeric_columns(df_orig: pd.DataFrame, df_cleaned: pd.DataFrame, numeric_notnull_set: set):
    """
    Restore columns named in numeric_notnull_set from df_orig to df_cleaned for better balance test
    results.

    df_orig should typically be the same input and df_cleaned and numeric_notnull_set the outputs of
    preprocess_for_balance_and_stratification().
    """
    if numeric_notnull_set:
        columns = sorted(numeric_notnull_set)
        for col in columns:
            if col not in df_orig.columns or col not in df_cleaned.columns:
                raise ValueError(f"Column {col} is missing from either df_orig or df_cleaned.")

        df_cleaned = df_cleaned.copy()
        df_cleaned[columns] = df_orig[columns]

    return df_cleaned


def check_balance_of_preprocessed_df(
    data: pd.DataFrame,
    treatment_col: str = "treat",
    exclude_col_set: set[str] | None = None,
    cluster_col: str | None = None,
) -> BalanceResult:
    """
    Perform a balance check on treatment assignment.  One should typically first use
    preprocess_for_balance_and_stratification(), then restore_original_numeric_columns(), and
    finally call this.

    Args:
        data: DataFrame containing preprocessed covariates and treatment assignments
        treatment_col: Name of treatment assignment column
        exclude_col_set: Columns to exclude from balance check. Typically should come from
            preprocess_for_balance_and_stratification().
        cluster_col: Name of column containing cluster identifiers. If provided, uses clustered
            standard errors instead of HC1.

    Returns:
        BalanceResult object containing test results
    """
    if data[treatment_col].nunique() <= 1:
        raise ValueError("Treatment column has insufficient arms.")

    exclude_from_covariates_set = {treatment_col}
    if exclude_col_set:
        exclude_from_covariates_set |= exclude_col_set
    if cluster_col is not None:
        exclude_from_covariates_set.add(cluster_col)

    covariates = sorted(set(data.columns) - exclude_from_covariates_set)
    if len(covariates) == 0:
        raise StatsBalanceError(
            "No usable fields for performing a balance check found. Please check your metrics "
            "and fields used for stratification."
        )

    # Fit regression model; for now only check the first two treatment groups.
    df_analysis = data[data[treatment_col].isin([0, 1])]

    # Build the design matrix directly with pd.get_dummies instead of patsy's formula
    # API. patsy iterates every element in Python to detect NAs and convert categoricals,
    # which costs ~5.6s for 1M rows. pd.get_dummies does this vectorized in C.
    categorical_cols = [c for c in covariates if not is_any_real_numeric_dtype(df_analysis[c])]
    numeric_cols = [c for c in covariates if c not in categorical_cols]

    parts: list[pd.DataFrame] = []
    if numeric_cols:
        parts.append(df_analysis[numeric_cols].astype(np.float64))
    if categorical_cols:
        dummies = pd.get_dummies(df_analysis[categorical_cols], drop_first=True, dtype=np.float64)
        parts.append(dummies)

    exog = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df_analysis.index)
    exog.insert(0, "Intercept", 1.0)
    exog = exog.astype(np.float64)
    endog = df_analysis[treatment_col].astype(np.float64)

    if cluster_col is not None:
        model = sm.OLS(endog, exog).fit(
            method="pinv",
            cov_type="cluster",
            cov_kwds={"groups": df_analysis[cluster_col]},
        )
    else:
        # While HC3 may be better at low sample sizes (Long & Ervin 2000), it is sensitive to high
        # leverage points, so use HC1 for now. Future work should consider:
        # https://blog.stata.com/2022/10/06/heteroskedasticity-robust-standard-errors-some-practical-considerations/
        model = sm.OLS(endog, exog).fit(method="pinv", cov_type="HC1")

    return BalanceResult(
        f_statistic=model.fvalue,
        f_pvalue=model.f_pvalue,
        numerator_df=model.df_model,
        denominator_df=model.df_resid,
        model_params=list(model.params),
        model_param_std_errors=list(model.bse),
        model_summary=model.summary().as_text(),
    )
