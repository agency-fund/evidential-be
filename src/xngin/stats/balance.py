from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import statsmodels.formula.api as smf


@dataclass(slots=True)
class BalanceResult:
    """Results from balance check."""

    f_statistic: float
    f_pvalue: float
    model_summary: str
    is_balanced: bool
    numerator_df: float
    denominator_df: float


def check_balance(
    data: pd.DataFrame,
    treatment_col: str = "treat",
    exclude_cols: list[str] | None = None,
    alpha: float = 0.5,
    quantiles: int = 4,
    missing_string="__NULL__",
) -> BalanceResult:
    """
    Perform balance check on treatment assignment.

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
    df_analysis, exclude_set = preprocess_for_balance_and_stratification(
        data, exclude_cols, quantiles, missing_string
    )

    return check_balance_of_preprocessed_df(
        data=df_analysis,
        treatment_col=treatment_col,
        exclude_col_set=exclude_set,
        alpha=alpha,
    )


def preprocess_for_balance_and_stratification(
    data: pd.DataFrame,
    exclude_cols: list[str] | None = None,
    quantiles: int = 4,
    missing_string=Literal["__NULL__"],
):
    """
    Preprocess data to quantize numerics and replace NaNs with missing_string.

    Args:
        data: DataFrame containing treatment assignments and covariates
        exclude_cols: List of columns to exclude from balance check
        quantiles: Number of quantiles to bucket numeric columns with NAs
        missing_string: value used internally for replacing NAs in non-numeric columns

    Returns tuple:
       df_analysis: processed df
       exclude_set: column names to exclude from stratification and balance checks
    """
    # Create copy of data for analysis
    df_analysis = data.copy()

    exclude_set = set() if exclude_cols is None else set(exclude_cols)

    # Exclude columns from the check that contain only the same value (including None).
    single_value_cols = df_analysis.columns[df_analysis.nunique(dropna=False) <= 1]
    exclude_set.union(single_value_cols)
    # TODO: check for is_unique columns

    # Handle missing values in numeric columns by converting to quartiles
    # TODO: handle is_bool_dtype() separately
    cols_with_missing_values = set(df_analysis.columns[df_analysis.isnull().any()])
    numeric_columns_with_na = {
        c for c in cols_with_missing_values if is_numeric_dtype(df_analysis[c])
    }
    for col in numeric_columns_with_na - exclude_set:
        labels = pd.qcut(
            df_analysis[col],
            q=quantiles,
            duplicates="drop",
            # No labels as dropping edges will misalign labels and trigger a ValueError.
            # Integer indicators starting at 0 will be returned instead.
            labels=False,
        )
        new_col = f"{col}_quantile"
        # Since there are NaNs, labels will be dtype=float64. To avoid bugs later due to dummy var
        # naming, first replace NaNs with an integer beyond the number of buckets, then *convert to
        # int*, and finally a category.
        df_analysis[new_col] = pd.Series(
            np.nan_to_num(labels, nan=quantiles).astype("int8"), dtype="category"
        )
        df_analysis = pd.get_dummies(
            df_analysis,
            columns=[new_col],
            prefix=[col],
            dummy_na=False,
            drop_first=True,
        )
        df_analysis.drop(columns=[col], inplace=True)

    # Handle missing values in non-numeric columns:
    non_numeric_columns_with_na = cols_with_missing_values - numeric_columns_with_na
    for col in non_numeric_columns_with_na - exclude_set:
        df_analysis.fillna({col: missing_string}, inplace=True)

    return df_analysis, exclude_set


def check_balance_of_preprocessed_df(
    data: pd.DataFrame,
    treatment_col: str = "treat",
    exclude_col_set: set[str] | None = None,
    alpha: float = 0.5,
) -> BalanceResult:
    """
    See check_balance(). Assumes the df and exclude_col_set came from
    preprocess_for_balance_and_stratification().
    """
    # Convert all non-numeric columns into dummy vars
    non_numeric_columns = {c for c in data.columns if not is_numeric_dtype(data[c])}
    cols_to_dummies = list(non_numeric_columns - exclude_col_set)
    df_analysis = pd.get_dummies(
        data,
        columns=cols_to_dummies,
        prefix=cols_to_dummies,
        dummy_na=False,
        drop_first=True,
    )

    # Create formula excluding specified columns
    covariates = [
        col
        for col in df_analysis.columns
        if col != treatment_col and col not in exclude_col_set
    ]

    # TODO(roboton): Run multi-class regression via MVLogit
    # df_analysis[treatment_col] = pd.Categorical(df_analysis[treatment_col])
    # only check the first two treatment groups
    df_analysis = df_analysis[df_analysis[treatment_col].isin([0, 1])]

    formula = f"{treatment_col} ~ " + " + ".join(covariates)
    # print(f"------FORMULA:\n\t{formula}")

    # Fit regression model
    model = smf.ols(formula=formula, data=df_analysis).fit()

    return BalanceResult(
        f_statistic=model.fvalue,
        f_pvalue=model.f_pvalue,
        is_balanced=bool(model.f_pvalue > alpha),
        numerator_df=model.df_model,
        denominator_df=model.df_resid,
        model_summary=model.summary().as_text(),
    )
