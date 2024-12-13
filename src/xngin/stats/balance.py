import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from dataclasses import dataclass


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
) -> BalanceResult:
    """
    Perform balance check on treatment assignment.

    Args:
        data: DataFrame containing treatment assignments and covariates
        treatment_col: Name of treatment assignment column
        exclude_cols: List of columns to exclude from balance check
        alpha: Significance level for balance test

    Returns:
        BalanceResult object containing test results
    """
    # Create copy of data for analysis
    df_analysis = data.copy()

    if exclude_cols is None:
        exclude_cols = []

    # Exclude columns from the check that contain only the same value (including None).
    single_value_cols = df_analysis.columns[df_analysis.nunique(dropna=False) <= 1]
    exclude_cols.extend(single_value_cols.to_list())

    # Handle missing values in numeric columns by converting to quartiles
    cols_with_missing = df_analysis.columns[df_analysis.isnull().any()].tolist()

    quantiles = 4
    for col in cols_with_missing:
        if pd.api.types.is_numeric_dtype(df_analysis[col]):
            labels = pd.qcut(
                df_analysis[col],
                q=quantiles,
                duplicates="drop",
                # No labels as dropping edges will misalign labels and trigger a ValueError.
                # Integer indicators starting at 0 will be returned instead.
                labels=False,
            )
            new_col = f"{col}_quartile"
            # Since there are NaNs, labels will be dtype=float64. To avoid bugs later due to dummy var naming, first
            # replace NaNs with an integer beyond the number of buckets, then convert to int, and finally a category.
            df_analysis[new_col] = pd.Categorical(
                np.nan_to_num(labels, nan=quantiles).astype("int8")
            )
            df_analysis = pd.get_dummies(
                df_analysis, columns=[new_col], prefix=[col], dummy_na=False
            )
            df_analysis.drop(columns=[col], inplace=True)

    # Create formula excluding specified columns
    covariates = [
        col
        for col in df_analysis.columns
        if col != treatment_col and col not in exclude_cols
    ]

    # TODO(roboton): Run multi-class regression via MVLogit
    # df_analysis[treatment_col] = pd.Categorical(df_analysis[treatment_col])
    # only check the first two treatment groups
    df_analysis = df_analysis[df_analysis[treatment_col].isin([0, 1])]

    formula = f"{treatment_col} ~ " + " + ".join(covariates)

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
