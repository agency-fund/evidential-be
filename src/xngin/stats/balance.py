from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_any_real_numeric_dtype
import statsmodels.formula.api as smf
from xngin.stats.stats_errors import StatsBalanceError


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
    if exclude_cols is None:
        exclude_cols = [treatment_col]
    else:
        exclude_cols.append(treatment_col)
    df_cleaned, exclude_set = preprocess_for_balance_and_stratification(
        data, exclude_cols, quantiles, missing_string
    )

    return check_balance_of_preprocessed_df(
        data=df_cleaned,
        orig_data=data,
        treatment_col=treatment_col,
        exclude_col_set=exclude_set,
        alpha=alpha,
    )


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
       exclude_set: column names to exclude from stratification and balance checks
    """
    # Create copy of data for analysis
    df_analysis = data.copy()

    exclude_set = set() if exclude_cols is None else set(exclude_cols)

    # Exclude columns from the check that contain only the same value (excluding None).
    # Note: pd.qcut() will return NaN for all objects if the non-None values are identical and we
    # drop duplicate bin edges, and pd.get_dummies() would end up dropping it, so drop here.
    single_value_cols = df_analysis.columns[df_analysis.nunique(dropna=True) <= 1]
    exclude_set = exclude_set.union(single_value_cols)
    # TODO: check for is_unique columns

    # Handle numeric columns (can include NaNs) by converting to quartiles. Excludes booleans.
    numeric_columns = {c for c in data.columns if is_any_real_numeric_dtype(data[c])}
    for col in numeric_columns - exclude_set:
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
    cols_with_missing_values = set(df_analysis.columns[df_analysis.isnull().any()])
    for col in cols_with_missing_values - exclude_set:
        if df_analysis[col].dtype == "category":
            df_analysis[col] = df_analysis[col].cat.add_categories(missing_string)
        df_analysis.fillna({col: missing_string}, inplace=True)

    return df_analysis, exclude_set


def check_balance_of_preprocessed_df(
    data: pd.DataFrame,
    orig_data: pd.DataFrame = None,
    treatment_col: str = "treat",
    exclude_col_set: set[str] | None = None,
    alpha: float = 0.5,
) -> BalanceResult:
    """
    See check_balance(). Assumes the df and exclude_col_set came from
    preprocess_for_balance_and_stratification().

    orig_data: Provide the original dataframe you used with preprocess_for_balance_and_stratification() in order to use the original numeric columns (if they had no missing data) for the balance check.
    """
    if data[treatment_col].nunique() <= 1:
        raise ValueError("Treatment column has insufficient arms.")

    if exclude_col_set is None:
        exclude_col_set = set()

    # Put back all the original numeric columns that had no missing data, since the balance test
    # will work better with the original continuous values.
    if orig_data is not None:
        numeric_columns = list(
            {c for c in orig_data.columns if is_any_real_numeric_dtype(orig_data[c])}
            - set(orig_data.columns[orig_data.isnull().any()])
            - exclude_col_set
        )
        if numeric_columns:
            data = data.drop(columns=numeric_columns).join(orig_data[numeric_columns])

    # Convert all non-numeric columns into dummy vars, including booleans
    non_numeric_columns = {
        c for c in data.columns if not is_any_real_numeric_dtype(data[c])
    }
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
    if len(covariates) == 0:
        raise StatsBalanceError(
            "No usable fields for performing a balance check found. Please check your metrics and fields used for stratification."
        )

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
