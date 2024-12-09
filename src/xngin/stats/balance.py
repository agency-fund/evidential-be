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
    alpha: float = 0.05,
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

    # Handle missing values in numeric columns by converting to quartiles
    cols_with_missing = df_analysis.columns[df_analysis.isnull().any()].tolist()

    for col in cols_with_missing:
        if pd.api.types.is_numeric_dtype(df_analysis[col]):
            df_analysis[f"{col}_quartile"] = pd.qcut(
                df_analysis[col],
                q=4,
                duplicates="drop",
                labels=["q1", "q2", "q3", "q4"],
            )
            df_analysis[f"{col}_quartile"] = df_analysis[
                f"{col}_quartile"
            ].cat.add_categories(["Missing"])
            df_analysis[f"{col}_quartile"] = df_analysis[f"{col}_quartile"].fillna(
                "Missing"
            )
            df_analysis = pd.get_dummies(
                df_analysis, columns=[f"{col}_quartile"], prefix=[col], dummy_na=False
            )
            df_analysis.drop(columns=[col], inplace=True)

    # Create formula excluding specified columns
    covariates = [
        col
        for col in df_analysis.columns
        if col != treatment_col and col not in exclude_cols
    ]

    formula = f"{treatment_col} ~ " + " + ".join(covariates)

    print(formula)
    # Fit regression model
    model = smf.ols(formula=formula, data=df_analysis).fit()
    # model = sm.OLS(balance_data["treat"], balance_data.drop("treat", axis=1)).fit()
    # print(model.summary().as_text())

    return BalanceResult(
        f_statistic=model.fvalue,
        f_pvalue=model.f_pvalue,
        is_balanced=bool(model.f_pvalue > alpha),
        numerator_df=model.df_model,
        denominator_df=model.df_resid,
        model_summary=model.summary().as_text(),
    )
