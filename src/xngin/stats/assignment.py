import pandas as pd
import numpy as np

# from dataclasses import dataclass
from stochatreat import stochatreat
import statsmodels.api as sm
import scipy.stats as stats
from xngin.apiserver.api_types import (
    ExperimentAssignment,
    ExperimentParticipant,
    ExperimentStrata,
)

# @dataclass(slots=True)
# class AssignmentResult:
#     """Results from treatment assignment."""
#     f_statistic: float
#     numerator_df: int
#     denominator_df: int
#     f_pvalue: float
#     balance_ok: bool
#     experiment_id: str
#     description: str
#     sample_size: int
#     assignments: pd.DataFrame


def assign_treatment(
    data: pd.DataFrame,
    stratum_cols: list[str],
    metric_cols: list[str],
    id_col: str,
    arm_names: list[str],
    experiment_id: str,
    description: str,
    fstat_thresh: float = 0.5,
    random_state: int | None = None,
) -> ExperimentAssignment:
    """
    Perform stratified random assignment and balance checking.

    Args:
        data: DataFrame containing units to be assigned
        stratum_cols: List of column names to stratify on
        metric_cols: List of metric column names
        id_col: Name of column containing unit identifiers
        arm_names: Names of treatment arms
        experiment_id: Unique identifier for experiment
        description: Description of experiment
        fstat_thresh: Threshold for F-statistic p-value
        random_state: Random seed for reproducibility

    Returns:
        AssignmentResult containing assignments and balance check results
    """
    # Create copy for analysis
    df = data.copy()

    # Create strata for numeric columns (no missing values)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() == 0:
            df[f"{col}_strata"] = df[col]
        elif df[col].isnull().sum() > 0:
            df[f"{col}_strata"] = pd.qcut(df[col], q=3, labels=False)
            df[f"{col}_strata"] = df[f"{col}_strata"].astype("category")
            df[f"{col}_strata"] = df[f"{col}_strata"].cat.add_categories(["_NA"])
            df[f"{col}_strata"] = df[f"{col}_strata"].fillna("_NA")

    # Create strata for character columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[f"{col}_strata"] = df[col].str.lower().fillna("_NA")

    # Combine strata columns
    strata_cols = [f"{col}_strata" for col in stratum_cols + metric_cols]
    df["selected_strata"] = df[strata_cols].apply(
        lambda x: "_".join(x.astype(str)), axis=1
    )

    # Assign treatments
    n_arms = len(arm_names)
    treatment_status = stochatreat(
        data=df,
        stratum_cols=["selected_strata"],
        treats=n_arms,
        probs=[1 / n_arms] * n_arms,
        idx_col=id_col,
        random_state=random_state,
    )
    df = df.merge(treatment_status, on=id_col)

    # Check balance (for first two arms only)
    balance_data = df[df["treat"].isin([0, 1])].copy()

    # Prepare covariates for balance check
    balance_cols = [
        c
        for c in df.columns
        if c.endswith("_strata")
        and c != "selected_strata"
        and df[c].nunique() > 1
        and not any(x in c for x in ["name", "id"])
    ]

    balance_cols.append("treat")

    balance_data = pd.get_dummies(balance_data[balance_cols])
    # Convert all columns to float
    balance_data = balance_data.astype(float)

    # Fit model for balance check
    model = sm.OLS(balance_data["treat"], balance_data.drop("treat", axis=1)).fit()
    f_stat = model.fvalue
    f_pvalue = 1 - stats.f.cdf(f_stat, model.df_model, model.df_resid)

    # Prepare assignments for return
    # assignments = df[[id_col, 'treat'] + stratum_cols].copy()
    assignments = df[[id_col, "treat", *stratum_cols]].copy()
    assignments = assignments.melt(
        id_vars=[id_col, "treat"], var_name="strata_name", value_name="strata_value"
    )
    assignments["strata_value"] = assignments["strata_value"].fillna("NA")

    # Convert the assignments DataFrame to a list of ExperimentParticipant objects
    participants_list = []
    for row in assignments.itertuples(index=False):
        row = row._asdict()
        strata = [
            ExperimentStrata(
                strata_name=row["strata_name"], strata_value=str(row["strata_value"])
            )
        ]
        # TODO(roboton): the id= field doesn't exist on ExperimentParticipant.
        participant = ExperimentParticipant(
            id=str(row[id_col]), treatment_assignment=str(row["treat"]), strata=strata
        )
        participants_list.append(participant)

    # Return the ExperimentAssignment with the list of participants
    return ExperimentAssignment(
        f_statistic=np.round(f_stat, 9),
        numerator_df=model.df_model,
        denominator_df=model.df_resid,
        p_value=np.round(f_pvalue, 9),
        balance_ok=f_pvalue > fstat_thresh,
        experiment_id=experiment_id,
        description=description,
        sample_size=len(df),
        assignments=participants_list,  # Use the list of participants here
    )
