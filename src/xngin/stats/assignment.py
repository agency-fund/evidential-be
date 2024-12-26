from uuid import UUID

import pandas as pd
from pandas.api.types import is_float_dtype
import numpy as np
from stochatreat import stochatreat
from xngin.apiserver.api_types import (
    AssignResponse,
    Assignment,
    Strata,
)
from xngin.stats.balance import check_balance


MAX_SAFE_INTEGER = (1 << 53) - 1


def assign_treatment(
    data: pd.DataFrame,
    stratum_cols: list[str],
    id_col: str,
    arm_names: list[str],
    experiment_id: str,
    description: str,
    fstat_thresh: float = 0.5,
    random_state: int | None = None,
) -> AssignResponse:
    """
    Perform stratified random assignment and balance checking.

    Args:
        data: DataFrame containing units to be assigned
        stratum_cols: List of column names to stratify on
        id_col: Name of column containing unit identifiers
        arm_names: Names of treatment arms
        experiment_id: Unique identifier for experiment
        description: Description of experiment
        fstat_thresh: Threshold for F-statistic p-value
        random_state: Random seed for reproducibility

    Returns:
        AssignmentResult containing assignments and balance check results
    """
    # Create copy for analysis while attempting to convert any numeric "object" types that pandas didn't originally
    # recognize when creating the dataframe. This does NOT handle Decimal types!
    df = data.infer_objects()
    # Check if the unique identifier was incorrectly inferred to be a float, and try to make it an integer type.
    # WARNING: if the original data actually had ints larger than are representable with full precision as a float64,
    # then the data in the data frame would be problematic to begin with and we should instead try using
    # https://pandas.pydata.org/docs/user_guide/pyarrow.html's pa.decimal128 and not convert to float in settings.py.
    if is_float_dtype(df[id_col].dtype):
        min_, max_ = df[id_col].aggregate(["min", "max"])
        if min_ < -MAX_SAFE_INTEGER or max_ > MAX_SAFE_INTEGER:
            raise ValueError(f"Cannot safely convert '{id_col}' from float to Int64")
        df[id_col] = df[id_col].astype("Int64")

    # Assign treatments
    n_arms = len(arm_names)
    treatment_status = stochatreat(
        data=df,
        stratum_cols=stratum_cols,
        treats=n_arms,
        probs=[1 / n_arms] * n_arms,
        idx_col=id_col,
        random_state=random_state,
    ).drop(columns=["stratum_id"])
    df = df.merge(treatment_status, on=id_col)

    # check balance
    balance_check = check_balance(
        df[[*stratum_cols, "treat"]], "treat", exclude_cols=[id_col], alpha=fstat_thresh
    )

    # Prepare assignments for return
    assignments = df[[id_col, "treat", *stratum_cols]].copy()
    assignments = assignments.melt(
        id_vars=[id_col, "treat"], var_name="strata_name", value_name="strata_value"
    )
    assignments["strata_value"] = assignments["strata_value"].fillna("NA")

    # Convert the assignments DataFrame to a list of ExperimentParticipant objects
    participants_dict = {}
    for row in assignments.itertuples(index=False):
        row = row._asdict()
        id_str = str(row[id_col])
        if id_str not in participants_dict:
            strata = [
                Strata(
                    strata_name=row["strata_name"],
                    strata_value=str(row["strata_value"]),
                )
            ]
            participant = Assignment(
                participant_id=str(row[id_col]),
                treatment_assignment=arm_names[row["treat"]],
                strata=strata,
            )
            participants_dict[id_str] = participant
        else:
            participants_dict[id_str].strata.append(
                Strata(
                    strata_name=row["strata_name"],
                    strata_value=str(row["strata_value"]),
                )
            )

    # Sort participants_list by participant_id
    participants_list = sorted(
        participants_dict.values(), key=lambda p: p.participant_id
    )

    # Return the ExperimentAssignment with the list of participants
    return AssignResponse(
        f_statistic=np.round(balance_check.f_statistic, 9),
        numerator_df=balance_check.numerator_df,
        denominator_df=balance_check.denominator_df,
        p_value=np.round(balance_check.f_pvalue, 9),
        balance_ok=balance_check.f_pvalue > fstat_thresh,
        experiment_id=UUID(experiment_id),
        description=description,
        sample_size=len(df),
        id_col=id_col,
        assignments=participants_list,  # Use the list of participants here
    )
