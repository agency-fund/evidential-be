from uuid import UUID

import pandas as pd
import numpy as np

from stochatreat import stochatreat
from xngin.apiserver.api_types import (
    ExperimentAssignment,
    ExperimentParticipant,
    ExperimentStrata,
)
from xngin.stats.balance import check_balance


def assign_treatment(
    data: pd.DataFrame,
    stratum_cols: list[str],
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
    # recognize when creating the dataframe.
    df = data.infer_objects()

    # Dedupe the strata names and then sort them for a stable output ordering
    stratum_cols = sorted(set(stratum_cols))

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

    balance_check = check_balance(
        df[[*stratum_cols, "treat"]], "treat", exclude_cols=[id_col], alpha=fstat_thresh
    )

    # Prepare assignments for return
    assignments = df[[id_col, "treat", *stratum_cols]].copy()
    # Convert the assignments DataFrame to a list of ExperimentParticipant objects
    participants_list = []
    for row in assignments.itertuples(index=False):
        # ExperimentStrata for each column
        row_dict = row._asdict()
        strata = [
            ExperimentStrata(
                strata_name=column,
                strata_value=str(
                    row_dict[column] if pd.notna(row_dict[column]) else "NA"
                ),
            )
            for column in stratum_cols
        ]

        participant = ExperimentParticipant(
            participant_id=str(row_dict[id_col]),
            treatment_assignment=arm_names[row_dict["treat"]],
            strata=strata,
        )
        participants_list.append(participant)

    # Sort participants by ID for stable output
    participants_list.sort(key=lambda p: p.participant_id)

    # Return the ExperimentAssignment with the list of participants
    return ExperimentAssignment(
        f_statistic=np.round(balance_check.f_statistic, 9),
        numerator_df=balance_check.numerator_df,
        denominator_df=balance_check.denominator_df,
        p_value=np.round(balance_check.f_pvalue, 9),
        balance_ok=balance_check.f_pvalue > fstat_thresh,
        experiment_id=UUID(experiment_id),
        description=description,
        sample_size=len(df),
        id_col=id_col,
        assignments=participants_list,
    )
