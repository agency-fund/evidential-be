import decimal
from collections.abc import Sequence
from typing import Any, Protocol
from uuid import UUID

import pandas as pd
from pandas import DataFrame
import numpy as np
from sqlalchemy import Table
from stochatreat import stochatreat
from xngin.apiserver.api_types import (
    AssignResponse,
    Assignment,
    BalanceCheck,
    Strata,
)
from xngin.stats.balance import (
    check_balance_of_preprocessed_df,
    preprocess_for_balance_and_stratification,
    restore_original_numeric_columns,
)


class RowProtocol(Protocol):
    """Minimal methods to approximate a sqlalchemy.engine.row.Row for testing."""

    def _asdict(self) -> dict[str, Any]:
        """https://docs.sqlalchemy.org/en/20/core/connections.html#sqlalchemy.engine.Row._asdict"""


def assign_treatment(
    sa_table: Table,
    data: Sequence[RowProtocol],
    stratum_cols: list[str],
    id_col: str,
    arm_names: list[str],
    experiment_id: str,
    fstat_thresh: float = 0.5,
    random_state: int | None = None,
) -> AssignResponse:
    """
    Perform stratified random assignment and balance checking.

    Args:
        sa_table: sqlalchemy table representation used for type info
        data: sqlalchemy result set of Rows representing units to be assigned
        stratum_cols: List of column names to stratify on
        id_col: Name of column containing unit identifiers
        arm_names: Names of treatment arms
        experiment_id: Unique identifier for experiment
        fstat_thresh: Threshold for F-statistic p-value
        random_state: Random seed for reproducibility

    Returns:
        AssignmentResult containing assignments and balance check results
    """
    # Create copy for analysis while attempting to convert any numeric "object" types that pandas didn't originally
    # recognize when creating the dataframe. This does NOT handle Decimal types!
    df = DataFrame(data).infer_objects()

    # Now convert any Decimal types to float (possible if the Table was created with reflection instead of cursor).
    decimal_columns = [
        c.name for c in sa_table.columns if c.type.python_type is decimal.Decimal
    ]
    df[decimal_columns] = df[decimal_columns].astype(float)

    # Dedupe the strata names and then sort them for a stable output ordering
    orig_stratum_cols = sorted(set(stratum_cols))

    orig_data_to_stratify = df[[id_col, *orig_stratum_cols]]
    df_cleaned, exclude_cols_set, numeric_notnull_set = (
        preprocess_for_balance_and_stratification(
            data=orig_data_to_stratify, exclude_cols=[id_col]
        )
    )
    # Our original target of columns to stratify on may have gotten smaller:
    post_stratum_cols = sorted(set(orig_stratum_cols) - exclude_cols_set)

    # Assign treatments
    n_arms = len(arm_names)
    # TODO: when we support unequal arm assigments, be careful about ensuring the right treatment
    # assignment id is mapped to the right arm_name.
    treatment_status = stochatreat(
        data=df_cleaned,
        idx_col=id_col,
        stratum_cols=post_stratum_cols,
        treats=n_arms,
        probs=[1 / n_arms] * n_arms,
        # internally uses legacy np.random.RandomState which can take None
        random_state=random_state,  # type: ignore[arg-type]
    ).drop(columns=["stratum_id"])
    df_cleaned = df_cleaned.merge(treatment_status, on=id_col)

    # Put back non-null numeric columns for a more robust balance check.
    df_cleaned = restore_original_numeric_columns(
        df_orig=orig_data_to_stratify,
        df_cleaned=df_cleaned,
        numeric_notnull_set=numeric_notnull_set,
    )
    # Do balance check with treatment assignments as the dependent var using preprocessed data.
    balance_check_cols = [*post_stratum_cols, "treat"]
    balance_check = check_balance_of_preprocessed_df(
        df_cleaned[balance_check_cols],
        treatment_col="treat",
        exclude_col_set=exclude_cols_set,
        alpha=fstat_thresh,
    )

    # Prepare assignments for return along with the original data as a list of ExperimentParticipant objects.
    participants_list = []
    for treatment_assignment, row in zip(df_cleaned["treat"], data, strict=False):
        # ExperimentStrata for each column
        row_dict = row._asdict()
        strata = [
            Strata(
                field_name=column,
                strata_value=str(
                    row_dict[column] if pd.notna(row_dict[column]) else "NA"
                ),
            )
            for column in orig_stratum_cols
        ]

        participant = Assignment(
            participant_id=str(row_dict[id_col]),
            treatment_assignment=arm_names[treatment_assignment],
            strata=strata,
        )
        participants_list.append(participant)

    # Sort participants by ID for stable output
    participants_list.sort(key=lambda p: p.participant_id)

    # Return the ExperimentAssignment with the list of participants
    return AssignResponse(
        balance_check=BalanceCheck(
            f_statistic=np.round(balance_check.f_statistic, 9),
            numerator_df=round(balance_check.numerator_df),
            denominator_df=round(balance_check.denominator_df),
            p_value=np.round(balance_check.f_pvalue, 9),
            balance_ok=bool(balance_check.f_pvalue > fstat_thresh),
        ),
        experiment_id=UUID(experiment_id),
        sample_size=len(df_cleaned),
        id_col=id_col,
        assignments=participants_list,
    )
