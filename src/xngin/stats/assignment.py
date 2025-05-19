from datetime import UTC, datetime
import decimal
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd
from pandas import DataFrame
from sqlalchemy import Table
from stochatreat import stochatreat
from xngin.apiserver.routers.stateless_api_types import (
    Arm,
    Assignment,
    AssignResponse,
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


STOCHATREAT_STRATUM_ID_NAME = "stratum_id"
STOCHATREAT_TREAT_NAME = "treat"


def assign_treatment(
    sa_table: Table,
    data: Sequence[RowProtocol],
    stratum_cols: list[str],
    id_col: str,
    arms: list[Arm],
    experiment_id: str,
    fstat_thresh: float = 0.5,
    quantiles: int = 4,
    stratum_id_name: str | None = None,
    random_state: int | None = None,
) -> AssignResponse:
    """
    Perform stratified random assignment and balance checking.

    Args:
        sa_table: sqlalchemy table representation used for type info
        data: sqlalchemy result set of Rows representing units to be assigned
        stratum_cols: List of column names to stratify on
        id_col: Name of column containing unit identifiers
        arms: Name & id of each treatment arm
        experiment_id: Unique identifier for experiment
        fstat_thresh: Threshold for F-statistic p-value
        quantiles: number of buckets to use for stratification of numerics
        stratum_id_name: If you want to output the strata group ids, provide a non-null name for
                         the column to add to the assignment output as a Strata field.
        random_state: Random seed for reproducibility

    Returns:
        AssignmentResult containing assignments and balance check results
    """
    if len(stratum_cols) == 0:
        # No stratification, so use simple random assignment
        treatment_ids = simple_random_assignment(data, arms, random_state)
        return _make_assign_response(
            data=data,
            orig_stratum_cols=[],
            id_col=id_col,
            arms=arms,
            experiment_id=experiment_id,
            balance_check=None,
            treatment_ids=treatment_ids,
            stratum_ids=None,
            stratum_id_name=None,
        )

    # Create copy for analysis while attempting to convert any numeric "object" types that pandas
    # didn't originally recognize when creating the dataframe. This does NOT handle Decimal types!
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
            data=orig_data_to_stratify,
            exclude_cols=[id_col],
            quantiles=quantiles,
        )
    )
    # Our original target of columns to stratify on may have gotten smaller:
    post_stratum_cols = sorted(set(orig_stratum_cols) - exclude_cols_set)

    if len(post_stratum_cols) == 0:
        # No stratification, so use simple random assignment while still outputting strata, even
        # though they're either all the same value or all unique values.
        treatment_ids = simple_random_assignment(data, arms, random_state)
        return _make_assign_response(
            data=data,
            orig_stratum_cols=orig_stratum_cols,
            id_col=id_col,
            arms=arms,
            experiment_id=experiment_id,
            balance_check=None,
            treatment_ids=treatment_ids,
            stratum_ids=None,
            stratum_id_name=None,
        )

    # Do stratified random assignment
    n_arms = len(arms)
    # TODO: when we support unequal arm assignments, be careful about ensuring the right treatment
    # assignment id is mapped to the right arm_name.
    treatment_status = stochatreat(
        data=df_cleaned,
        idx_col=id_col,
        stratum_cols=post_stratum_cols,
        treats=n_arms,
        probs=[1 / n_arms] * n_arms,
        # internally uses legacy np.random.RandomState which can take None
        random_state=random_state,  # type: ignore[arg-type]
    )
    df_cleaned = df_cleaned.merge(treatment_status, on=id_col)
    stratum_ids = df_cleaned[STOCHATREAT_STRATUM_ID_NAME]
    treatment_ids = df_cleaned[STOCHATREAT_TREAT_NAME]

    # Put back non-null numeric columns for a more robust balance check.
    df_cleaned.drop(columns=[STOCHATREAT_STRATUM_ID_NAME], inplace=True)
    df_cleaned_for_balance_check = restore_original_numeric_columns(
        df_orig=orig_data_to_stratify,
        df_cleaned=df_cleaned,
        numeric_notnull_set=numeric_notnull_set,
    )
    # Explicitly delete to avoid accidental reuse and free memory. Could gc.collect() if needed.
    del orig_data_to_stratify
    del df_cleaned
    # Do balance check with treatment assignments as the dependent var using preprocessed data.
    balance_check_cols = [*post_stratum_cols, STOCHATREAT_TREAT_NAME]
    balance_result = check_balance_of_preprocessed_df(
        df_cleaned_for_balance_check[balance_check_cols],
        treatment_col=STOCHATREAT_TREAT_NAME,
        exclude_col_set=exclude_cols_set,
        alpha=fstat_thresh,
    )
    del df_cleaned_for_balance_check
    balance_check = BalanceCheck(
        f_statistic=np.round(balance_result.f_statistic, 9),
        numerator_df=round(balance_result.numerator_df),
        denominator_df=round(balance_result.denominator_df),
        p_value=np.round(balance_result.f_pvalue, 9),
        balance_ok=bool(balance_result.f_pvalue > fstat_thresh),
    )

    return _make_assign_response(
        data=data,
        orig_stratum_cols=orig_stratum_cols,
        id_col=id_col,
        arms=arms,
        experiment_id=experiment_id,
        balance_check=balance_check,
        treatment_ids=treatment_ids,
        stratum_ids=stratum_ids,
        stratum_id_name=stratum_id_name,
    )


def simple_random_assignment(
    data: Sequence[RowProtocol],
    arms: list[Arm],
    random_state: int | None = None,
) -> list[int]:
    """
    Perform simple random assignment of data into the given arms.

    Args:
        data: sqlalchemy result set of Rows representing units to be assigned
        arms: Name & uuid of each treatment arm
        random_state: Random seed for reproducibility

    Returns:
        List of treatment ids
    """
    rng = np.random.default_rng(random_state)
    n_arms = len(arms)
    # Create an equal number of treatment ids for each arm and shuffle to ensure arms are as balanced as possible.
    treatment_ids = list(range(n_arms))
    treatment_mask = np.repeat(treatment_ids, np.ceil(len(data) / n_arms))
    rng.shuffle(treatment_mask)
    return treatment_mask[: len(data)].tolist()


def _make_assign_response(
    data: Sequence[RowProtocol],
    orig_stratum_cols: list[str],
    id_col: str,
    arms: list[Arm],
    experiment_id: str,
    balance_check: BalanceCheck | None,
    treatment_ids: list[int],
    stratum_ids: list[int] | None,
    stratum_id_name: str | None = None,
) -> AssignResponse:
    """Prepare assignments for return along with the original data as a list of ExperimentParticipant objects."""
    participants_list = []
    arm_sizes_by_treatment_id: dict[int, int] = defaultdict(int)

    stratum_ids = [0] * len(treatment_ids) if stratum_ids is None else stratum_ids
    for stratum_id, treatment_assignment, row in zip(
        stratum_ids, treatment_ids, data, strict=False
    ):
        strata = None
        row_dict = row._asdict()

        if orig_stratum_cols:
            # Output the participant's strata values as seen at this time of assignment.
            strata = [
                Strata(
                    field_name=column,
                    strata_value=str(
                        row_dict[column] if pd.notna(row_dict[column]) else "NA"
                    ),
                )
                for column in orig_stratum_cols
            ]
            if stratum_id_name is not None:
                strata.append(
                    Strata(field_name=stratum_id_name, strata_value=str(stratum_id))
                )

        arm_sizes_by_treatment_id[treatment_assignment] += 1
        participant = Assignment(
            participant_id=str(row_dict[id_col]),
            arm_id=arms[treatment_assignment].arm_id,
            arm_name=arms[treatment_assignment].arm_name,
            created_at=datetime.now(UTC),
            strata=strata,
        )
        participants_list.append(participant)

    # Sort participants by ID for stable output
    participants_list.sort(key=lambda p: p.participant_id)

    # Return the ExperimentAssignment with the list of participants
    return AssignResponse(
        balance_check=balance_check,
        experiment_id=experiment_id,
        sample_size=len(treatment_ids),
        unique_id_field=id_col,
        assignments=participants_list,
    )
