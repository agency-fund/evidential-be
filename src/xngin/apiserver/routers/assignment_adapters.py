"""Bridges our API types with assignment logic that operates on DataFrames."""

import decimal
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd
from pandas import DataFrame
from sqlalchemy import Insert, Table
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.routers.common_api_types import (
    Arm,
    Assignment,
    AssignResponse,
    BalanceCheck,
    Strata,
)
from xngin.apiserver.sqla import tables
from xngin.stats.assignment import AssignmentResult, assign_treatment_and_check_balance
from xngin.stats.balance import BalanceResult


class RowProtocol(Protocol):
    """Minimal methods to approximate a sqlalchemy.engine.row.Row for testing."""

    def _asdict(self) -> dict[str, Any]:
        """https://docs.sqlalchemy.org/en/20/core/connections.html#sqlalchemy.engine.Row._asdict"""
        ...


def make_balance_check(balance_result: BalanceResult | None, fstat_thresh: float) -> BalanceCheck | None:
    """Convert stats lib's BalanceResult to our API's BalanceCheck."""
    if balance_result is None:
        return None

    return BalanceCheck(
        f_statistic=np.round(balance_result.f_statistic, 9),
        numerator_df=round(balance_result.numerator_df),
        denominator_df=round(balance_result.denominator_df),
        p_value=np.round(balance_result.f_pvalue, 9),
        balance_ok=bool(balance_result.f_pvalue > fstat_thresh),
    )


def assign_treatment(
    sa_table: Table,
    data: Sequence[RowProtocol],
    stratum_cols: list[str],
    id_col: str,
    arms: Sequence[Arm],
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
        AssignResponse containing assignments and balance check results
    """
    # Convert SQLAlchemy data to DataFrame
    df = DataFrame(data)

    # Extract Decimal column names from SQLAlchemy table and convert to float.
    # (Decimals are possible if the Table was created with SA's autoload instead of cursor).
    if decimals := [c.name for c in sa_table.columns if c.type.python_type is decimal.Decimal]:
        df[decimals] = df[decimals].astype(float)

    # Call the core assignment function
    result = assign_treatment_and_check_balance(
        df=df,
        stratum_cols=stratum_cols,
        id_col=id_col,
        n_arms=len(arms),
        quantiles=quantiles,
        random_state=random_state,
    )

    return _make_assign_response(
        data=data,
        orig_stratum_cols=result.orig_stratum_cols,
        id_col=id_col,
        arms=arms,
        experiment_id=experiment_id,
        balance_check=make_balance_check(result.balance_result, fstat_thresh),
        treatment_ids=result.treatment_ids,
        stratum_ids=result.stratum_ids,
        stratum_id_name=stratum_id_name,
    )


def _make_assign_response(
    data: Sequence[RowProtocol],
    orig_stratum_cols: list[str],
    id_col: str,
    arms: Sequence[Arm],
    experiment_id: str,
    balance_check: BalanceCheck | None,
    treatment_ids: list[int],
    stratum_ids: list[int] | None,
    stratum_id_name: str | None = None,
) -> AssignResponse:
    """Prepare assignments and metadata for return."""
    participants_list = []
    arm_sizes_by_treatment_id: dict[int, int] = defaultdict(int)

    # Track if we originally had valid strata
    had_valid_strata = stratum_ids is not None
    stratum_ids = [0] * len(treatment_ids) if stratum_ids is None else stratum_ids

    for stratum_id, treatment_assignment, row in zip(stratum_ids, treatment_ids, data, strict=True):
        strata = None
        row_dict = row._asdict()

        if orig_stratum_cols:
            # Output the participant's strata values as seen at this time of assignment.
            strata = [
                Strata(
                    field_name=column,
                    strata_value=str(row_dict[column] if pd.notna(row_dict[column]) else "NA"),
                )
                for column in orig_stratum_cols
            ]
            # Only add stratum_id if we had valid strata and stratum_id_name is provided
            if stratum_id_name is not None and had_valid_strata:
                strata.append(Strata(field_name=stratum_id_name, strata_value=str(stratum_id)))

        arm_sizes_by_treatment_id[treatment_assignment] += 1
        # Fix linter error: ensure arm_id is not None
        arm_id = arms[treatment_assignment].arm_id
        if arm_id is None:
            raise ValueError(f"Arm at index {treatment_assignment} has no arm_id")

        participant = Assignment(
            participant_id=str(row_dict[id_col]),
            arm_id=arm_id,
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


def assign_treatments_with_balance(
    sa_table: Table,
    data: Sequence[RowProtocol],
    stratum_cols: list[str],
    id_col: str,
    n_arms: int,
    quantiles: int = 4,
    random_state: int | None = None,
) -> AssignmentResult:
    """
    Perform stratified random assignment and balance checking.

    Does lightweight preprocessing of the input data for use by our stats library.

    Args:
        sa_table: sqlalchemy table representation used for type info
        data: sqlalchemy result set of Rows representing units to be assigned
        stratum_cols: List of column names to stratify on
        id_col: Name of column containing unit identifiers
        n_arms: Number of treatment arms
        quantiles: number of buckets to use for stratification of numerics
        random_state: Random seed for reproducibility

    Returns:
        AssignmentResult containing assignments and balance check results
    """
    # Convert SQLAlchemy data to DataFrame
    df = DataFrame(data)

    # Extract Decimal column names from SQLAlchemy table and convert to float.
    # (Decimals are possible if the Table was created with SA's autoload instead of cursor).
    if decimals := [c.name for c in sa_table.columns if c.type.python_type is decimal.Decimal]:
        df[decimals] = df[decimals].astype(float)

    # Call the core assignment function
    return assign_treatment_and_check_balance(
        df=df,
        stratum_cols=stratum_cols,
        id_col=id_col,
        n_arms=n_arms,
        quantiles=quantiles,
        random_state=random_state,
    )


async def bulk_insert_arm_assignments(
    xngin_session: AsyncSession,
    experiment_id: str,
    arms: Sequence[Arm],
    participant_type: str,
    participant_id_col: str,
    data: Sequence[RowProtocol],
    assignment_result: AssignmentResult,
    stratum_id_name: str | None = None,
) -> None:
    """Bulk insert arm assignments into the database.

    Args:
        xngin_session: sqlalchemy session
        experiment_id: Unique identifier for experiment
        arms: Name & id of each treatment arm
        participant_type: Type of participant in the experiment
        participant_id_col: Name of column in `data` containing participant identifiers
        data: sqlalchemy result set of Rows representing units to be assigned
        assignment_result: AssignmentResult containing assignments and balance check results
        stratum_id_name: If you want to output the strata group ids, provide a non-null name for
                         the column to add to the assignment output as a Strata field.
    """
    participants_to_insert = []

    # Track if we originally had valid strata
    had_valid_strata = assignment_result.stratum_ids is not None
    stratum_ids = assignment_result.stratum_ids or [0] * len(assignment_result.treatment_ids)
    # These columns were the original columns to stratify on.
    orig_stratum_cols = assignment_result.orig_stratum_cols

    for stratum_id, treatment_assignment, row in zip(stratum_ids, assignment_result.treatment_ids, data, strict=True):
        row_dict = row._asdict()

        if not orig_stratum_cols:
            strata = []
        else:
            # Output the participant's strata values as seen at this time of assignment.
            strata = [
                Strata(
                    field_name=column,
                    strata_value=str(row_dict[column] if pd.notna(row_dict[column]) else "NA"),
                ).model_dump(mode="json")
                for column in orig_stratum_cols
            ]
            # Only add stratum_id if we had valid strata and stratum_id_name is provided
            if stratum_id_name is not None and had_valid_strata:
                strata.append(Strata(field_name=stratum_id_name, strata_value=str(stratum_id)).model_dump(mode="json"))

        arm_id = arms[treatment_assignment].arm_id
        if arm_id is None:
            raise ValueError(f"Arm at index {treatment_assignment} has no arm_id")

        participant = tables.ArmAssignment(
            experiment_id=experiment_id,
            participant_id=str(row_dict[participant_id_col]),
            participant_type=participant_type,
            arm_id=arm_id,
            strata=strata,
        )
        participants_to_insert.append(participant.to_dict())

    await xngin_session.execute(Insert(tables.ArmAssignment), participants_to_insert)
