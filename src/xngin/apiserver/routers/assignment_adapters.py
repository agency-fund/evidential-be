"""Bridges our API types with assignment logic that operates on DataFrames."""

import decimal
import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import orjson
import pandas as pd
from pandas import DataFrame
from psycopg.types.json import Jsonb
from sqlalchemy import Table, insert
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.routers.common_api_types import BalanceCheck, StrataTypedDict
from xngin.apiserver.sql.queries import with_driver_connection
from xngin.apiserver.sqla import tables
from xngin.ops import performance
from xngin.stats.assignment import AssignmentResult, assign_treatment_and_check_balance
from xngin.stats.balance import BalanceResult


class RowProtocol(Protocol):
    """Minimal methods to approximate a sqlalchemy.engine.row.Row for testing."""

    @property
    def _mapping(self) -> Mapping[str, Any]:
        """https://docs.sqlalchemy.org/en/20/core/connections.html#sqlalchemy.engine.Row._mapping"""
        ...


def _is_present_scalar(value: object) -> bool:
    """Return whether a scalar should be serialized or rendered as "NA".

    This is much faster than pd.notna and equivalent for our use cases.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return False
    if isinstance(value, float) or type(value) is decimal.Decimal:
        return not math.isnan(value)
    return True


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


def assign_treatments_with_balance(
    sa_table: Table,
    data: Sequence[RowProtocol],
    stratum_cols: list[str],
    id_col: str,
    n_arms: int,
    quantiles: int = 4,
    random_state: int | None = None,
    arm_weights: list[float] | None = None,
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
        arm_weights: Optional list of weights (summing to 100) for unbalanced arm allocation

    Returns:
        AssignmentResult containing assignments and balance check results
    """
    # Convert SQLAlchemy data to DataFrame
    df = DataFrame(data)

    # Extract Decimal column names from SQLAlchemy table and convert to float.
    # (Decimals are possible if the Table was created with SA's autoload instead of cursor).
    if decimals := [c.name for c in sa_table.columns if c.type.python_type is decimal.Decimal and c.name in df]:
        df[decimals] = df[decimals].astype(float)

    # Call the core assignment function
    return assign_treatment_and_check_balance(
        df=df,
        stratum_cols=stratum_cols,
        id_col=id_col,
        n_arms=n_arms,
        quantiles=quantiles,
        random_state=random_state,
        arm_weights=arm_weights,
    )


async def bulk_insert_arm_assignments(
    xngin_session: AsyncSession,
    experiment_id: str,
    arm_ids: list[str],
    participant_type: str,
    participant_id_col: str,
    data: Sequence[RowProtocol],
    assignment_result: AssignmentResult,
    stratum_id_name: str | None = None,
) -> None:
    """Bulk insert arm assignments into the database via async COPY.

    Args:
        xngin_session: sqlalchemy session
        experiment_id: Database ID of the experiment
        arm_ids: Database ID of each treatment arm, ordered by arm index used in assignment.
        participant_type: Type of participant in the experiment
        participant_id_col: Name of column in `data` containing participant identifiers
        data: sqlalchemy result set of Rows representing units to be assigned
        assignment_result: AssignmentResult containing assignments and balance check results. AssignmentResult.arm_pop
          indexes are parallel to indexes on arm_ids.
        stratum_id_name: If you want to output the strata group ids, provide a non-null name for
                         the column to add to the assignment output as a Strata field.
    """
    with performance.timing("_bulk_insert_async"):
        await _bulk_insert_async(
            xngin_session=xngin_session,
            experiment_id=experiment_id,
            arm_ids=arm_ids,
            participant_type=participant_type,
            participant_id_col=participant_id_col,
            data=data,
            assignment_result=assignment_result,
            stratum_id_name=stratum_id_name,
        )

    arm_stats_rows = [
        {"arm_id": arm_id, "population": int(assignment_result.arm_pop[i])} for i, arm_id in enumerate(arm_ids)
    ]
    await xngin_session.execute(insert(tables.ArmStats).values(arm_stats_rows))


async def _bulk_insert_async(
    *,
    xngin_session: AsyncSession,
    experiment_id: str,
    arm_ids: list[str],
    participant_type: str,
    participant_id_col: str,
    data: Sequence[RowProtocol],
    assignment_result: AssignmentResult,
    stratum_id_name: str | None,
) -> None:
    """Write arm assignments in bulk via COPY on the session's driver connection."""
    # Track if we originally had valid strata
    had_valid_strata = assignment_result.stratum_ids is not None
    stratum_ids = assignment_result.stratum_ids or [0] * len(assignment_result.treatment_ids)
    # These columns were the original columns to stratify on.
    orig_stratum_cols = assignment_result.orig_stratum_cols

    copy_sql = "COPY arm_assignments (experiment_id, participant_id, participant_type, arm_id, strata) FROM STDIN"
    async with (
        with_driver_connection(xngin_session) as driver_conn,
        driver_conn.cursor() as cur,
        cur.copy(copy_sql) as copy,
    ):
        copy.set_types(["text", "text", "text", "text", "jsonb"])
        for stratum_id, treatment_assignment, row in zip(
            stratum_ids, assignment_result.treatment_ids, data, strict=True
        ):
            row_mapping = row._mapping

            strata: list[StrataTypedDict]  # Use TypedDict to avoid runtime overhead of Pydantic.
            if not orig_stratum_cols:
                strata = []
            else:
                # Output the participant's strata values as seen at this time of assignment.
                strata = [
                    {
                        "field_name": column,
                        "strata_value": str(row_mapping[column]) if _is_present_scalar(row_mapping[column]) else "NA",
                    }
                    for column in orig_stratum_cols
                ]
                # Only add stratum_id if we had valid strata and stratum_id_name is provided
                if stratum_id_name is not None and had_valid_strata:
                    strata.append({"field_name": stratum_id_name, "strata_value": str(stratum_id)})

            arm_id = arm_ids[treatment_assignment]
            await copy.write_row((
                experiment_id,
                str(row_mapping[participant_id_col]),
                participant_type,
                arm_id,
                Jsonb(strata, dumps=orjson.dumps),
            ))
