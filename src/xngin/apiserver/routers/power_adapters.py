"""Bridges database queries with cluster power / ICC stats functions."""

from collections.abc import Sequence

import pandas as pd
from sqlalchemy import Table
from sqlalchemy.orm import Session

from xngin.apiserver.dwh.queries import get_cluster_outcome_data
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import Filter
from xngin.stats.cluster_icc import calculate_icc_from_dataframe
from xngin.stats.stats_errors import StatsPowerError


def calculate_cluster_stats_from_database(
    session: Session,
    sa_table: Table,
    cluster_column: str,
    outcome_columns: Sequence[str],
    filters: list[Filter],
) -> dict[str, dict[str, float]]:
    """
    Calculate ICC and cluster statistics for one or more metrics from a DWH table.

    Fetches all metrics' outcome data in a single DWH query, then computes the stats in
    pandas. Orchestrates DWH queries with stats calculations, keeping the stats layer
    free of SQLAlchemy dependencies.

    Args:
        session: SQLAlchemy session for the DWH
        sa_table: SQLAlchemy Table object
        cluster_column: Column name containing cluster IDs
        outcome_columns: Column names containing outcome values
        filters: List of filters to apply

    Returns:
        dict keyed by outcome column name, each value a dict with keys:
        icc, avg_cluster_size, cv
    """
    data = get_cluster_outcome_data(session, sa_table, cluster_column, outcome_columns, filters)
    df = pd.DataFrame(data)

    # Cluster sizes and CV are computed over every row with a cluster key, including rows
    # whose outcome values are null: metrics are outcomes that may be filled in as the
    # experiment runs, so the analysis-time cluster size is the full-population size.
    # Do not narrow this to each outcome's non-null rows.
    sizes = df.groupby(cluster_column).size()
    avg_cluster_size = float(sizes.mean())
    # ddof=0 is the population stddev, matching the SQL stddev_pop this replaced.
    cv = float(sizes.std(ddof=0) / sizes.mean())

    stats_by_outcome: dict[str, dict[str, float]] = {}
    for outcome_column in outcome_columns:
        # ICC uses only rows with a value for this outcome, mirroring the per-metric
        # "outcome IS NOT NULL" filter used when each metric was queried separately.
        outcome_df = df[[cluster_column, outcome_column]].dropna(subset=[outcome_column])
        if outcome_df.empty:
            raise LateValidationError(
                f"No data found for cluster column '{cluster_column}' and outcome '{outcome_column}'"
            )
        try:
            icc = calculate_icc_from_dataframe(outcome_df, cluster_column=cluster_column, outcome_column=outcome_column)
        except ValueError as verr:
            raise StatsPowerError.from_error(verr, outcome_column) from verr
        stats_by_outcome[outcome_column] = {
            "icc": icc,
            "avg_cluster_size": avg_cluster_size,
            "cv": cv,
        }

    return stats_by_outcome
