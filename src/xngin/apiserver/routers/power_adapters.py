"""Bridges database queries with cluster power / ICC stats functions."""

from collections.abc import Mapping, Sequence

import numpy as np
from sqlalchemy import Table
from sqlalchemy.orm import Session

from xngin.apiserver.dwh.queries import get_cluster_sufficient_stats
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import Filter
from xngin.stats.cluster_icc import calculate_icc_from_sufficient_stats
from xngin.stats.stats_errors import StatsPowerError


def calculate_cluster_stats_from_database(
    session: Session,
    sa_table: Table,
    cluster_column: str,
    outcome_columns: Sequence[str],
    filters: list[Filter],
    outcome_shifts: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Calculate ICC and cluster statistics for one or more metrics from a DWH table.

    Fetches per-cluster sufficient statistics for all metrics in a single DWH query, then
    computes the stats in pure Python: only the query touches the database session.

    Args:
        session: SQLAlchemy session for the DWH
        sa_table: SQLAlchemy Table object
        cluster_column: Column name containing cluster IDs
        outcome_columns: Column names containing outcome values
        filters: List of filters to apply
        outcome_shifts: Optional per-outcome constants (e.g. approximate means) subtracted
            in SQL before summing, to keep the sums of squares numerically stable

    Returns:
        dict keyed by outcome column name, each value a dict with keys:
        icc, avg_cluster_size, cv
    """
    rows = get_cluster_sufficient_stats(session, sa_table, cluster_column, outcome_columns, filters, outcome_shifts)

    # Cluster sizes and CV are computed over every row with a cluster key, including rows
    # whose outcome values are null: metrics are outcomes that may be filled in as the
    # experiment runs, so the analysis-time cluster size is the full-population size.
    # Do not narrow this to each outcome's non-null rows.
    sizes = np.array([row["rows__count"] for row in rows], dtype=np.float64)
    avg_cluster_size = float(sizes.mean())
    # np.std defaults to the population stddev, matching the SQL stddev_pop this replaced.
    cv = float(sizes.std() / sizes.mean())

    stats_by_outcome: dict[str, dict[str, float]] = {}
    for outcome_column in outcome_columns:
        # ICC uses only clusters with values for this outcome, mirroring the per-metric
        # "outcome IS NOT NULL" filter used when each metric was queried separately.
        counts = np.array([row[f"{outcome_column}__count"] for row in rows], dtype=np.float64)
        has_values = counts > 0
        if not has_values.any():
            raise LateValidationError(
                f"No data found for cluster column '{cluster_column}' and outcome '{outcome_column}'"
            )
        try:
            icc = calculate_icc_from_sufficient_stats(
                counts=counts[has_values],
                sums=np.array([row[f"{outcome_column}__sum"] for row in rows], dtype=np.float64)[has_values],
                sumsqs=np.array([row[f"{outcome_column}__sumsq"] for row in rows], dtype=np.float64)[has_values],
            )
        except ValueError as verr:
            raise StatsPowerError.from_error(verr, outcome_column) from verr
        stats_by_outcome[outcome_column] = {
            "icc": icc,
            "avg_cluster_size": avg_cluster_size,
            "cv": cv,
        }

    return stats_by_outcome
