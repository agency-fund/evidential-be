"""Pure computations turning raw DWH query rows into power / ICC stats.

These functions take the rows fetched by the queries in dwh/queries.py and never touch a
database session, so callers can run them after the DwhSession block has closed.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sqlalchemy import Table
from sqlalchemy.engine.row import RowMapping

from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import DesignSpecMetric, DesignSpecMetricRequest
from xngin.apiserver.routers.common_enums import MetricType
from xngin.stats.cluster_icc import calculate_icc_from_sufficient_stats
from xngin.stats.stats_errors import StatsPowerError


def build_metric_stats(
    raw_stats: Mapping[str, Any] | RowMapping,
    sa_table: Table,
    metrics: list[DesignSpecMetricRequest],
) -> list[DesignSpecMetric]:
    """Build DesignSpecMetric objects from the row returned by get_raw_metric_stats."""
    metrics_to_return = []
    for metric in metrics:
        field_name = metric.field_name
        metric_type = MetricType.from_python_type(sa_table.c[field_name].type.python_type)
        metrics_to_return.append(
            DesignSpecMetric(
                field_name=field_name,
                metric_pct_change=metric.metric_pct_change,
                metric_target=metric.metric_target,
                metric_type=metric_type,
                metric_baseline=raw_stats[f"{field_name}__mean"],
                metric_stddev=raw_stats[f"{field_name}__stddev"] if metric_type is MetricType.NUMERIC else None,
                available_nonnull_n=raw_stats[f"{field_name}__count"],
                # This value is the same across all metrics, but we replicate for convenience:
                available_n=raw_stats["rows__count"],
            )
        )

    return metrics_to_return


def calculate_cluster_stats(
    cluster_rows: Sequence[Mapping[str, Any] | RowMapping],
    cluster_column: str,
    outcome_columns: Sequence[str],
) -> dict[str, dict[str, float]]:
    """
    Calculate ICC and cluster statistics for one or more metrics from per-cluster
    sufficient statistics, as returned by get_cluster_sufficient_stats.

    Args:
        cluster_rows: One mapping per cluster with rows__count and per-outcome
            {name}__count / {name}__sum / {name}__sumsq keys
        cluster_column: Cluster column name, used in error messages
        outcome_columns: Column names containing outcome values

    Returns:
        dict keyed by outcome column name, each value a dict with keys:
        icc, avg_cluster_size, cv
    """
    # Cluster sizes and CV are computed over every row with a cluster key, including rows
    # whose outcome values are null: metrics are outcomes that may be filled in as the
    # experiment runs, so the analysis-time cluster size is the full-population size.
    # Do not narrow this to each outcome's non-null rows.
    sizes = np.array([row["rows__count"] for row in cluster_rows], dtype=np.float64)
    avg_cluster_size = float(sizes.mean())
    # np.std defaults to the population stddev, matching the SQL stddev_pop this replaced.
    cv = float(sizes.std() / sizes.mean())

    stats_by_outcome: dict[str, dict[str, float]] = {}
    for outcome_column in outcome_columns:
        # ICC uses only clusters with values for this outcome, mirroring the per-metric
        # "outcome IS NOT NULL" filter used when each metric was queried separately.
        counts = np.array([row[f"{outcome_column}__count"] for row in cluster_rows], dtype=np.float64)
        has_values = counts > 0
        if not has_values.any():
            raise LateValidationError(
                f"No data found for cluster column '{cluster_column}' and outcome '{outcome_column}'"
            )
        try:
            icc = calculate_icc_from_sufficient_stats(
                counts=counts[has_values],
                sums=np.array([row[f"{outcome_column}__sum"] for row in cluster_rows], dtype=np.float64)[has_values],
                sumsqs=np.array([row[f"{outcome_column}__sumsq"] for row in cluster_rows], dtype=np.float64)[
                    has_values
                ],
            )
        except ValueError as verr:
            raise StatsPowerError.from_error(verr, outcome_column) from verr
        stats_by_outcome[outcome_column] = {
            "icc": icc,
            "avg_cluster_size": avg_cluster_size,
            "cv": cv,
        }

    return stats_by_outcome
