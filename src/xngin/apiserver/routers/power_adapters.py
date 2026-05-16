"""Bridges database queries with cluster power / ICC stats functions."""

import pandas as pd
from sqlalchemy.orm import Session

from xngin.apiserver.dwh.queries import get_cluster_outcome_data, get_cluster_size_stats
from xngin.apiserver.routers.common_api_types import Filter
from xngin.stats.cluster_icc import calculate_icc_from_dataframe
from xngin.stats.stats_errors import StatsPowerError


def calculate_icc_and_cv_from_database(
    session: Session,
    sa_table,
    cluster_column: str,
    outcome_column: str,
    filters: list[Filter],
) -> dict[str, float]:
    """
    Calculate ICC and cluster statistics from a DWH table.

    Orchestrates DWH queries with stats calculations, keeping the stats layer
    free of SQLAlchemy dependencies.

    Args:
        session: SQLAlchemy session for the DWH
        sa_table: SQLAlchemy Table object
        cluster_column: Column name containing cluster IDs
        outcome_column: Column name containing outcome values
        filters: List of filters to apply

    Returns:
        dict with keys: icc, avg_cluster_size, cv
    """
    cluster_stats = get_cluster_size_stats(session, sa_table, cluster_column, filters)

    data = get_cluster_outcome_data(session, sa_table, cluster_column, outcome_column, filters)

    df = pd.DataFrame(data)
    try:
        icc = calculate_icc_from_dataframe(df, cluster_column=cluster_column, outcome_column=outcome_column)
    except ValueError as verr:
        raise StatsPowerError(verr, outcome_column) from verr

    return {
        "icc": icc,
        "avg_cluster_size": cluster_stats["avg_cluster_size"],
        "cv": cluster_stats["cv"],
    }
