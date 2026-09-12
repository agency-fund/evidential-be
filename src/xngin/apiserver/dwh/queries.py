from collections.abc import Sequence

from sqlalchemy import (
    Float,
    Integer,
    Label,
    Table,
    cast,
    func,
    select,
)
from sqlalchemy.engine.row import RowMapping
from sqlalchemy.orm import Session

from xngin.apiserver.dwh.query_constructors import create_query_filters
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import (
    DesignSpecMetric,
    DesignSpecMetricRequest,
    Filter,
)
from xngin.apiserver.routers.common_enums import MetricType


def get_stats_on_metrics(
    session,
    sa_table: Table,
    metrics: list[DesignSpecMetricRequest],
    filters: list[Filter],
) -> list[DesignSpecMetric]:
    missing_metrics = {m.field_name for m in metrics if m.field_name not in sa_table.c}
    if len(missing_metrics) > 0:
        raise LateValidationError(f"Missing metrics (check your Datasource configuration): {missing_metrics}")

    # build our query
    metric_types = [MetricType.from_python_type(sa_table.c[m.field_name].type.python_type) for m in metrics]
    # Include in our list of stats a total count of rows targeted by the audience filters,
    # whereas the individual aggregate functions per metric ignore NULLs by default.
    select_columns: list[Label] = [func.count().label("rows__count")]
    for metric, metric_type in zip(metrics, metric_types, strict=False):
        field_name = metric.field_name
        col = sa_table.c[field_name]
        # Coerce everything to Float to avoid Decimal/Integer/Boolean issues across backends.
        if metric_type is MetricType.NUMERIC:
            cast_column = cast(col, Float)
        else:  # re: avg(boolean) doesn't work on pg-like backends
            cast_column = cast(cast(col, Integer), Float)
        select_columns.extend((
            func.avg(cast_column).label(f"{field_name}__mean"),
            func.stddev_pop(cast_column).label(f"{field_name}__stddev"),
            func.count(col).label(f"{field_name}__count"),
        ))
    filters_expr = create_query_filters(sa_table, filters)
    query = select(*select_columns).where(*filters_expr)
    stats = session.execute(query).mappings().fetchone()

    # finally backfill with the stats
    metrics_to_return = []
    for metric, metric_type in zip(metrics, metric_types, strict=False):
        field_name = metric.field_name
        metrics_to_return.append(
            DesignSpecMetric(
                field_name=metric.field_name,
                metric_pct_change=metric.metric_pct_change,
                metric_target=metric.metric_target,
                metric_type=metric_type,
                metric_baseline=stats[f"{field_name}__mean"],
                metric_stddev=stats[f"{field_name}__stddev"] if metric_type is MetricType.NUMERIC else None,
                available_nonnull_n=stats[f"{field_name}__count"],
                # This value is the same across all metrics, but we replicate for convenience:
                available_n=stats["rows__count"],
            )
        )

    return metrics_to_return


def get_cluster_outcome_data(
    session: Session,
    sa_table: Table,
    cluster_column_name: str,
    outcome_column_names: Sequence[str],
    filters: list[Filter],
) -> Sequence[RowMapping]:
    """Fetch cluster and outcome data for cluster power statistics in a single query.

    Each row returned is a SQLAlchemy ``RowMapping`` (by column name; same keys as
    ``cluster_column_name`` / ``outcome_column_names``). Outcomes are SQL-cast to Float.

    Rows are restricted to non-null cluster keys, but rows where an outcome is null are
    included (with a None value): metrics are outcomes that may be filled in as the
    experiment runs, so cluster-size statistics must count the full population, while ICC
    calculations drop each outcome's nulls individually.
    """
    if cluster_column_name not in sa_table.c:
        raise LateValidationError(f"Cluster column '{cluster_column_name}' not found in table")
    for outcome_column_name in outcome_column_names:
        if outcome_column_name not in sa_table.c:
            raise LateValidationError(f"Outcome column '{outcome_column_name}' not found in table")

    cluster_col = sa_table.c[cluster_column_name]
    filters_expr = create_query_filters(sa_table, filters)

    outcome_cols = []
    for outcome_column_name in outcome_column_names:
        outcome_col = sa_table.c[outcome_column_name]
        # PostgreSQL cannot cast BOOLEAN directly to FLOAT; go through INTEGER first.
        if outcome_col.type.python_type is bool:
            cast_outcome = cast(cast(outcome_col, Integer), Float)
        else:
            cast_outcome = cast(outcome_col, Float)
        outcome_cols.append(cast_outcome.label(outcome_column_name))

    query = select(cluster_col, *outcome_cols).where(cluster_col.is_not(None), *filters_expr)

    # Explicitly ask for dict-like RowMapping objects for downstream use of each row as a dict.
    results = session.execute(query).mappings().fetchall()

    if not results:
        raise LateValidationError(f"No clusters found in column '{cluster_column_name}'")

    return results
