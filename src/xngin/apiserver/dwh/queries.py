from collections.abc import Mapping, Sequence

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
    DesignSpecMetricRequest,
    Filter,
)
from xngin.apiserver.routers.common_enums import MetricType


def get_raw_metric_stats(
    session: Session,
    sa_table: Table,
    metrics: list[DesignSpecMetricRequest],
    filters: list[Filter],
) -> RowMapping:
    """Fetch aggregate metric statistics in a single query.

    Returns one row with ``rows__count`` (total rows matching the filters) and, per
    metric, ``{name}__mean``, ``{name}__stddev``, and ``{name}__count`` over its non-null
    values. Use build_metric_stats to turn the row into DesignSpecMetric objects.
    """
    missing_metrics = {m.field_name for m in metrics if m.field_name not in sa_table.c}
    if len(missing_metrics) > 0:
        raise LateValidationError(f"Missing metrics (check your Datasource configuration): {missing_metrics}")

    # Include in our list of stats a total count of rows targeted by the audience filters,
    # whereas the individual aggregate functions per metric ignore NULLs by default.
    select_columns: list[Label] = [func.count().label("rows__count")]
    for metric in metrics:
        field_name = metric.field_name
        col = sa_table.c[field_name]
        # Coerce everything to Float to avoid Decimal/Integer/Boolean issues across backends.
        if MetricType.from_python_type(col.type.python_type) is MetricType.NUMERIC:
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
    return session.execute(query).mappings().one()


def get_cluster_outcome_data(
    session: Session,
    sa_table: Table,
    cluster_column_name: str,
    outcome_column_names: Sequence[str],
    filters: list[Filter],
    outcome_shifts: Mapping[str, float] | None = None,
) -> Sequence[RowMapping]:
    """Fetch per-cluster sufficient statistics for cluster power calculations.

    Returns one ``RowMapping`` per cluster with ``rows__count`` (all rows, including rows
    whose outcomes are null: metrics are outcomes that may be filled in as the experiment
    runs, so cluster-size statistics must count the full population) and, per outcome,
    ``{name}__count``, ``{name}__sum``, and ``{name}__sumsq`` over that outcome's non-null
    values. ICC and cluster-size statistics can be computed exactly from these without
    fetching individual rows. Outcomes are SQL-cast to Float.

    ``outcome_shifts`` optionally maps outcome names to a constant subtracted from each
    value before summing (e.g. the metric's approximate mean). ICC is shift-invariant, and
    centered sums avoid the precision loss of summing squares of large raw values.
    """
    if cluster_column_name not in sa_table.c:
        raise LateValidationError(f"Cluster column '{cluster_column_name}' not found in table")
    for outcome_column_name in outcome_column_names:
        if outcome_column_name not in sa_table.c:
            raise LateValidationError(f"Outcome column '{outcome_column_name}' not found in table")

    cluster_col = sa_table.c[cluster_column_name]
    filters_expr = create_query_filters(sa_table, filters)

    select_columns: list[Label] = [func.count().label("rows__count")]
    for outcome_column_name in outcome_column_names:
        outcome_col = sa_table.c[outcome_column_name]
        # PostgreSQL cannot cast BOOLEAN directly to FLOAT; go through INTEGER first.
        if outcome_col.type.python_type is bool:
            cast_outcome = cast(cast(outcome_col, Integer), Float)
        else:
            cast_outcome = cast(outcome_col, Float)
        shift = (outcome_shifts or {}).get(outcome_column_name, 0.0)
        shifted = cast_outcome - shift
        select_columns.extend((
            func.count(outcome_col).label(f"{outcome_column_name}__count"),
            func.sum(shifted).label(f"{outcome_column_name}__sum"),
            func.sum(shifted * shifted).label(f"{outcome_column_name}__sumsq"),
        ))

    query = select(*select_columns).where(cluster_col.is_not(None), *filters_expr).group_by(cluster_col)

    results = session.execute(query).mappings().fetchall()

    if not results:
        raise LateValidationError(f"No clusters found in column '{cluster_column_name}'")

    return results
