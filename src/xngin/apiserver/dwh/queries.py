from collections.abc import Sequence

import sqlalchemy
from sqlalchemy import (
    Float,
    Integer,
    Label,
    Select,
    Table,
    cast,
    distinct,
    func,
    select,
)
from sqlalchemy.engine.row import RowMapping
from sqlalchemy.orm import Session

from xngin.apiserver.dwh.inspection_types import FieldDescriptor
from xngin.apiserver.dwh.query_constructors import create_query_filters
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import (
    DesignSpecMetric,
    DesignSpecMetricRequest,
    Filter,
    GetFiltersResponseDiscrete,
    GetFiltersResponseElement,
    GetFiltersResponseNumericOrDate,
)
from xngin.apiserver.routers.common_enums import FilterClass, MetricType
from xngin.db_extensions import custom_functions


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
            custom_functions.stddev_pop(cast_column).label(f"{field_name}__stddev"),
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


def get_stats_on_filters(
    session: Session,
    sa_table: Table,
    db_schema: dict[str, FieldDescriptor],
    filter_schema: dict[str, FieldDescriptor],
    expensive: bool,
) -> list[GetFiltersResponseElement]:
    """Runs SELECT queries for metrics (min, max, distinct, etc) on filter fields.

    This async method runs the queries against the synchronous Session in a thread.

    Args:
        session: SQLAlchemy session for customer data warehouse
        sa_table: SQLAlchemy Table object
        db_schema: The latest table schema in the database described as FieldDescriptors
        filter_schema: The latest filter schema in the participant type config described as FieldDescriptors
        expensive: If true, we run expensive min/max/distinct queries on all the columns.

    Returns:
        A mapper function that takes (column_name, column_descriptor) and returns GetFiltersResponseElement
    """

    def query(col_name: str, ptype_fd: FieldDescriptor) -> GetFiltersResponseElement:
        db_col = db_schema.get(col_name)
        if not db_col:
            raise ValueError(f"Column {col_name} not found in schema.")

        filter_class = db_col.data_type.filter_class(col_name)

        # Collect metadata on the values in the database.
        sa_col = sa_table.columns[col_name]
        match filter_class:
            case FilterClass.DISCRETE:
                distinct_values = None
                if expensive:
                    stmt: Select = (
                        sqlalchemy.select(distinct(sa_col)).where(sa_col.is_not(None)).limit(1000).order_by(sa_col)
                    )
                    result_discrete = session.scalars(stmt)
                    distinct_values = [str(v) for v in result_discrete]
                return GetFiltersResponseDiscrete(
                    field_name=col_name,
                    data_type=db_col.data_type,
                    relations=filter_class.valid_relations(),
                    description=ptype_fd.description,
                    distinct_values=distinct_values,
                )
            case FilterClass.NUMERIC:
                min_, max_ = None, None
                if expensive:
                    min_, max_ = session.execute(
                        sqlalchemy.select(sqlalchemy.func.min(sa_col), sqlalchemy.func.max(sa_col)).where(
                            sa_col.is_not(None)
                        )
                    ).one()
                return GetFiltersResponseNumericOrDate(
                    field_name=col_name,
                    data_type=db_col.data_type,
                    relations=filter_class.valid_relations(),
                    description=ptype_fd.description,
                    min=min_,
                    max=max_,
                )
            case _:
                raise RuntimeError("unexpected filter class")

    return [query(col_name, ptype_fd) for col_name, ptype_fd in filter_schema.items() if db_schema.get(col_name)]


def get_cluster_size_stats(
    session: Session,
    sa_table: Table,
    cluster_column_name: str,
    filters: list[Filter],
) -> dict[str, float]:
    """Calculate cluster size statistics using SQL CTE."""
    if cluster_column_name not in sa_table.c:
        raise LateValidationError(f"Cluster column '{cluster_column_name}' not found in table")

    cluster_col = sa_table.c[cluster_column_name]
    filters_expr = create_query_filters(sa_table, filters)

    # CTE for cluster counts
    cluster_counts = (
        select(cluster_col.label("cluster_id"), func.count().label("cluster_size"))
        .where(cluster_col.is_not(None), *filters_expr)
        .group_by(cluster_col)
        .cte("cluster_counts")
    )

    # Aggregate statistics
    stats_query = select(
        func.avg(cluster_counts.c.cluster_size).label("avg_cluster_size"),
        func.min(cluster_counts.c.cluster_size).label("min_cluster_size"),
        func.max(cluster_counts.c.cluster_size).label("max_cluster_size"),
        (custom_functions.stddev_pop(cluster_counts.c.cluster_size) / func.avg(cluster_counts.c.cluster_size)).label(
            "cv"
        ),
        func.count().label("num_clusters"),
    )

    result = session.execute(stats_query).mappings().fetchone()

    if result is None or result["num_clusters"] == 0:
        raise LateValidationError(f"No clusters found in column '{cluster_column_name}'")

    return {
        "avg_cluster_size": float(result["avg_cluster_size"]),
        "min_cluster_size": int(result["min_cluster_size"]),
        "max_cluster_size": int(result["max_cluster_size"]),
        "cv": float(result["cv"]) if result["cv"] is not None else 0.0,
        "num_clusters": int(result["num_clusters"]),
    }


def get_cluster_outcome_data(
    session: Session,
    sa_table: Table,
    cluster_column_name: str,
    outcome_column_name: str,
    filters: list[Filter],
) -> Sequence[RowMapping]:
    """Fetch cluster and outcome data for ICC calculation.

    Each row returned is a SQLAlchemy ``RowMapping`` (by column name; same keys as
    ``cluster_column_name`` / ``outcome_column_name``). Outcomes are SQL-cast to Float.
    """
    if cluster_column_name not in sa_table.c:
        raise LateValidationError(f"Cluster column '{cluster_column_name}' not found in table")
    if outcome_column_name not in sa_table.c:
        raise LateValidationError(f"Outcome column '{outcome_column_name}' not found in table")

    cluster_col = sa_table.c[cluster_column_name]
    outcome_col = sa_table.c[outcome_column_name]
    filters_expr = create_query_filters(sa_table, filters)

    query = select(cluster_col, cast(outcome_col, Float)).where(
        cluster_col.is_not(None), outcome_col.is_not(None), *filters_expr
    )

    # Explicitly ask for dict-like RowMapping objects for downstream use of each row as a dict.
    results = session.execute(query).mappings().fetchall()

    if not results:
        raise LateValidationError(
            f"No data found for cluster column '{cluster_column_name}' and outcome '{outcome_column_name}'"
        )

    return results
