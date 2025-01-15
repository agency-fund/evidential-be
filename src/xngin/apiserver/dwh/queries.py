import re

import sqlalchemy
from sqlalchemy import (
    Float,
    Integer,
    Label,
    cast,
    or_,
    func,
    ColumnOperators,
    Table,
    not_,
    select,
)
from sqlalchemy.orm import Session

from xngin.apiserver.api_types import (
    AudienceSpec,
    DesignSpecMetricRequest,
    MetricType,
    Relation,
    AudienceSpecFilter,
    EXPERIMENT_IDS_SUFFIX,
    DesignSpecMetric,
)
from xngin.db_extensions import custom_functions


def get_stats_on_metrics(
    session,
    sa_table: Table,
    metrics: list[DesignSpecMetricRequest],
    audience_spec: AudienceSpec,
) -> list[DesignSpecMetric]:
    # First prep the list that will hold our annotated metrics to return:
    def init_metric_to_return(metric: DesignSpecMetricRequest) -> DesignSpecMetric:
        """Make a copy of metric while deriving type from dwh."""
        # Union a dict representation of the input metric with the metric_type and create the output metric object.
        return DesignSpecMetric.model_validate(
            metric.model_dump()
            | {
                "metric_type": MetricType.from_python_type(
                    sa_table.c[metric.field_name].type.python_type
                )
            }
        )

    metrics_to_return = [init_metric_to_return(m) for m in metrics]

    # now build our query
    select_columns: list[Label] = []
    for metric in metrics_to_return:
        field_name = metric.field_name
        col = sa_table.c[field_name]
        # Coerce everything to Float to avoid Decimal/Integer/Boolean issues across backends.
        if metric.metric_type is MetricType.NUMERIC:
            cast_column = cast(col, Float)
        else:  # re: avg(boolean) doesn't work on pg-like backends
            cast_column = cast(cast(col, Integer), Float)
        # TODO(roboton): consider whether mitigations for null are important
        select_columns.extend((
            func.avg(cast_column).label(f"{field_name}__mean"),
            custom_functions.stddev_pop(cast_column).label(f"{field_name}__stddev"),
            func.count(col).label(f"{field_name}__count"),
        ))
    filters = create_filters(sa_table, audience_spec)
    query = select(*select_columns).filter(*filters)
    stats = session.execute(query).mappings().fetchone()

    # finally backfill with the stats
    for metric in metrics_to_return:
        field_name = metric.field_name
        metric.metric_baseline = stats[f"{field_name}__mean"]
        metric.metric_stddev = stats[f"{field_name}__stddev"]
        metric.available_n = stats[f"{field_name}__count"]

    return metrics_to_return


def query_for_participants(
    session: Session,
    sa_table: Table,
    audience_spec: AudienceSpec,
    chosen_n: int,
):
    """Samples participants."""
    filters = create_filters(sa_table, audience_spec)
    query = compose_query(sa_table, chosen_n, filters)
    return session.execute(query).all()


# TODO: rename for clarity
def create_filters(sa_table: sqlalchemy.Table, audience_spec: AudienceSpec):
    """Converts an AudienceSpec into a list of SQLAlchemy filters."""

    def create_one_filter(filter_: AudienceSpecFilter, sa_table: sqlalchemy.Table):
        if filter_.field_name.endswith(EXPERIMENT_IDS_SUFFIX):
            return create_special_experiment_id_filter(
                sa_table.columns[filter_.field_name], filter_
            )
        return create_filter(sa_table.columns[filter_.field_name], filter_)

    return [create_one_filter(filter_, sa_table) for filter_ in audience_spec.filters]


def create_special_experiment_id_filter(
    col: sqlalchemy.Column, filter_: AudienceSpecFilter
) -> ColumnOperators:
    matching_regex = make_csv_regex(filter_.value)
    match filter_.relation:
        case Relation.INCLUDES:
            return func.lower(col).regexp_match(matching_regex)
        case Relation.EXCLUDES:
            return or_(
                col.is_(None),
                func.char_length(col) == 0,
                not_(func.lower(col).regexp_match(matching_regex)),
            )
    # This should be impossible as it's caught by the AudienceSpecFilter validator:
    raise ValueError(
        f"Experiment id filter on {filter_.field_name} has invalid relation: {filter_.relation}"
    )


def make_csv_regex(values):
    """Constructs a regular expression for matching a CSV string against a list of values.

    The generated regexp is to be used by re.search() or equivalent. We assume that most database engines
    will support identical syntax.
    """
    value_regexp = (
        r"("
        + r"|".join(re.escape(str(v).lower()) for v in values if v is not None)
        + r")"
    )
    return r"(^x$)|(^x,)|(,x$)|(,x,)".replace("x", value_regexp)


def create_filter(
    col: sqlalchemy.Column, filter_: AudienceSpecFilter
) -> ColumnOperators:
    """Converts a single AudienceSpecFilter to a sqlalchemy filter."""
    match filter_.relation:
        case Relation.BETWEEN:
            match filter_.value:
                case (left, None):
                    return col >= left
                case (None, right):
                    return col <= right
                case (left, right):
                    return col.between(left, right)
        case Relation.EXCLUDES:
            return or_(col.is_(None), col.not_in(filter_.value))
        case Relation.INCLUDES:
            return col.in_(filter_.value)


def compose_query(sa_table: Table, chosen_n: int, filters):
    return (
        select(sa_table)
        .filter(*filters)
        .order_by(custom_functions.our_random(sa_table))
        .limit(chosen_n)
    )
