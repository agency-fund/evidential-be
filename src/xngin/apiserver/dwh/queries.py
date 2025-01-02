import re

import sqlalchemy
from sqlalchemy import Float, cast, or_, func, ColumnOperators, Table, not_, select
from sqlalchemy.orm import Session

from xngin.apiserver.api_types import (
    AudienceSpec,
    MetricType,
    Relation,
    AudienceSpecFilter,
    EXPERIMENT_IDS_SUFFIX,
    DesignSpecMetric,
)
from xngin.sqlite_extensions import custom_functions


def get_metric_meta():
    # TODO: implement
    pass


def get_stats_on_metrics(
    session,
    sa_table: Table,
    metrics: list[DesignSpecMetric],
    audience_spec: AudienceSpec,
) -> list[DesignSpecMetric]:
    metric_columns = []

    for metric in metrics:
        metric_name = metric.metric_name
        col = sa_table.c[metric_name]
        # TODO(roboton): consider whether mitigations for null are important
        metric_columns.extend((
            func.avg(cast(col, Float)).label(f"{metric_name}__mean"),
            custom_functions.stddev_pop(cast(col, Float)).label(
                f"{metric_name}__stddev"
            ),
            func.count(col).label(f"{metric_name}__count"),
        ))
    query = select(*metric_columns)
    filters = create_filters(sa_table, audience_spec)
    query = query.filter(*filters)
    stats = session.execute(query).mappings().fetchone()

    metrics_with_stats = []
    for metric in metrics:
        metric_name = metric.metric_name
        metric_with_stats = metric.model_copy()
        # Derive type from the dwh if not supplied in the spec:
        if metric_with_stats.metric_type is None:
            metric_with_stats.metric_type = MetricType.from_python_type(
                sa_table.c[metric_name].type.python_type
            )
        # Explicit cast to float in case the db mean is a decimal.Decimal
        metric_with_stats.metric_baseline = float(stats[f"{metric_name}__mean"])
        metric_with_stats.metric_stddev = stats[f"{metric_name}__stddev"]
        metric_with_stats.available_n = stats[f"{metric_name}__count"]
        metrics_with_stats.append(metric_with_stats)

    return metrics_with_stats


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

    def create_one_filter(filter_, sa_table):
        if filter_.filter_name.endswith(EXPERIMENT_IDS_SUFFIX):
            return create_special_experiment_id_filter(
                sa_table.columns[filter_.filter_name], filter_
            )
        return create_filter(sa_table.columns[filter_.filter_name], filter_)

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
