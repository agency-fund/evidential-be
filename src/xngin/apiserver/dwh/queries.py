import re
from datetime import datetime, timedelta

import sqlalchemy
from sqlalchemy import (
    Float,
    Integer,
    Label,
    and_,
    cast,
    or_,
    func,
    ColumnOperators,
    Table,
    not_,
    select,
    DateTime,
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


class LateValidationError(Exception):
    """Raised by API request validation failures that can only occur late in processing.

    Example: datetime value validations cannot happen until we know we are dealing with a datetime field, and that
    information is not available until we have table reflection data.
    """

    pass


def get_stats_on_metrics(
    session,
    sa_table: Table,
    metrics: list[DesignSpecMetricRequest],
    audience_spec: AudienceSpec,
) -> list[DesignSpecMetric]:
    # build our query
    metric_types = [
        MetricType.from_python_type(sa_table.c[m.field_name].type.python_type)
        for m in metrics
    ]
    select_columns: list[Label] = []
    for metric, metric_type in zip(metrics, metric_types, strict=False):
        field_name = metric.field_name
        col = sa_table.c[field_name]
        # Coerce everything to Float to avoid Decimal/Integer/Boolean issues across backends.
        if metric_type is MetricType.NUMERIC:
            cast_column = cast(col, Float)
        else:  # re: avg(boolean) doesn't work on pg-like backends
            cast_column = cast(cast(col, Integer), Float)
        # TODO(roboton): consider whether mitigations for null are important
        select_columns.extend((
            func.avg(cast_column).label(f"{field_name}__mean"),
            custom_functions.stddev_pop(cast_column).label(f"{field_name}__stddev"),
            func.count(col).label(f"{field_name}__count"),
        ))
    filters = create_query_filters_from_spec(sa_table, audience_spec)
    query = select(*select_columns).filter(*filters)
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
                metric_stddev=stats[f"{field_name}__stddev"]
                if metric_type is MetricType.NUMERIC
                else None,
                available_n=stats[f"{field_name}__count"],
            )
        )

    return metrics_to_return


def query_for_participants(
    session: Session,
    sa_table: Table,
    audience_spec: AudienceSpec,
    chosen_n: int,
):
    """Samples participants."""
    filters = create_query_filters_from_spec(sa_table, audience_spec)
    query = compose_query(sa_table, chosen_n, filters)
    return session.execute(query).all()


def create_query_filters_from_spec(
    sa_table: sqlalchemy.Table, audience_spec: AudienceSpec
):
    """Converts an AudienceSpec into a list of SQLAlchemy filters."""

    def create_one_filter(filter_: AudienceSpecFilter, sa_table: sqlalchemy.Table):
        if isinstance(sa_table.columns[filter_.field_name].type, DateTime):
            return create_datetime_filter(sa_table.columns[filter_.field_name], filter_)
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


def create_datetime_filter(
    col: sqlalchemy.Column, filter_: AudienceSpecFilter
) -> ColumnOperators:
    """Converts a single AudienceSpecFilter for a DateTime-typed column into a sqlalchemy filter."""

    def str_to_datetime(s: str | int | float | bool) -> datetime:
        """Convert an ISO8601 string to a timezone-unaware datetime.

        LateValidationError is raised if the ISO8601 string specifies a non-UTC timezone.

        For maximum compatibility between backends, any microseconds value, if specified, is truncated to zero.
        """
        try:
            parsed = datetime.fromisoformat(s).replace(microsecond=0)
        except (ValueError, TypeError) as exc:
            raise LateValidationError(
                "datetime-type filter values must be strings containing an ISO8601 formatted date."
            ) from exc
        if not parsed.tzinfo:
            return parsed
        offset = parsed.tzinfo.utcoffset(parsed)
        if offset == timedelta():  # 0 timedelta is equivalent to UTC
            return parsed.replace(tzinfo=None)
        raise LateValidationError(
            f"datetime-type filter values must be in UTC, or not be tagged with an explicit timezone: {s}"
        )

    match filter_.relation:
        case Relation.BETWEEN:
            match filter_.value:
                case (left, None):
                    return col >= str_to_datetime(left)
                case (None, right):
                    return col <= str_to_datetime(right)
                case (left, right):
                    return col.between(str_to_datetime(left), str_to_datetime(right))
        case _:
            raise LateValidationError(
                "The only valid Relation on a datetime field is BETWEEN."
            )


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
        case Relation.EXCLUDES if isinstance(col.type, sqlalchemy.Boolean):
            return and_(*[
                col.is_not(value) if value is not None else col.is_not(None)
                for value in filter_.value
            ])
        case Relation.EXCLUDES:
            return or_(col.is_(None), col.not_in(filter_.value))
        case Relation.INCLUDES if isinstance(col.type, sqlalchemy.Boolean):
            return or_(*[
                col.is_(value) if value is not None else col.is_(None)
                for value in filter_.value
            ])
        case Relation.INCLUDES:
            return col.in_(filter_.value)
        case Relation.INCLUDES:
            return col.in_(filter_.value)


def compose_query(sa_table: Table, chosen_n: int, filters):
    return (
        select(sa_table)
        .filter(*filters)
        .order_by(custom_functions.Random(sa_table=sa_table))
        .limit(chosen_n)
    )
