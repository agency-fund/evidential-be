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
    AudienceSpecFilter,
    DesignSpecMetric,
    DesignSpecMetricRequest,
    EXPERIMENT_IDS_SUFFIX,
    FilterValueTypes,
    MetricType,
    ParticipantOutcome,
    Relation,
)
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.db_extensions import custom_functions


def get_stats_on_metrics(
    session,
    sa_table: Table,
    metrics: list[DesignSpecMetricRequest],
    audience_spec: AudienceSpec,
) -> list[DesignSpecMetric]:
    missing_metrics = {m.field_name for m in metrics if m.field_name not in sa_table.c}
    if len(missing_metrics) > 0:
        raise LateValidationError(
            f"Missing metrics (check your Datasource configuration): {missing_metrics}"
        )

    # build our query
    metric_types = [
        MetricType.from_python_type(sa_table.c[m.field_name].type.python_type)
        for m in metrics
    ]
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
                available_nonnull_n=stats[f"{field_name}__count"],
                # This value is the same across all metrics, but we replicate for convenience:
                available_n=stats["rows__count"],
            )
        )

    return metrics_to_return


def get_participant_metrics(
    session,
    sa_table: Table,
    metrics: list[DesignSpecMetricRequest],
    unique_id_field: str,
    participant_ids: list[str],
) -> list[ParticipantOutcome]:
    missing_metrics = {m.field_name for m in metrics if m.field_name not in sa_table.c}
    if len(missing_metrics) > 0:
        raise LateValidationError(
            f"Missing metrics (check your Datsource configuration): {missing_metrics}"
        )

    # build our query
    metric_types = [
        MetricType.from_python_type(sa_table.c[m.field_name].type.python_type)
        for m in metrics
    ]
    # Include in our list of stats a total count of rows targeted by the audience filters,
    # whereas the individual aggregate functions per metric ignore NULLs by default.
    # select_columns: list[Label] = [func.count().label("rows__count")]

    # select participant_id field
    select_columns: list[Label] = [sa_table.c[unique_id_field]]

    # add metrics from the experiment design
    for metric, metric_type in zip(metrics, metric_types, strict=False):
        field_name = metric.field_name
        col = sa_table.c[field_name]
        # Coerce everything to Float to avoid Decimal/Integer/Boolean issues across backends.
        if metric_type is MetricType.NUMERIC:
            cast_column = cast(col, Float)
        else:  # re: avg(boolean) doesn't work on pg-like backends
            cast_column = cast(cast(col, Integer), Float)
        select_columns.extend(cast_column)

    # create a single filter, filtering on the unique_id_field using
    # participant_ids from the treatment assignment.
    participant_id_filter = AudienceSpecFilter(
        field_name=unique_id_field, relation=Relation.INCLUDES, value=participant_ids
    )
    participant_filter = create_one_filter(participant_id_filter, sa_table)
    query = select(*select_columns).filter(participant_filter)

    return session.execute(query).mappings().fetchone()

    # finally backfill with the stats
    # metrics_to_return = []
    # for metric, metric_type in zip(metrics, metric_types, strict=False):
    #     field_name = metric.field_name
    #     metrics_to_return.append(
    #         DesignSpecMetric(
    #             field_name=metric.field_name,
    #             metric_pct_change=metric.metric_pct_change,
    #             metric_target=metric.metric_target,
    #             metric_type=metric_type,
    #             metric_baseline=stats[f"{field_name}__mean"],
    #             metric_stddev=stats[f"{field_name}__stddev"]
    #             if metric_type is MetricType.NUMERIC
    #             else None,
    #             available_nonnull_n=stats[f"{field_name}__count"],
    #             # This value is the same across all metrics, but we replicate for convenience:
    #             available_n=stats["rows__count"],
    #         )
    #     )


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


def create_one_filter(filter_: AudienceSpecFilter, sa_table: sqlalchemy.Table):
    """Converts an AudienceSpecFilter into a SQLAlchemy filter."""
    if isinstance(sa_table.columns[filter_.field_name].type, DateTime):
        return create_datetime_filter(sa_table.columns[filter_.field_name], filter_)
    if filter_.field_name.endswith(EXPERIMENT_IDS_SUFFIX):
        return create_special_experiment_id_filter(
            sa_table.columns[filter_.field_name], filter_
        )
    return create_filter(sa_table.columns[filter_.field_name], filter_)


def create_query_filters_from_spec(
    sa_table: sqlalchemy.Table, audience_spec: AudienceSpec
):
    """Converts an AudienceSpec into a list of SQLAlchemy filters."""

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


def general_excludes_filter(col: sqlalchemy.Column, value: list[FilterValueTypes]):
    if None in value:
        non_null_list = [v for v in value if v is not None]
        if len(non_null_list) == 0:
            return col.is_not(sqlalchemy.null())
        return and_(
            col.is_not(sqlalchemy.null()),
            col.not_in(non_null_list),
        )
    # Else if we didn't explicitly exclude NULL, explicitly include it now
    return or_(
        col.is_(None),
        col.not_in(value),
    )


def create_datetime_filter(
    col: sqlalchemy.Column, filter_: AudienceSpecFilter
) -> ColumnOperators:
    """Converts a single AudienceSpecFilter for a DateTime-typed column into a sqlalchemy filter."""

    def str_to_datetime(s: int | float | str | None) -> datetime | None:
        """Convert an ISO8601 string to a timezone-unaware datetime.

        LateValidationError is raised if the ISO8601 string specifies a non-UTC timezone.

        For maximum compatibility between backends, any microseconds value, if specified, is truncated to zero.
        """
        if s is None:
            return None
        if not isinstance(s, str):
            raise LateValidationError(
                "{col.name}: datetime-type filter values must be strings containing an ISO8601 formatted date."
            )
        try:
            parsed = datetime.fromisoformat(s).replace(microsecond=0)
        except (ValueError, TypeError) as exc:
            raise LateValidationError(
                "{col.name}: datetime-type filter values must be strings containing an ISO8601 formatted date."
            ) from exc
        if not parsed.tzinfo:
            return parsed
        offset = parsed.tzinfo.utcoffset(parsed)
        if offset == timedelta():  # 0 timedelta is equivalent to UTC
            return parsed.replace(tzinfo=None)
        raise LateValidationError(
            f"{col.name}: datetime-type filter values must be in UTC, or not be tagged with an explicit timezone: {s}"
        )

    parsed_values = list(map(str_to_datetime, filter_.value))
    if filter_.relation == Relation.EXCLUDES:
        return general_excludes_filter(col, parsed_values)

    if filter_.relation == Relation.INCLUDES:
        return sqlalchemy.not_(general_excludes_filter(col, parsed_values))

    # Else it's Relation.BETWEEN:
    match parsed_values:
        case (left, None):
            return col >= left
        case (None, right):
            return col <= right
        case (left, right):
            return col.between(left, right)
    raise RuntimeError("Bug: invalid AudienceSpecFilter.")


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
            return general_excludes_filter(col, filter_.value)
        case Relation.INCLUDES if isinstance(col.type, sqlalchemy.Boolean):
            return or_(*[
                col.is_(value) if value is not None else col.is_(None)
                for value in filter_.value
            ])
        case Relation.INCLUDES:
            return sqlalchemy.not_(general_excludes_filter(col, filter_.value))
    raise RuntimeError("Bug: invalid AudienceSpecFilter.")


def compose_query(sa_table: Table, chosen_n: int, filters):
    return (
        select(sa_table)
        .filter(*filters)
        .order_by(custom_functions.Random(sa_table=sa_table))
        .limit(chosen_n)
    )
