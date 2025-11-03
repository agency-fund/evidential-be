import re
from collections.abc import Sequence
from datetime import date, datetime
from typing import Literal

import sqlalchemy
from sqlalchemy import (
    ColumnElement,
    Date,
    DateTime,
    Float,
    Integer,
    Label,
    Select,
    String,
    Table,
    and_,
    cast,
    distinct,
    func,
    not_,
    or_,
    select,
)
from sqlalchemy.orm import Session

from xngin.apiserver.dwh.analysis_types import MetricValue, ParticipantOutcome
from xngin.apiserver.dwh.inspection_types import FieldDescriptor
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import (
    EXPERIMENT_IDS_SUFFIX,
    DesignSpecMetric,
    DesignSpecMetricRequest,
    Filter,
    FilterValueTypes,
    GetFiltersResponseDiscrete,
    GetFiltersResponseElement,
    GetFiltersResponseNumericOrDate,
    Relation,
)
from xngin.apiserver.routers.common_enums import DataType, FilterClass, MetricType
from xngin.apiserver.routers.experiments.property_filters import str_to_date_or_datetime, validate_filter_value
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
) -> list[GetFiltersResponseElement]:
    """Runs SELECT queries for metrics (min, max, distinct, etc) on filter fields.

    This async method runs the queries against the synchronous Session in a thread.

    Args:
        session: SQLAlchemy session for customer data warehouse
        sa_table: SQLAlchemy Table object
        db_schema: The latest table schema in the database described as FieldDescriptors
        filter_schema: The latest filter schema in the participant type config described as FieldDescriptors

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


def get_participant_metrics(
    session,
    sa_table: Table,
    metrics: list[DesignSpecMetricRequest],
    unique_id_field: str,
    participant_ids: list[str],
) -> list[ParticipantOutcome]:
    missing_metrics = {m.field_name for m in metrics if m.field_name not in sa_table.c}
    if len(missing_metrics) > 0:
        raise LateValidationError(f"Missing metrics (check your Datasource configuration): {missing_metrics}")

    # build our query
    metric_types = [MetricType.from_python_type(sa_table.c[m.field_name].type.python_type) for m in metrics]

    # select participant_id field
    if unique_id_field not in sa_table.columns:
        raise LateValidationError(f"Unique ID field {unique_id_field} not found in table.")
    participant_id_column = sa_table.c[unique_id_field]
    # We always store participant_id as a string, so select it back as such.
    select_columns: list[Label] = [cast(participant_id_column, String).label("participant_id")]

    field_names = ["participant_id"]
    # add metrics from the experiment design
    for metric, metric_type in zip(metrics, metric_types, strict=False):
        field_name = metric.field_name
        field_names.append(field_name)
        col = sa_table.c[field_name]
        # Coerce everything to Float to avoid Decimal/Integer/Boolean issues across backends.
        if metric_type is MetricType.NUMERIC:
            cast_column = cast(col, Float)
        else:  # re: avg(boolean) doesn't work on pg-like backends
            cast_column = cast(cast(col, Integer), Float)
        select_columns.append(cast_column.label(field_name))

    # create a single filter, filtering on the unique_id_field using
    # participant_ids from the treatment assignment.
    participant_id_filter = Filter(
        field_name=unique_id_field,
        relation=Relation.INCLUDES,
        value=[Filter.cast_participant_id(pid, participant_id_column.type) for pid in participant_ids],
    )
    participant_filter = create_one_filter(participant_id_filter, sa_table)
    query = select(*select_columns).filter(participant_filter)
    results = session.execute(query)

    participant_outcomes: list[ParticipantOutcome] = []
    for result in results:
        metric_values: list[MetricValue] = []
        participant_id = None
        for i, field_name in enumerate(field_names):
            if field_name == "participant_id":
                participant_id = result[i]
            else:
                metric_values.append(MetricValue(metric_name=field_name, metric_value=result[i]))
        if participant_id is None:
            # Should never happen as we filter on the participant_id field.
            raise LateValidationError("Participant ID is required.")
        participant_outcomes.append(ParticipantOutcome(participant_id=str(participant_id), metric_values=metric_values))
    return participant_outcomes


def create_one_filter(filter_: Filter, sa_table: sqlalchemy.Table):
    """Converts a Filter into a SQLAlchemy filter."""
    if isinstance(sa_table.columns[filter_.field_name].type, (DateTime, Date)):
        return create_date_or_datetime_filter(sa_table.columns[filter_.field_name], filter_)
    if filter_.field_name.endswith(EXPERIMENT_IDS_SUFFIX):
        return create_special_experiment_id_filter(sa_table.columns[filter_.field_name], filter_)
    return create_filter(sa_table.columns[filter_.field_name], filter_)


def create_query_filters(sa_table: sqlalchemy.Table, filters: list[Filter]):
    """Converts a list of Filter into a list of SQLAlchemy filters."""
    return [create_one_filter(filter_, sa_table) for filter_ in filters]


def create_special_experiment_id_filter(col: sqlalchemy.Column, filter_: Filter) -> ColumnElement:
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
        case Relation.BETWEEN:
            # This should be impossible as it's caught by the Filter validator:
            raise ValueError(f"Experiment id filter on {filter_.field_name} has invalid relation: {filter_.relation}")


def make_csv_regex(values):
    """Constructs a regular expression for matching a CSV string against a list of values.

    The generated regexp is to be used by re.search() or equivalent. We assume that most database engines
    will support identical syntax.
    """
    value_regexp = r"(" + r"|".join(re.escape(str(v).lower()) for v in values if v is not None) + r")"
    return r"(^x$)|(^x,)|(,x$)|(,x,)".replace("x", value_regexp)


def general_excludes_filter(
    col: sqlalchemy.Column, value: FilterValueTypes | Sequence[datetime | None] | Sequence[date | None]
) -> ColumnElement[bool]:
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


def create_date_or_datetime_filter(col: sqlalchemy.Column, filter_: Filter) -> ColumnElement:
    """Converts a single Filter for a DateTime or Date-typed column into a sqlalchemy filter."""
    # First validate that we're working with the right column type.
    if not isinstance(col.type, (DateTime, Date)):
        raise LateValidationError(f"Column {col.name} is not a DateTime or Date type; cannot create filter.")

    target_type: Literal["date", "datetime"] = "date" if isinstance(col.type, Date) else "datetime"
    parsed_values = list(map(lambda s: str_to_date_or_datetime(col.name, s, target_type), filter_.value))

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
        case (left, None, None):
            return or_(col >= left, col.is_(sqlalchemy.null()))
        case (None, right, None):
            return or_(col <= right, col.is_(sqlalchemy.null()))
        case (left, right, None):
            return or_(col.between(left, right), col.is_(sqlalchemy.null()))
        case _:
            raise RuntimeError("Bug: invalid filter.")


def create_filter(col: sqlalchemy.Column, filter_: Filter) -> ColumnElement:
    """Converts a single Filter to a sqlalchemy filter."""
    parsed_values: Sequence = [
        validate_filter_value(filter_.field_name, value, DataType.match(col.type)) for value in filter_.value
    ]
    match filter_.relation:
        case Relation.BETWEEN:
            match parsed_values:
                case (left, None):
                    return col >= left  # type: ignore
                case (None, right):
                    return col <= right  # type: ignore
                case (left, right):
                    return col.between(left, right)
                case (left, None, None):
                    return or_(col >= left, col.is_(sqlalchemy.null()))
                case (None, right, None):
                    return or_(col <= right, col.is_(sqlalchemy.null()))
                case (left, right, None):
                    return or_(col.between(left, right), col.is_(sqlalchemy.null()))
                case _:
                    raise RuntimeError("Bug: invalid filter.")
        case Relation.EXCLUDES if isinstance(col.type, sqlalchemy.Boolean):
            return and_(*[col.is_not(value) if value is not None else col.is_not(None) for value in parsed_values])
        case Relation.EXCLUDES:
            return general_excludes_filter(col, parsed_values)
        case Relation.INCLUDES if isinstance(col.type, sqlalchemy.Boolean):
            return or_(*[col.is_(value) if value is not None else col.is_(None) for value in parsed_values])
        case Relation.INCLUDES:
            return sqlalchemy.not_(general_excludes_filter(col, parsed_values))
        case _:
            raise RuntimeError("Bug: invalid Filter.")


def compose_query(sa_table: Table, select_columns: set[str], filters: list[ColumnElement], chosen_n: int):
    """Builds a query to fetch rows from a list of filters and a set of column names to select."""

    if not select_columns:
        raise ValueError("select_columns must have at least one item.")

    columns = []
    for col in sorted(select_columns):  # sort for stable generated sql
        if col not in sa_table.c:
            raise ValueError(f"Column {col} not found in schema.")
        columns.append(sa_table.c[col])

    return select(*columns).filter(*filters).order_by(custom_functions.Random(sa_table=sa_table)).limit(chosen_n)
