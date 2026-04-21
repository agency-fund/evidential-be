from collections.abc import Sequence
from datetime import date, datetime

import sqlalchemy
from sqlalchemy import ColumnElement, Table, and_, or_, select

from xngin.apiserver.routers.common_api_types import Filter, FilterValueTypes
from xngin.apiserver.routers.common_enums import DataType, Relation
from xngin.apiserver.routers.experiments.property_filters import validate_filter_value
from xngin.db_extensions import custom_functions


def create_one_filter(filter_: Filter, sa_table: sqlalchemy.Table):
    """Converts a Filter into a SQLAlchemy filter."""
    return create_filter(sa_table.columns[filter_.field_name], filter_)


def create_query_filters(sa_table: sqlalchemy.Table, filters: list[Filter]):
    """Converts a list of Filter into a list of SQLAlchemy filters."""
    return [create_one_filter(filter_, sa_table) for filter_ in filters]


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


def general_includes_filter(
    col: sqlalchemy.Column, value: FilterValueTypes | Sequence[datetime | None] | Sequence[date | None]
) -> ColumnElement[bool]:
    if None in value:
        non_null_list = [v for v in value if v is not None]
        if len(non_null_list) == 0:
            return col.is_(sqlalchemy.null())
        return or_(
            col.is_(sqlalchemy.null()),
            col.in_(non_null_list),
        )
    return col.in_(value)


def create_between_filter(col: sqlalchemy.Column, values: Sequence) -> ColumnElement:
    """Helper function to create a BETWEEN SQLAlchemy filter expression.

    Args:
        col: SQLAlchemy column to filter on
        values: Tuple of (left, right) or (left, right, None). None indicates NULL inclusion.
    """
    match values:
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


def create_filter(col: sqlalchemy.Column, filter_: Filter) -> ColumnElement:
    """Converts a single Filter to a sqlalchemy filter."""
    parsed_values: Sequence = [
        validate_filter_value(filter_.field_name, value, DataType.match(col.type)) for value in filter_.value
    ]
    match filter_.relation:
        case Relation.BETWEEN:
            return create_between_filter(col, parsed_values)
        case Relation.EXCLUDES if isinstance(col.type, sqlalchemy.Boolean):
            return and_(*[col.is_not(value) if value is not None else col.is_not(None) for value in parsed_values])
        case Relation.EXCLUDES:
            return general_excludes_filter(col, parsed_values)
        case Relation.INCLUDES if isinstance(col.type, sqlalchemy.Boolean):
            return or_(*[col.is_(value) if value is not None else col.is_(None) for value in parsed_values])
        case Relation.INCLUDES:
            return general_includes_filter(col, parsed_values)
        case _:
            raise RuntimeError("Bug: invalid Filter.")


def compose_query(sa_table: Table, select_columns: set[str], filters: list[ColumnElement], desired_n: int):
    """Builds a query to fetch rows from a list of filters and a set of column names to select."""

    if not select_columns:
        raise ValueError("select_columns must have at least one item.")

    columns = []
    for col in sorted(select_columns):  # sort for stable generated sql
        if col not in sa_table.c:
            raise ValueError(f"Column {col} not found in schema.")
        columns.append(sa_table.c[col])

    return select(*columns).filter(*filters).order_by(custom_functions.Random(sa_table=sa_table)).limit(desired_n)
