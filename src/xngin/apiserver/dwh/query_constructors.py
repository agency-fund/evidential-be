import re
from collections.abc import Sequence
from datetime import date, datetime

import sqlalchemy
from sqlalchemy import ColumnElement, Table, and_, func, literal_column, not_, or_, select
from sqlalchemy import table as sa_table

from xngin.apiserver.routers.common_api_types import EXPERIMENT_IDS_SUFFIX, Filter, FilterValueTypes
from xngin.apiserver.routers.common_enums import DataType, Relation
from xngin.apiserver.routers.experiments.property_filters import validate_filter_value
from xngin.db_extensions import custom_functions


def create_one_filter(filter_: Filter, sa_table: sqlalchemy.Table):
    """Converts a Filter into a SQLAlchemy filter."""
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


def create_inspect_table_from_cursor_query(table_name: str) -> sqlalchemy.Select:
    """Returns a zero-row SELECT used to read column metadata via cursor.description.

    We use SQLAlchemy's identifier quoting to prevent table-name-based SQL injection
    (e.g. "foo; DROP TABLE bar" becomes a quoted identifier rather than executable SQL).
    """
    return select(literal_column("*")).select_from(sa_table(table_name)).limit(0)


def build_search_path_sql(preparer: sqlalchemy.sql.compiler.IdentifierPreparer, schemas: list[str]) -> str:
    """Builds a SET SESSION search_path=... statement with quoted identifiers.

    Uses the provided SQLA engine dialect's IdentifierPreparer to handle double-quote escaping and
    identifier length limits.

    Schema names are also expected to have been validated against SEARCH_PATH_PATTERN (enforced by
    models in settings.py and admin_api_types.py).
    """
    return f"SET SESSION search_path={', '.join(preparer.quote_identifier(s) for s in schemas)}"


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
