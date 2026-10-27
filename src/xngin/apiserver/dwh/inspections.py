"""Methods for converting SQLAlchemy metadata into our application-specific types."""

import re

import sqlalchemy
from sqlalchemy_bigquery import STRUCT

from xngin.apiserver.common_field_types import VALID_SQL_COLUMN_REGEX
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.admin.admin_api_types import FieldMetadata, InspectDatasourceTableResponse
from xngin.apiserver.routers.common_enums import DataType


def create_inspect_table_response_from_table(
    table: sqlalchemy.Table,
) -> InspectDatasourceTableResponse:
    """Creates an InspectDatasourceTableResponse from a sqlalchemy.Table.

    Columns whose names cannot be used as a FieldName (a Postgres column named "a.b", say) are left
    out, as are columns with unsupported data types.

    Raises:
        LateValidationError: When the table has nested columns, which we do not support.
    """
    nested_columns = sorted(c.name for c in table.columns.values() if _is_nested_column(table, c))
    if nested_columns:
        raise LateValidationError(
            f"Table '{table.name}' has nested columns ({', '.join(nested_columns)}), which are not supported. "
            "Create a view that exposes the fields you need as top-level columns, then select that view."
        )

    usable_columns = [c for c in table.columns.values() if re.match(VALID_SQL_COLUMN_REGEX, c.name)]
    possible_id_columns = {
        c.name
        for c in usable_columns
        if c.name.endswith("id") or c.name.endswith("hash") or isinstance(c.type, sqlalchemy.sql.sqltypes.UUID)
    }
    primary_key_columns = {c.name for c in usable_columns if c.primary_key}
    if len(primary_key_columns) > 1:
        # If there is more than one PK, it probably isn't usable for experiments.
        primary_key_columns = set()
    possible_id_columns |= primary_key_columns

    collected = []
    for column in usable_columns:
        type_hint = column.type
        data_type = DataType.match(type_hint)
        if data_type.is_supported():
            collected.append(
                FieldMetadata(
                    field_name=column.name,
                    data_type=data_type,
                    description=column.comment or "",
                )
            )

    return InspectDatasourceTableResponse(
        primary_key_fields=list(sorted(primary_key_columns)),
        detected_unique_id_fields=list(sorted(possible_id_columns)),
        fields=list(sorted(collected, key=lambda f: f.field_name)),
    )


def _is_nested_column(table: sqlalchemy.Table, column: sqlalchemy.Column) -> bool:
    """Whether column is a sub-field of a BigQuery RECORD column.

    The BigQuery dialect reflects a RECORD column as a STRUCT column followed by one column per
    sub-field, named "parent.child". Other dialects allow dots in ordinary column names, so a dot
    alone does not make a column nested: its parent must be a STRUCT in the same table.
    """
    parent_name, dot, _ = column.name.partition(".")
    if not dot:
        return False
    parent = table.columns.get(parent_name)
    return parent is not None and isinstance(parent.type, STRUCT)
