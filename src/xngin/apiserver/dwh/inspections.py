"""Methods for converting SQLAlchemy metadata into our application-specific types."""

import sqlalchemy
from sqlalchemy_bigquery import STRUCT

from xngin.apiserver.common_field_types import has_control_characters
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.admin.admin_api_types import FieldMetadata, InspectDatasourceTableResponse
from xngin.apiserver.routers.common_enums import DataType


def create_inspect_table_response_from_table(
    table: sqlalchemy.Table,
) -> InspectDatasourceTableResponse:
    """Creates an InspectDatasourceTableResponse from a sqlalchemy.Table.

    The response lists each column whose data type is supported, under the name the warehouse reports, including
    names that need quoting in SQL, such as a Postgres column named "a.b". Names with control characters, which
    Postgres and Redshift allow, are left out, since they cannot be field names.

    Raises:
        LateValidationError: When the table has nested columns, which we do not support.
    """
    nested_columns = sorted(c.name for c in table.columns.values() if _is_nested_column(table, c))
    if nested_columns:
        raise LateValidationError(
            f"Table '{table.name}' has nested columns ({', '.join(nested_columns)}), which are not supported. "
            "Create a view that exposes the fields you need as top-level columns, then select that view."
        )

    usable_columns = [c for c in table.columns.values() if not has_control_characters(c.name)]
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
