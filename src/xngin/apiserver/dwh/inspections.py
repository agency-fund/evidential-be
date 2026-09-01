"""Methods for converting SQLAlchemy metadata into our application-specific types."""

import sqlalchemy

from xngin.apiserver.dwh.inspection_types import FieldDescriptor, ParticipantsSchema
from xngin.apiserver.routers.admin.admin_api_types import FieldMetadata, InspectDatasourceTableResponse
from xngin.apiserver.routers.common_enums import DataType


def create_schema_from_table(table: sqlalchemy.Table, unique_id_col: str | None = None, *, set_unique_id: bool = True):
    """Attempts to get name and type info from the database Table itself.

    If set_unique_id is True, unique_id_col is set to None, we will look for a primary key or assume a column named "id"
    is primary key. If set_unique_id is false, no fields will be marked as the unique id.
    """

    collected = []
    if unique_id_col is None:
        unique_id_col = next((c.name for c in table.columns.values() if c.primary_key), "id")
    for column in table.columns.values():
        type_hint = column.type
        descriptor = FieldDescriptor(
            field_name=column.name,
            data_type=DataType.match(type_hint),
            description="",  # Note: we ignore column.comment
            is_unique_id=column.name == unique_id_col and set_unique_id,
            is_strata=False,
            is_filter=False,
            is_metric=False,
        )
        collected.append(descriptor)
    # Sort order is: unique ID first, then string fields, then the rest by name.
    rows = sorted(
        collected,
        key=lambda r: (
            not r.is_unique_id,
            r.data_type != DataType.CHARACTER_VARYING,
            r.field_name,
        ),
    )
    return ParticipantsSchema.model_validate({"table_name": table.name, "fields": rows})


def create_inspect_table_response_from_table(
    table: sqlalchemy.Table,
) -> InspectDatasourceTableResponse:
    """Creates an InspectDatasourceTableResponse from a sqlalchemy.Table.

    This is similar to config_sheet.create_schema_from_table but tailored to use in the API.
    """
    possible_id_columns = {
        c.name
        for c in table.columns.values()
        if c.name.endswith("id") or c.name.endswith("hash") or isinstance(c.type, sqlalchemy.sql.sqltypes.UUID)
    }
    primary_key_columns = {c.name for c in table.columns.values() if c.primary_key}
    if len(primary_key_columns) > 1:
        # If there is more than one PK, it probably isn't usable for experiments.
        primary_key_columns = set()
    possible_id_columns |= primary_key_columns

    collected = []
    for column in table.columns.values():
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


def generate_field_descriptors(table: sqlalchemy.Table, unique_id_col: str):
    """Fetches a map of column name to schema metadata.

    Uniqueness of the values in the column unique_id_col is assumed, not verified!
    """
    return {c.field_name: c for c in create_schema_from_table(table, unique_id_col).fields}
