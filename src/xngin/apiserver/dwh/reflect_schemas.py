"""Methods for converting SQLAlchemy metadata into our application-specific types."""

import sqlalchemy

from xngin.apiserver.routers.admin.admin_api_types import (
    FieldMetadata,
    InspectDatasourceTableResponse,
)
from xngin.apiserver.routers.stateless.stateless_api_types import DataType
from xngin.schema.schema_types import FieldDescriptor, ParticipantsSchema


def create_schema_from_table(table: sqlalchemy.Table, unique_id_col: str | None = None):
    """Attempts to get name and type info from the database Table itself (formerly done via gsheets).

    If unique_id_col is explicitly set to None, we will look for a primary key else assume "id".
    (This mode should only be used if bootstrapping a sheet config from a table's schema.)
    """

    collected = []
    if unique_id_col is None:
        unique_id_col = next(
            (c.name for c in table.columns.values() if c.primary_key), "id"
        )
    for column in table.columns.values():
        type_hint = column.type
        collected.append(
            FieldDescriptor(
                field_name=column.name,
                data_type=DataType.match(type_hint),
                description="",
                is_unique_id=column.name == unique_id_col,
                is_strata=False,
                is_filter=False,
                is_metric=False,
            )
        )
    # Sort order is: unique ID first, then string fields, then the rest by name.
    rows = sorted(
        collected,
        key=lambda r: (
            not r.is_unique_id,
            r.data_type != DataType.CHARACTER_VARYING,
            r.field_name,
        ),
    )
    return ParticipantsSchema(table_name=table.name, fields=rows)


def create_inspect_table_response_from_table(
    table: sqlalchemy.Table,
) -> InspectDatasourceTableResponse:
    """Creates an InspectDatasourceTableResponse from a sqlalchemy.Table.

    This is similar to config_sheet.create_schema_from_table but tailored to use in the API.
    """
    possible_id_columns = {
        c.name
        for c in table.columns.values()
        if c.name.endswith("id") or isinstance(c.type, sqlalchemy.sql.sqltypes.UUID)
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
        detected_unique_id_fields=list(sorted(possible_id_columns)),
        fields=list(sorted(collected, key=lambda f: f.field_name)),
    )
