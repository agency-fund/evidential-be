"""Methods for converting SQLAlchemy metadata into our application-specific types."""

import sqlalchemy

from xngin.apiserver.dwh.inspection_types import FieldDescriptor, ParticipantsSchema
from xngin.apiserver.routers.admin.admin_api_types import (
    ColumnDeleted,
    Drift,
    FieldChangedType,
    FieldMetadata,
    InspectDatasourceTableResponse,
    TableDiff,
)
from xngin.apiserver.routers.common_enums import DataType
from xngin.apiserver.settings import ParticipantsDef


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
    return ParticipantsSchema.model_validate(
        {"table_name": table.name, "fields": rows}, context={"skip_unique_id_check": not set_unique_id}
    )


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


def dehydrate_participants(participants: ParticipantsDef, tables: sqlalchemy.Table) -> ParticipantsDef:
    """Removes fields from ParticipantsDef that are not a filter, metric, strata, unique ID, described, or annotated."""
    participants.fields = [
        f for f in participants.fields if f.description or f.is_filter or f.is_metric or f.is_strata or f.is_unique_id
    ]
    return participants


def _sort_by_fields(fields: list[FieldDescriptor]) -> list[FieldDescriptor]:
    """Sort fields so that unique ID field(s) come first, then by field_name."""
    return sorted(fields, key=lambda f: (not f.is_unique_id, f.field_name))


def rehydrate_participants(participants: ParticipantsDef, table: sqlalchemy.Table) -> ParticipantsDef:
    """Adds fields from table that are not already in participants.fields."""
    dehydrated = {p.field_name: p for p in participants.fields}
    full_schema = create_schema_from_table(table, set_unique_id=False)

    def maybe_merge(latest: FieldDescriptor, edited: FieldDescriptor | None) -> FieldDescriptor:
        if edited is None:  # column is not defined in participant type
            return latest
        fd = latest.model_copy()
        fd.description = edited.description
        fd.is_filter = edited.is_filter
        fd.is_metric = edited.is_metric
        fd.is_strata = edited.is_strata
        fd.is_unique_id = edited.is_unique_id
        return fd

    new_ptype = participants.model_copy()
    # Ensure a deterministic order of fields.
    new_ptype.fields = [maybe_merge(f, dehydrated.get(f.field_name)) for f in _sort_by_fields(full_schema.fields)]
    ParticipantsDef.model_validate(new_ptype)
    return new_ptype


def build_proposed_and_drift(participants: ParticipantsDef, table: sqlalchemy.Table) -> tuple[ParticipantsDef, Drift]:
    schema_diff: list[TableDiff] = []
    schema = create_schema_from_table(table, set_unique_id=False)
    schema_fields = {f.field_name: f for f in schema.fields}

    # Check for changes in a ParticipantsDef at the column level against a full table schema.
    for field in _sort_by_fields(participants.fields):
        if field.field_name not in schema_fields:
            schema_diff.append(ColumnDeleted(table_name=participants.table_name, column_name=field.field_name))
        else:
            # Check for type mismatch
            live_field = schema_fields[field.field_name]
            if field.data_type != live_field.data_type:
                schema_diff.append(
                    FieldChangedType(
                        table_name=participants.table_name,
                        column_name=field.field_name,
                        old_type=field.data_type,
                        new_type=live_field.data_type,
                    )
                )

    proposed = rehydrate_participants(participants, table)
    return proposed, Drift(schema_diff=schema_diff)
