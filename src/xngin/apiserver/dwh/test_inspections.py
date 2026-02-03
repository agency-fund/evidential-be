import pytest
from deepdiff import DeepDiff
from pydantic import ValidationError
from sqlalchemy import BigInteger, Column, Double, Integer, MetaData, String, Table

from xngin.apiserver.dwh.inspection_types import FieldDescriptor
from xngin.apiserver.dwh.inspections import build_proposed_and_drift, create_schema_from_table
from xngin.apiserver.routers.admin.admin_api_types import ColumnDeleted, FieldChangedType
from xngin.apiserver.routers.common_enums import DataType
from xngin.apiserver.settings import ParticipantsDef


def test_create_schema_from_table_success():
    metadata_obj = MetaData()
    my_table = Table(
        "table_name",
        metadata_obj,
        Column("id", BigInteger),
        Column("name", String),
        Column("primary_id", Integer, primary_key=True),
    )

    # Explicit column found
    worksheet = create_schema_from_table(my_table, "name")
    assert worksheet.get_unique_id_field() == "name"
    assert len(worksheet.fields) == 3
    expected_type = {
        "id": DataType.BIGINT,
        "name": DataType.CHARACTER_VARYING,
        "primary_id": DataType.INTEGER,
    }
    for c in worksheet.fields:
        assert c.data_type == expected_type.get(c.field_name, "BAD_COLUMN"), c.field_name

    # PK found
    worksheet = create_schema_from_table(my_table, None)
    assert worksheet.get_unique_id_field() == "primary_id"

    # default id found
    my_table = Table(
        "table_name_without_pk",
        metadata_obj,
        Column("id", BigInteger),
        Column("name", String),
    )
    worksheet = create_schema_from_table(my_table, None)
    assert worksheet.get_unique_id_field() == "id"


def test_create_schema_from_table_fails_if_no_unique_id():
    my_table = Table(
        "table_name",
        MetaData(),
        Column("_id", Integer),
        Column("name", String),
    )

    # Doesn't find the specified id:
    with pytest.raises(ValidationError):
        create_schema_from_table(my_table, "id")

    # Has no primary key or generic "id"
    with pytest.raises(ValidationError):
        create_schema_from_table(my_table, None)


def test_create_schema_from_table_does_not_raise_if_no_unique_id_and_set_unique_id_is_false():
    my_table = Table(
        "table_name",
        MetaData(),
        Column("_id", Integer),
        Column("name", String),
    )

    # Doesn't find the specified id
    schema = create_schema_from_table(my_table, "id", set_unique_id=False)
    assert schema.get_unique_id_field() is None
    assert len(schema.fields) == 2

    # Has no primary key or generic "id"
    schema = create_schema_from_table(my_table, None, set_unique_id=False)
    assert schema.get_unique_id_field() is None
    assert len(schema.fields) == 2


def test_build_proposed_and_drift_no_drift():
    """Verifies the merge of field descriptors occurs as expected."""
    table = Table(
        "test_table",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("name", String),
        Column("outcome", Integer),
    )

    participants = ParticipantsDef(
        participant_type="test_type",
        type="schema",
        table_name="test_table",
        fields=[
            FieldDescriptor(field_name="id", data_type=DataType.INTEGER, is_unique_id=True),
            FieldDescriptor(field_name="name", data_type=DataType.CHARACTER_VARYING),
            FieldDescriptor(field_name="outcome", data_type=DataType.INTEGER, is_metric=True, is_strata=True),
        ],
    )

    proposed, drift = build_proposed_and_drift(participants, table)

    assert len(drift.schema_diff) == 0
    # Verify the merge.
    assert len(proposed.fields) == 3
    diff = DeepDiff(proposed, participants, ignore_order=True)
    assert not diff, f"Objects differ:\n{diff.pretty()}"


def test_build_proposed_and_drift_column_deleted():
    """Old deleted_cols in participants are missing from the current table."""
    table = Table("test_table", MetaData(), Column("id", Integer, primary_key=True))

    participants = ParticipantsDef(
        participant_type="test_type",
        type="schema",
        table_name="test_table",
        fields=[
            FieldDescriptor(field_name="id", data_type=DataType.INTEGER, is_unique_id=True),
            FieldDescriptor(field_name="deleted_col_b", data_type=DataType.INTEGER),
            FieldDescriptor(field_name="deleted_col_a", data_type=DataType.CHARACTER_VARYING),
        ],
    )

    proposed, drift = build_proposed_and_drift(participants, table)

    assert len(drift.schema_diff) == 2
    assert drift.schema_diff == [
        ColumnDeleted(table_name="test_table", column_name="deleted_col_a"),
        ColumnDeleted(table_name="test_table", column_name="deleted_col_b"),
    ]

    # Proposed should reflect the merged table state with deleted column removed
    assert proposed.table_name == "test_table"
    assert proposed.participant_type == "test_type"
    assert len(proposed.fields) == 1
    assert proposed.fields[0] == FieldDescriptor(field_name="id", data_type=DataType.INTEGER, is_unique_id=True)


def test_build_proposed_and_drift_type_changed():
    """Table has an 'age' column as String, but participants expected Integer."""
    table = Table(
        "test_table",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("age", String),
    )

    participants = ParticipantsDef(
        participant_type="test_type",
        type="schema",
        table_name="test_table",
        fields=[
            FieldDescriptor(field_name="id", data_type=DataType.INTEGER, is_unique_id=True),
            FieldDescriptor(field_name="age", data_type=DataType.INTEGER),
        ],
    )

    proposed, drift = build_proposed_and_drift(participants, table)

    assert len(drift.schema_diff) == 1
    diff = drift.schema_diff[0]
    assert diff == FieldChangedType(
        table_name="test_table",
        column_name="age",
        old_type=DataType.INTEGER,
        new_type=DataType.CHARACTER_VARYING,
    )
    # Proposed should have the new type
    assert len(proposed.fields) == 2
    assert proposed.fields == [
        FieldDescriptor(field_name="age", data_type=DataType.CHARACTER_VARYING),
        FieldDescriptor(field_name="id", data_type=DataType.INTEGER, is_unique_id=True),
    ]


def test_build_proposed_and_drift_new_column_added():
    """A 'new_col' is present in the table that participants does not have."""
    table = Table(
        "test_table",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("new_col", Double),
    )

    participants = ParticipantsDef(
        participant_type="test_type",
        type="schema",
        table_name="test_table",
        fields=[
            FieldDescriptor(field_name="id", data_type=DataType.INTEGER, is_unique_id=True),
        ],
    )

    proposed, drift = build_proposed_and_drift(participants, table)

    # New columns are not considered a possible breaking change type of drift worth reporting.
    assert len(drift.schema_diff) == 0
    assert len(proposed.fields) == 2
    assert proposed.fields == [
        FieldDescriptor(field_name="id", data_type=DataType.INTEGER, is_unique_id=True),
        FieldDescriptor(field_name="new_col", data_type=DataType.DOUBLE_PRECISION),
    ]
