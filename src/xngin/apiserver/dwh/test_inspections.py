import pytest
from sqlalchemy import JSON, BigInteger, Boolean, Column, Integer, MetaData, String, Table
from sqlalchemy_bigquery import STRUCT

from xngin.apiserver.dwh.inspections import create_inspect_table_response_from_table, create_schema_from_table
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.admin.admin_api_types import FieldMetadata, InspectDatasourceTableResponse
from xngin.apiserver.routers.common_enums import DataType


def test_create_schema_from_table_success():
    metadata_obj = MetaData()
    my_table = Table(
        "table_name",
        metadata_obj,
        Column("id", BigInteger),
        Column("name", String),
        Column("primary_id", Integer, primary_key=True),
    )

    worksheet = create_schema_from_table(my_table, "name")
    assert worksheet.get_unique_id_field() == "name"
    assert len(worksheet.fields) == 3
    expected_type = {
        "id": DataType.BIGINT,
        "name": DataType.CHARACTER_VARYING,
        "primary_id": DataType.INTEGER,
    }
    for column in worksheet.fields:
        assert column.data_type == expected_type.get(column.field_name, "BAD_COLUMN"), column.field_name

    worksheet = create_schema_from_table(my_table, None)
    assert worksheet.get_unique_id_field() == "primary_id"

    my_table = Table(
        "table_name_without_pk",
        metadata_obj,
        Column("id", BigInteger),
        Column("name", String),
    )
    worksheet = create_schema_from_table(my_table, None)
    assert worksheet.get_unique_id_field() == "id"


def test_create_schema_from_table_does_not_fail_if_no_unique_id():
    my_table = Table(
        "table_name",
        MetaData(),
        Column("_id", Integer),
        Column("name", String),
    )

    schema = create_schema_from_table(my_table, "id")
    assert schema.get_unique_id_field() is None
    assert len(schema.fields) == 2

    schema = create_schema_from_table(my_table, None)
    assert schema.get_unique_id_field() is None
    assert len(schema.fields) == 2


def test_create_schema_from_table_does_not_raise_if_no_unique_id_and_set_unique_id_is_false():
    my_table = Table(
        "table_name",
        MetaData(),
        Column("_id", Integer),
        Column("name", String),
    )

    schema = create_schema_from_table(my_table, "id", set_unique_id=False)
    assert schema.get_unique_id_field() is None
    assert len(schema.fields) == 2

    schema = create_schema_from_table(my_table, None, set_unique_id=False)
    assert schema.get_unique_id_field() is None
    assert len(schema.fields) == 2


def test_create_inspect_table_response_from_table_success():
    my_table = Table(
        "flat_table",
        MetaData(),
        Column("participant_id", String, primary_key=True),
        Column("is_onboarded", Boolean),
        Column("payload", JSON),
    )

    assert create_inspect_table_response_from_table(my_table) == InspectDatasourceTableResponse(
        primary_key_fields=["participant_id"],
        detected_unique_id_fields=["participant_id"],
        # JSON columns are unsupported and left out.
        fields=[
            FieldMetadata(field_name="is_onboarded", data_type=DataType.BOOLEAN, description=""),
            FieldMetadata(field_name="participant_id", data_type=DataType.CHARACTER_VARYING, description=""),
        ],
    )


def test_create_inspect_table_response_from_table_rejects_nested_columns():
    # Mirrors how the BigQuery dialect reflects a RECORD column: the parent, then one column per sub-field.
    my_table = Table(
        "replicated_table",
        MetaData(),
        Column("_id", String),
        Column("datastream_metadata", STRUCT(uuid=String, source_timestamp=Integer)),
        Column("datastream_metadata.uuid", String),
        Column("datastream_metadata.source_timestamp", Integer),
    )

    with pytest.raises(LateValidationError) as excinfo:
        create_inspect_table_response_from_table(my_table)

    message = str(excinfo.value)
    assert "Table 'replicated_table' has nested columns" in message
    assert "datastream_metadata.source_timestamp, datastream_metadata.uuid" in message
    assert "Create a view" in message
