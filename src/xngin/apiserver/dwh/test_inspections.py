from sqlalchemy import BigInteger, Column, Integer, MetaData, String, Table

from xngin.apiserver.dwh.inspections import create_schema_from_table
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
