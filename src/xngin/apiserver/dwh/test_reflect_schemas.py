import pytest
from pydantic import ValidationError
from sqlalchemy import BigInteger, Column, Integer, MetaData, String, Table

from xngin.apiserver.dwh.reflect_schemas import create_schema_from_table
from xngin.apiserver.routers.stateless.stateless_api_types import DataType


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
        assert c.data_type == expected_type.get(c.field_name, "BAD_COLUMN"), (
            c.field_name
        )

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

    # Doesn't find the specified id
    with pytest.raises(ValidationError):
        create_schema_from_table(my_table, "id")

    # Has no primary key or generic "id"
    with pytest.raises(ValidationError):
        create_schema_from_table(my_table, None)
