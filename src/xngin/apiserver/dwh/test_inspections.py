import pytest
from pydantic import ValidationError
from sqlalchemy import BigInteger, Column, Integer, MetaData, String, Table

from xngin.apiserver import conftest
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.dwh.inspections import (
    create_schema_from_table,
    generate_field_descriptors,
)
from xngin.apiserver.routers.common_api_types import DataType


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


async def test_generate_column_descriptors():
    settings = conftest.get_settings_for_test()
    config = settings.get_datasource("testing").config
    async with DwhSession(config.dwh) as dwh:
        sa_table = await dwh.infer_table("dwh")

    db_schema = generate_field_descriptors(sa_table, "last_name")

    # Check a few columns:
    assert db_schema["gender"].field_name == "gender"
    assert db_schema["gender"].data_type == DataType.CHARACTER_VARYING
    assert db_schema["gender"].description == ""
    assert db_schema["gender"].is_unique_id is False
    assert db_schema["gender"].is_strata is False
    assert db_schema["gender"].is_filter is False
    assert db_schema["gender"].is_metric is False
    assert db_schema["gender"].extra is None  # only necessary info loaded
    assert db_schema["last_name"].field_name == "last_name"
    assert db_schema["last_name"].data_type == DataType.CHARACTER_VARYING
    # Next assertion ust because we labeled it that way in settings!
    assert db_schema["last_name"].is_unique_id
    assert db_schema["current_income"].field_name == "current_income"
    assert db_schema["current_income"].data_type == DataType.NUMERIC
    assert db_schema["current_income"].is_unique_id is False
    assert db_schema["is_recruited"].field_name == "is_recruited"
    assert db_schema["is_recruited"].data_type == DataType.BOOLEAN
    assert db_schema["is_recruited"].is_unique_id is False
