from pathlib import Path

from pydantic import ValidationError
import pytest

from sqlalchemy import BigInteger, Column, Integer, MetaData, String, Table
from xngin.apiserver.api_types import DataType
from xngin.sheets.config_sheet import (
    create_configworksheet_from_table,
)
from xngin.schema.sheet_types import FieldDescriptor, ParticipantSchema
from xngin.sheets.gsheets import google_app_credentials_file, read_sheet_df


@pytest.mark.integration
def test_read_sheet():
    assert Path(google_app_credentials_file()).exists()
    # The testing service account has been granted read access to this spreadsheet.
    test_sheet_url = "https://docs.google.com/spreadsheets/d/redacted/edit?usp=sharing"

    sheet = read_sheet_df(test_sheet_url, "Sheet1")
    assert sheet["col1"][0] == "r1c1"
    assert sheet["col2"][1] == "r2c2", sheet
    assert sheet["col2"][2] == "", sheet
    assert sheet["col3"][2] == "r3c3"
    assert sheet["col2"][3] == 0
    assert sheet["col3"][3] == 34.5


def test_config_worksheet_get_unique_id_col():
    fake_worksheet = ParticipantSchema(
        table_name="table_name",
        fields=[
            FieldDescriptor(
                field_name="first_name",
                data_type=DataType.CHARACTER_VARYING,
                description="d",
                is_unique_id=False,
                is_strata=False,
                is_filter=False,
                is_metric=True,
                extra={"column_group": "g"},
            ),
            FieldDescriptor(
                field_name="last_name",
                data_type=DataType.CHARACTER_VARYING,
                description="d",
                is_unique_id=True,
                is_strata=False,
                is_filter=False,
                is_metric=True,
            ),
        ],
    )
    assert fake_worksheet.get_unique_id_field() == "last_name"

    fake_worksheet.fields[1].is_unique_id = False
    assert fake_worksheet.get_unique_id_field() is None


def test_create_configworksheet_from_table_success():
    metadata_obj = MetaData()
    my_table = Table(
        "table_name",
        metadata_obj,
        Column("id", BigInteger),
        Column("name", String),
        Column("primary_id", Integer, primary_key=True),
    )

    # Explicit column found
    worksheet = create_configworksheet_from_table(my_table, "name")
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
    worksheet = create_configworksheet_from_table(my_table, None)
    assert worksheet.get_unique_id_field() == "primary_id"

    # default id found
    my_table = Table(
        "table_name_without_pk",
        metadata_obj,
        Column("id", BigInteger),
        Column("name", String),
    )
    worksheet = create_configworksheet_from_table(my_table, None)
    assert worksheet.get_unique_id_field() == "id"


def test_create_configworksheet_from_table_fails_if_no_unique_id():
    my_table = Table(
        "table_name",
        MetaData(),
        Column("_id", Integer),
        Column("name", String),
    )

    # Doesn't find the specified id
    with pytest.raises(ValidationError):
        create_configworksheet_from_table(my_table, "id")

    # Has no primary key or generic "id"
    with pytest.raises(ValidationError):
        create_configworksheet_from_table(my_table, None)
