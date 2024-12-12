from pathlib import Path

from pydantic import ValidationError
import pytest

from sqlalchemy import BigInteger, Column, Integer, MetaData, String, Table
from xngin.apiserver.api_types import DataType
from xngin.sheets.config_sheet import (
    ColumnDescriptor,
    ConfigWorksheet,
    create_sheetconfig_from_table,
)
from xngin.sheets.gsheets import read_sheet_df


@pytest.mark.integration
def test_read_sheet():
    assert (Path.home() / Path(".config/gspread/service_account.json")).exists()
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
    fake_worksheet = ConfigWorksheet(
        table_name="table_name",
        columns=[
            ColumnDescriptor(
                column_name="first_name",
                data_type=DataType.CHARACTER_VARYING,
                description="d",
                is_unique_id=False,
                is_strata=False,
                is_filter=False,
                is_metric=True,
                extra={"column_group": "g"},
            ),
            ColumnDescriptor(
                column_name="last_name",
                data_type=DataType.CHARACTER_VARYING,
                description="d",
                is_unique_id=True,
                is_strata=False,
                is_filter=False,
                is_metric=True,
            ),
        ],
    )
    assert fake_worksheet.get_unique_id_col() == "last_name"

    fake_worksheet.columns[1].is_unique_id = False
    assert fake_worksheet.get_unique_id_col() is None


def test_create_sheetconfig_from_table_success():
    metadata_obj = MetaData()
    my_table = Table(
        "table_name",
        metadata_obj,
        Column("id", BigInteger),
        Column("name", String),
        Column("primary_id", Integer, primary_key=True),
    )

    # Explicit column found
    worksheet = create_sheetconfig_from_table(my_table, "name")
    assert worksheet.get_unique_id_col() == "name"
    assert len(worksheet.columns) == 3
    expected_type = {
        "id": DataType.BIGINT,
        "name": DataType.CHARACTER_VARYING,
        "primary_id": DataType.INTEGER,
    }
    for c in worksheet.columns:
        assert c.data_type == expected_type.get(c.column_name, "BAD_COLUMN"), (
            c.column_name
        )

    # PK found
    worksheet = create_sheetconfig_from_table(my_table, None)
    assert worksheet.get_unique_id_col() == "primary_id"

    # default id found
    my_table = Table(
        "table_name_without_pk",
        metadata_obj,
        Column("id", BigInteger),
        Column("name", String),
    )
    worksheet = create_sheetconfig_from_table(my_table, None)
    assert worksheet.get_unique_id_col() == "id"


def test_create_sheetconfig_from_table_fails_if_no_unique_id():
    my_table = Table(
        "table_name",
        MetaData(),
        Column("_id", Integer),
        Column("name", String),
    )

    # Doesn't find the specified id
    with pytest.raises(ValidationError):
        create_sheetconfig_from_table(my_table, "id")

    # Has no primary key or generic "id"
    with pytest.raises(ValidationError):
        create_sheetconfig_from_table(my_table, None)
