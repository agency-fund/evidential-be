from pathlib import Path

import pytest

from xngin.apiserver.api_types import DataType
from xngin.sheets.config_sheet import ColumnDescriptor, ConfigWorksheet
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
                column_group="g",
                description="d",
                is_unique_id=False,
                is_strata=False,
                is_filter=False,
                is_metric=True,
            ),
            ColumnDescriptor(
                column_name="last_name",
                data_type=DataType.CHARACTER_VARYING,
                column_group="g",
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
