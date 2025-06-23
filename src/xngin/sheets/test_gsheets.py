from pathlib import Path

import pytest

from xngin.apiserver.routers.common_api_types import DataType
from xngin.schema.schema_types import FieldDescriptor, ParticipantsSchema
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
    fake_worksheet = ParticipantsSchema(
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
