import unittest
from pathlib import Path

from xngin.sheets.gsheets import read_sheet_df


class ReadSheetTest(unittest.TestCase):
    def test_read_sheet(self):
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
