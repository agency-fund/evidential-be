import gspread
import pandas


def fetch_sheet(url, worksheet) -> gspread.Worksheet:
    """Reads a Google Spreadsheet."""
    gc = gspread.service_account()
    sheet = gc.open_by_url(url)
    return sheet.worksheet(worksheet)


def read_sheet_from_gsheet(url, worksheet) -> list[dict[str, str | int | bool]]:
    """Reads a Google Spreadsheet."""
    return fetch_sheet(url, worksheet).get_all_records()


def read_sheet_df(url: str, worksheet: str) -> pandas.DataFrame:
    """Reads a Google Spreadsheet into a Pandas DataFrame.

    See the docs for worksheet.get_all_records() to understand how the spreadsheet data will be represented.
    """
    worksheet = fetch_sheet(url, worksheet)
    return pandas.DataFrame(worksheet.get_all_records())
