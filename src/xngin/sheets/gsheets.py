import os

import gspread
import pandas as pd
from loguru import logger

DEFAULT_GSPREAD_CREDENTIALS = "~/.config/gspread/service_account.json"


class GSheetsPermissionError(Exception):
    """Raised when your provided credentials do not have permission to access a sheet."""


def google_app_credentials_file():
    return os.path.expanduser(
        os.environ.get(
            "GSHEET_GOOGLE_APPLICATION_CREDENTIALS", DEFAULT_GSPREAD_CREDENTIALS
        )
    )


def fetch_sheet(url, worksheet) -> gspread.Worksheet:
    """Reads a Google Spreadsheet."""
    try:
        gc = gspread.service_account(filename=google_app_credentials_file())
        sheet = gc.open_by_url(url)
    except PermissionError as pe:
        if isinstance(pe.__cause__, gspread.exceptions.APIError):
            logger.exception(
                "Credentials in %s do not have permission to access %s",
                google_app_credentials_file(),
                url,
            )
            raise GSheetsPermissionError() from pe
        raise

    return sheet.worksheet(worksheet)


def read_sheet_from_gsheet(url, worksheet):
    """Reads a Google Spreadsheet."""
    return fetch_sheet(url, worksheet).get_all_records()


def read_sheet_df(url: str, worksheet: str) -> pd.DataFrame:
    """Reads a Google Spreadsheet into a Pandas DataFrame.

    See the docs for worksheet.get_all_records() to understand how the spreadsheet data will be represented.
    """
    ws = fetch_sheet(url, worksheet)
    return pd.DataFrame(ws.get_all_records())
