"""Command line tool for various xngin-related operations."""

import csv
import sys
from typing import List

import gspread
import sqlalchemy
import typer
from gspread import GSpreadException
from rich.console import Console

from xngin.apiserver.api_types import DataType
from xngin.sheets.config_sheet import (
    InvalidSheetException,
    fetch_and_parse_sheet,
    RowConfig,
    SheetConfig,
)

err_console = Console(stderr=True)
app = typer.Typer(help=__doc__)


def infer_config_from_schema(dsn: str, table: str):
    """Infers a configuration from a SQLAlchemy schema.

    :param dsn The SQLAlchemy-compatible DSN.
    :param table The name of the table to inspect.
    """
    engine = sqlalchemy.create_engine(dsn)
    metadata = sqlalchemy.MetaData()
    dwh = sqlalchemy.Table(table, metadata, autoload_with=engine)
    collected = []
    for column in dwh.columns.values():
        collected.append(
            RowConfig(
                table=table,
                column_name=column.name,
                data_type=DataType.match(column.config.python_type),
                column_group="",
                description="",
                is_strata=False,
                is_filter=False,
                is_metric=False,
            )
        )
    return SheetConfig(rows=collected)


@app.command()
def bootstrap_spreadsheet(
    dsn: str,
    table: str,
    gsheet_name: str | None = None,
    share_email: List[str] | None = None,
):
    """Generates a Google Spreadsheet from a SQLAlchemy DSN and a table name.

    If --gsheet-name is provided, it will be used as the name of a Google Spreadsheet and the URL of the sheet will be
    printed on stdout. Pass one or more email addresses with --share-email to share the spreadsheet with others.

    If --gsheet-name is not provided, the config spreadsheet will be written to the terminal in CSV format.
    """
    config = infer_config_from_schema(dsn, table)

    # Exclude the `extra` field.
    column_names = [c for c in RowConfig.model_fields if c != "extra"]
    rows = [column_names]

    def convert(v):
        if isinstance(v, DataType):
            return str(v)
        return v

    for row in config.rows:
        # Exclude the `extra` field.
        rows.append([
            convert(n) for n in row.model_dump().values() if not isinstance(n, dict)
        ])

    if gsheet_name:
        gc = gspread.service_account()
        sheet = gc.create(gsheet_name)
        worksheet = sheet.worksheet("Sheet1")
        worksheet.append_rows(rows)
        # Bold the first row.
        formats = [
            {
                "range": "1:1",
                "format": {
                    "textFormat": {
                        "bold": True,
                    },
                },
            },
        ]
        worksheet.batch_format(formats)
        if share_email:
            for email in share_email:
                # TODO: if running as a service account, also transfer ownership to one fo the email addresses.
                sheet.share(email, perm_type="user", role="writer")
                print(f"# Sheet shared with {email}")
        print(sheet.url)
    else:
        writer = csv.writer(sys.stdout)
        writer.writerows(rows)


@app.command()
def parse_config_spreadsheet(
    url: str, worksheet: str = "Sheet1", write: str | None = None
):
    """Parses a Google Spreadsheet and displays it on the console or writes it to a file."""
    try:
        parsed = fetch_and_parse_sheet(url, worksheet)
        as_json = parsed.model_dump_json(indent=2)
        if write:
            with open(write, "w") as f:
                f.write(as_json)
        else:
            print(as_json)
    except GSpreadException as gse:
        err_console.print(gse)
        raise typer.Exit(1) from gse
    except PermissionError as pe:
        if isinstance(pe.__cause__, gspread.exceptions.APIError):
            err_console.print("You do not have permission to open this spreadsheet.")
            raise typer.Exit(1) from pe
        raise
    except InvalidSheetException as ise:
        err_console.print(f"Error(s):\n{ise}")
        raise typer.Exit(1) from ise


if __name__ == "__main__":
    app()
