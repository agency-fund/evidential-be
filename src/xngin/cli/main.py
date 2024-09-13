"""Command line tool for various xngin-related operations."""

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Tuple

import gspread
import typer
from gspread import GSpreadException
from rich.console import Console

from xngin.apiserver.api_types import DataType
from xngin.apiserver.settings import (
    get_sqlalchemy_table,
    SqlalchemyAndTable,
    SheetRef,
    XnginSettings,
    CannotFindTheTableException,
)
from xngin.apiserver.testing import testing_dwh
from xngin.sheets.config_sheet import (
    InvalidSheetException,
    fetch_and_parse_sheet,
    ColumnDescriptor,
    create_sheetconfig_from_table,
    ConfigWorksheet,
)

err_console = Console(stderr=True)
app = typer.Typer(help=__doc__)

logging.basicConfig(
    level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def infer_config_from_schema(dsn: str, table: str):
    """Infers a configuration from a SQLAlchemy schema.

    :param dsn The SQLAlchemy-compatible DSN.
    :param table The name of the table to inspect.
    """
    try:
        dwh = get_sqlalchemy_table(
            SqlalchemyAndTable(sqlalchemy_url=dsn, table_name=table)
        )
    except CannotFindTheTableException as cfte:
        err_console.print(cfte.message)
        raise typer.Exit(1) from cfte
    return create_sheetconfig_from_table(dwh)


@app.command()
def bootstrap_testing_dwh(
    src: Path = testing_dwh.TESTING_DWH_RAW_DATA,
    dest: Path = testing_dwh.TESTING_DWH_SQLITE_PATH,
    force: bool = False,
):
    """Bootstraps the local testing data warehouse."""
    testing_dwh.create_dwh_sqlite_database(src, dest, force=force)


@app.command()
def bootstrap_spreadsheet(
    dsn: Annotated[
        str, typer.Argument(..., help="The SQLALchemy DSN of a data warehouse.")
    ],
    table_name: Annotated[
        str,
        typer.Argument(
            ...,
            help="The name of the table to pull field metadata from. If creating a Google Sheet, this will also be "
            "used as the worksheet name, unless overrideen by --worksheet-name.",
        ),
    ],
    create_gsheet: Annotated[
        bool,
        typer.Option(
            ...,
            help="Create a Google Sheet version of the configuration spreadsheet from `table_name`.",
        ),
    ] = False,
    worksheet_name: Annotated[
        str | None,
        typer.Option(
            ...,
            help="When creating the Google Sheet, use the specified value rather than the table name "
            "as the worksheet name.",
        ),
    ] = None,
    share_email: Annotated[
        Tuple[str] | None,
        typer.Option(
            help="Share the newly created Google Sheet with one or more email addresses.",
        ),
    ] = None,
):
    """Generates a Google Spreadsheet from a SQLAlchemy DSN and a table name.

    Use this to get a customer started on configuring an experiment.
    """
    config = infer_config_from_schema(dsn, table_name)

    # Exclude the `extra` field.
    column_names = [c for c in ColumnDescriptor.model_fields if c != "extra"]
    rows = [column_names]

    def convert(v):
        if isinstance(v, bool):
            if v:
                return "true"
            return ""
        if isinstance(v, DataType):
            return str(v)
        return v

    for row in config.columns:
        # Exclude the `extra` field.
        rows.append([
            convert(n) for n in row.model_dump().values() if not isinstance(n, dict)
        ])

    if not create_gsheet:
        writer = csv.writer(sys.stdout)
        writer.writerows(rows)
        return

    gc = gspread.service_account()
    # TODO: if the sheet exists already, add a new worksheet instead of erroring.
    if worksheet_name is None:
        worksheet_name = table_name
    sheet = gc.create(worksheet_name)
    # The "Sheet1" worksheet is created automatically. We don't want to use that, so hold on to its ID for later
    # so that we can delete it.
    sheets_to_delete = [s.id for s in sheet.worksheets()]
    worksheet = sheet.add_worksheet(table_name, rows=len(rows), cols=len(column_names))
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
    for sheet_id in sheets_to_delete:
        sheet.del_worksheet_by_id(sheet_id)
    if share_email:
        for email in share_email:
            # TODO: if running as a service account, also transfer ownership to one fo the email addresses.
            sheet.share(email, perm_type="user", role="writer")
            print(f"# Sheet shared with {email}")
    print(sheet.url)


@app.command()
def parse_config_spreadsheet(
    url: Annotated[
        str,
        typer.Argument(
            ..., help="URL to the Google Sheet, or file://-style path to a CSV."
        ),
    ],
    worksheet: Annotated[
        str,
        typer.Argument(
            ...,
            help="The worksheet to parse. If parsing CSV, specify the name of the table the CSV was generated from.",
        ),
    ],
):
    """Parses a Google Spreadsheet and displays the parsed configuration on the console.

    This is primarily useful for confirming that the spreadsheet passes validations.
    """
    try:
        parsed = fetch_and_parse_sheet(SheetRef(url=url, worksheet=worksheet))
        print(parsed.model_dump_json(indent=2))
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


@app.command()
def export_json_schemas(output: Path = ".schemas"):
    """Generates JSON schemas for Xngin settings files."""
    if not output.exists():
        output.mkdir()
    for model in (XnginSettings, ConfigWorksheet):
        filename = output / (model.__name__ + ".schema.json")
        with open(filename, "w") as outf:
            outf.write(json.dumps(model.model_json_schema(), indent=2, sort_keys=True))
            print(f"Wrote {filename}.")


if __name__ == "__main__":
    app()
