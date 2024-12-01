"""Command line tool for various xngin-related operations."""

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import boto3
import gspread
import pandas as pd
import psycopg2
import sqlalchemy
import typer
from gspread import GSpreadException
from rich.console import Console
from sqlalchemy import create_engine, make_url
from sqlalchemy.exc import NoSuchTableError

from xngin.apiserver import settings
from xngin.apiserver.api_types import DataType
from xngin.apiserver.settings import (
    SqlalchemyAndTable,
    SheetRef,
    XnginSettings,
    CannotFindTableException,
)
from xngin.apiserver.testing import testing_dwh
from xngin.sheets.config_sheet import (
    InvalidSheetException,
    fetch_and_parse_sheet,
    ColumnDescriptor,
    create_sheetconfig_from_table,
    ConfigWorksheet,
)

REDSHIFT_HOSTNAME_SUFFIX = "redshift.amazonaws.com"

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
            SqlalchemyAndTable(
                sqlalchemy_url=sqlalchemy.engine.make_url(dsn), table_name=table
            )
        )
    except CannotFindTableException as cfte:
        err_console.print(cfte.message)
        raise typer.Exit(1) from cfte
    return create_sheetconfig_from_table(dwh)


def get_sqlalchemy_table(sqlat: SqlalchemyAndTable):
    """Connects to a SQLAlchemy DSN and creates a sqlalchemy.Table for introspection."""
    engine = settings.sqlalchemy_connect(sqlat.sqlalchemy_url)
    metadata = sqlalchemy.MetaData()
    try:
        return sqlalchemy.Table(sqlat.table_name, metadata, autoload_with=engine)
    except NoSuchTableError as nste:
        metadata.reflect(engine)
        existing_tables = metadata.tables.keys()
        raise CannotFindTableException(sqlat.table_name, existing_tables) from nste


@app.command()
def bootstrap_testing_dwh(
    src: Path = testing_dwh.TESTING_DWH_RAW_DATA,
    dest: Path = testing_dwh.TESTING_DWH_SQLITE_PATH,
    force: Annotated[
        bool, typer.Option(help="Forcibly recreate the testing database.")
    ] = False,
):
    """Bootstraps the local testing data warehouse."""
    testing_dwh.create_dwh_sqlite_database(src, dest, force=force)


def csv_to_ddl(csv_path: Path, table_name: str) -> str:
    """Helper to transform a CSV with Pandas-inferred schema into a CREATE TABLE statement."""
    df = pd.read_csv(csv_path)
    type_map = {
        "int64": "INTEGER",
        "float64": "DECIMAL",
        "object": "VARCHAR(255)",
        "datetime64[ns]": "TIMESTAMP",
        "bool": "BOOLEAN",
    }
    columns = [
        f'"{col}" {type_map.get(str(dtype), "VARCHAR(255)")}'
        for col, dtype in df.dtypes.items()
    ]
    return f"""CREATE TABLE {table_name} ({",\n    ".join(columns)});"""


@app.command()
def create_testing_dwh(
    password: Annotated[
        str, typer.Option(envvar="PGPASSWORD", help="The database password.")
    ],
    dsn: Annotated[str, typer.Option(help="The SQLAlchemy URL for the database.")],
    src: Annotated[
        Path,
        typer.Option(
            help="Local path to the testing data warehouse CSV. This may be zstd-compressed."
        ),
    ] = testing_dwh.TESTING_DWH_RAW_DATA,
    table_name: Annotated[
        str,
        typer.Option(
            help="Desired name of the data warehouse table. This will be replaced if it already exists."
        ),
    ] = "dwh",
    bucket: Annotated[
        str | None,
        typer.Option(
            help="Name of the temporary S3 bucket that is readable by Redshfit when --iam-role is assumed. Required when connecting to Redshift."
        ),
    ] = None,
    iam_role: Annotated[
        str | None,
        typer.Option(
            help="ARN of an IAM Role for Redshift to assume when reading from the bucket specified by --bucket. Required when connecting to Redshift."
        ),
    ] = None,
):
    """Loads the testing data warehouse CSV into a SQLAlchemy database or Redshift Cluster.

    The schema of the CSV is inferred by Pandas df.read_csv helper method.

    Note: The schemas may be different
    """
    url = make_url(dsn).set(password=password)
    if url.host.endswith(REDSHIFT_HOSTNAME_SUFFIX):
        if not bucket:
            print("--bucket is required when importing into Redshift.")
            raise typer.Exit(2)
        if not iam_role:
            print("--iam-role is required when importing into Redshift.")
            raise typer.Exit(2)
        create_table = csv_to_ddl(src, table_name)
        print(create_table)
        with (
            psycopg2.connect(
                database=url.database,
                host=url.host,
                password=url.password,
                port=url.port,
                user=url.username,
            ) as conn,
            conn.cursor() as cur,
        ):
            print("Dropping...")
            cur.execute(f"DROP TABLE IF EXISTS {table_name} ")
            print("Creating...")
            cur.execute(create_table)
            key = src.name
            print(f"Uploading to s3://{bucket}/{key}...")
            s3 = boto3.client("s3")
            s3.upload_file(src, bucket, f"{key}")
            print("Loading...")
            zstd = "ZSTD" if key.endswith(".zst") else ""
            cur.execute(
                f"""COPY {table_name} FROM 's3://{bucket}/{key}' IAM_ROLE '{iam_role}' FORMAT CSV IGNOREHEADER 1 {zstd};"""
            )
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]
            print(f"Loaded {count} rows into {table_name}.")
    else:
        conn = create_engine(url)
        df = pd.read_csv(src)
        row_count = df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Loaded {row_count} rows into {table_name}.")


@app.command()
def bootstrap_spreadsheet(
    dsn: Annotated[
        str, typer.Argument(..., help="The SQLAlchemy DSN of a data warehouse.")
    ],
    table_name: Annotated[
        str,
        typer.Argument(
            ...,
            help="The name of the table to pull field metadata from. If creating a Google Sheet, this will also be "
            "used as the worksheet name, unless overridden by --worksheet-name.",
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
        tuple[str] | None,
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
