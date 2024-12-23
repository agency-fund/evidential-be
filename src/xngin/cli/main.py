"""Command line tool for various xngin-related operations."""

from google.cloud import bigquery
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import boto3
import gspread
import pandas as pd
import pandas_gbq
import psycopg2
import sqlalchemy
import typer
import zstandard
from gspread import GSpreadException
from gspread.worksheet import CellFormat
from pandas import DataFrame
from pydantic import ValidationError
from pydantic_core import from_json
from rich.console import Console
from sqlalchemy import create_engine, make_url
from sqlalchemy.sql.compiler import IdentifierPreparer

from xngin.apiserver import settings
from xngin.apiserver.api_types import DataType
from xngin.apiserver.settings import (
    SheetRef,
    XnginSettings,
    CannotFindTableError,
    ClientConfig,
)
from xngin.apiserver.testing import testing_dwh
from xngin.sheets.config_sheet import (
    InvalidSheetError,
    fetch_and_parse_sheet,
    ColumnDescriptor,
    create_configworksheet_from_table,
    ConfigWorksheet,
)
import sqlalchemy.dialects.postgresql.psycopg2 as psycopg2sa

REDSHIFT_HOSTNAME_SUFFIX = "redshift.amazonaws.com"

err_console = Console(stderr=True)
app = typer.Typer(help=__doc__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)


def infer_config_from_schema(
    dsn: str, table: str, use_reflection: bool, unique_id_col: str | None = None
):
    """Infers a configuration from a SQLAlchemy schema.

    :param dsn The SQLAlchemy-compatible DSN.
    :param table The name of the table to inspect.
    :param use_reflection True if you want to use SQLAlchemy's reflection, else infer from cursor.
    :param unique_id_col The column name in the table to use as a participant's unique identifier.
    """
    try:
        dwh = settings.infer_table(
            sqlalchemy.create_engine(sqlalchemy.engine.make_url(dsn)),
            table,
            use_reflection=use_reflection,
        )
    except CannotFindTableError as cfte:
        err_console.print(cfte.message)
        raise typer.Exit(1) from cfte
    return create_configworksheet_from_table(dwh, unique_id_col=unique_id_col)


def csv_to_ddl(
    csv_path: Path,
    *,
    table_name: str,
    quoter: IdentifierPreparer,
) -> str:
    """Helper to transform a CSV with Pandas-inferred schema into a CREATE TABLE statement."""
    df = pd.read_csv(csv_path)
    return df_to_ddl(df, table_name=table_name, quoter=quoter)


def df_to_ddl(
    df: DataFrame,
    *,
    table_name: str,
    quoter: IdentifierPreparer,
):
    """Helper to transform a DataFrame into a CREATE TABLE statement."""
    # TODO: replace these types with more generic/widely supported defaults.
    # TODO: warn/fail if Pandas type does not map onto a known SQL type
    default_sql_type = "VARCHAR(255)"
    type_map = {
        "int64": "INTEGER",
        "float64": "DECIMAL",
        "object": "VARCHAR(255)",
        "datetime64[ns]": "TIMESTAMP",
        "bool": "BOOLEAN",
    }
    columns = [
        f"{quoter.quote(col)} {type_map.get(str(dtype), default_sql_type)}"
        for col, dtype in df.dtypes.items()
    ]
    return f"""CREATE TABLE {table_name} ({",\n    ".join(columns)});"""


@app.command()
def create_testing_dwh(
    dsn: Annotated[str, typer.Option(help="The SQLAlchemy URL for the database.")],
    src: Annotated[
        Path,
        typer.Option(
            help="Local path to the testing data warehouse CSV. This may be zstd-compressed."
        ),
    ] = testing_dwh.TESTING_DWH_RAW_DATA,
    nrows: Annotated[
        int | None,
        typer.Option(
            help="Limit to the number of rows to load from the CSV. Does not apply to Redshift or postgresql+psycopg."
        ),
    ] = None,
    schema_name: Annotated[
        str | None,
        typer.Option(
            help="Desired schema to use with the table, else uses your warehouse's default schema. Only applies to "
            "Postgres-like databases."
        ),
    ] = None,
    table_name: Annotated[
        str,
        typer.Option(
            envvar="XNGIN_CLI_TABLE_NAME",
            help="Desired name of the data warehouse table. This will be replaced if it already exists.",
        ),
    ] = "dwh",
    bucket: Annotated[
        str | None,
        typer.Option(
            help="Name of the temporary S3 bucket that is readable by Redshift when --iam-role is assumed. Required "
            "when connecting to Redshift."
        ),
    ] = None,
    iam_role: Annotated[
        str | None,
        typer.Option(
            help="ARN of an IAM Role for Redshift to assume when reading from the bucket specified by --bucket. "
            "Required when connecting to Redshift."
        ),
    ] = None,
    password: Annotated[
        str | None, typer.Option(envvar="PGPASSWORD", help="The database password.")
    ] = None,
):
    """Loads the testing data warehouse CSV into a database.

    Any existing table will be replaced.

    On Redshift Clusters: CSV is parsed by Redshift's native CSV parser. Table DDL is derived from Pandas read_csv.

    On postgres+psycopg connections: CSV is parsed with Postgres' CSV parser. Table DDL is derived from Pandas read_csv.

    On BigQuery: CSV is parsed by Pandas. Table DDL is derived by pandas-gbq and written via to_gbq().

    On all other databases: CSV is parsed by Pandas. Table DDL is derived from Pandas read_csv and written via
    SQLAlchemy and Pandas to_sql().

    Due to variations in all of the above, the loaded data may vary in small ways when loaded with different data
    stores. E.g. floats may not roundtrip.
    """
    create_schema_ddl = (
        f"CREATE SCHEMA IF NOT EXISTS {schema_name}" if schema_name else ""
    )
    full_table_name = f"{schema_name}.{table_name}" if schema_name else table_name
    drop_table_ddl = f"DROP TABLE IF EXISTS {full_table_name}"
    is_compressed = src.suffix == ".zst"

    url = make_url(dsn)
    if password is not None and url.username is not None:
        url = url.set(password=password)

    def read_csv():
        return pd.read_csv(src, nrows=nrows)

    def drop_and_create(cur, create_table_ddl: str):
        cur.execute(drop_table_ddl)
        if schema_name is not None:
            cur.execute(create_schema_ddl)
        print(f"Creating table:\n{create_table_ddl}")
        cur.execute(create_table_ddl)

    def count(cur):
        cur.execute(f"SELECT COUNT(*) FROM {full_table_name}")
        ct = cur.fetchone()[0]
        print(f"Loaded {ct} rows into {full_table_name}.")

    if url.host and url.host.endswith(REDSHIFT_HOSTNAME_SUFFIX):
        if not bucket:
            print("--bucket is required when importing into Redshift.")
            raise typer.Exit(2)
        if not iam_role:
            print("--iam-role is required when importing into Redshift.")
            raise typer.Exit(2)
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
            drop_and_create(
                cur,
                csv_to_ddl(
                    src,
                    table_name=full_table_name,
                    quoter=psycopg2sa.dialect.identifier_preparer,
                ),
            )
            key = src.name
            print(f"Uploading to s3://{bucket}/{key}...")
            s3 = boto3.client("s3")
            s3.upload_file(src, bucket, f"{key}")
            try:
                print("Loading...")
                zstd = "ZSTD" if is_compressed else ""
                cur.execute(
                    f"COPY {full_table_name} FROM 's3://{bucket}/{key}' "
                    f"IAM_ROLE '{iam_role}' FORMAT CSV IGNOREHEADER 1 {zstd};"
                )
                count(cur)
            finally:
                print("Deleting temporary file...")
                s3.delete_object(Bucket=bucket, Key=key)
    elif url.drivername == "postgresql+psycopg":
        df = read_csv()
        engine = create_engine(url)
        with engine.connect() as conn, conn.begin():
            cursor = conn.connection.cursor()
            drop_and_create(
                cursor,
                df_to_ddl(
                    df,
                    table_name=full_table_name,
                    quoter=engine.dialect.identifier_preparer,
                ),
            )
            opener = (lambda x: zstandard.open(x, "r")) if is_compressed else open
            print("Loading...")
            with opener(src) as reader:
                cols = [h.strip() for h in reader.readline().split(",")]
                sql = f"COPY {full_table_name} ({', '.join(cols)}) FROM STDIN (FORMAT CSV, DELIMITER ',')"
                with cursor.copy(sql) as copy:
                    while data := reader.read(1 << 20):
                        copy.write(data)
            count(cursor)
    elif url.drivername == "bigquery":
        df = read_csv()
        destination_table = f"{url.database}.{table_name}"
        print("Loading...")
        pandas_gbq.to_gbq(
            df, destination_table, project_id=url.host, if_exists="replace"
        )
    else:
        df = read_csv()
        engine = create_engine(url)
        with engine.connect() as conn, conn.begin():
            cursor = conn.connection.cursor()
            drop_and_create(
                cursor,
                df_to_ddl(
                    df,
                    table_name=full_table_name,
                    quoter=engine.dialect.identifier_preparer,
                ),
            )
            print("Loading...")
            df.to_sql(
                table_name,
                conn,
                schema=schema_name,
                if_exists="append",
                index=False,
            )
            count(cursor)


@app.command()
def bootstrap_spreadsheet(
    dsn: Annotated[
        str, typer.Argument(..., help="The SQLAlchemy DSN of a data warehouse.")
    ],
    table_name: Annotated[
        str,
        typer.Argument(
            ...,
            envvar="XNGIN_CLI_TABLE_NAME",
            help="The name of the table to pull field metadata from. If creating a Google Sheet, this will also be "
            "used as the worksheet name, unless overridden by --participant-type.",
        ),
    ],
    unique_id_col: Annotated[
        str | None,
        typer.Option(
            help="Specify the column name within table_name to use as the unique identifier for each participant. If "
            "None, will attempt to infer a reasonable column from the schema or raise an error."
        ),
    ] = None,
    create_gsheet: Annotated[
        bool,
        typer.Option(
            ...,
            help="Create a Google Sheet version of the configuration spreadsheet from `table_name`.",
        ),
    ] = False,
    participant_type: Annotated[
        str | None,
        typer.Option(
            ...,
            help="When creating the Google Sheet, use the specified value rather than the table name "
            "as the worksheet name. This corresponds to the participant_type field in the settings file.",
        ),
    ] = None,
    share_email: Annotated[
        tuple[str] | None,
        typer.Option(
            help="Share the newly created Google Sheet with one or more email addresses.",
        ),
    ] = None,
    use_reflection: Annotated[
        bool,
        typer.Option(
            help="True to use SQLAlchemy's table reflection, else use a cursor to infer types",
        ),
    ] = True,
):
    """Generates a Google Spreadsheet from a SQLAlchemy DSN and a table name.

    Use this to get a customer started on configuring an experiment.
    """
    config = infer_config_from_schema(dsn, table_name, use_reflection, unique_id_col)

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
        rows.append([convert(v) for k, v in row.model_dump().items() if k != "extra"])

    if not create_gsheet:
        writer = csv.writer(sys.stdout)
        writer.writerows(rows)
        return

    gc = gspread.service_account()
    # TODO: if the sheet exists already, add a new worksheet instead of erroring.
    if participant_type is None:
        participant_type = table_name
    sheet = gc.create(participant_type)
    # The "Sheet1" worksheet is created automatically. We don't want to use that, so hold on to its ID for later
    # so that we can delete it.
    sheets_to_delete = [s.id for s in sheet.worksheets()]
    worksheet = sheet.add_worksheet(table_name, rows=len(rows), cols=len(column_names))
    worksheet.append_rows(rows)
    # Bold the first row.
    formats: list[CellFormat] = [
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
    except InvalidSheetError as ise:
        err_console.print(f"Error(s):\n{ise}")
        raise typer.Exit(1) from ise


@app.command()
def export_json_schemas(output: Path = Path(".schemas")):
    """Generates JSON schemas for Xngin settings files."""
    if not output.exists():
        output.mkdir()
    for model in (XnginSettings, ConfigWorksheet, ClientConfig):
        filename = output / (model.__name__ + ".schema.json")
        with open(filename, "w") as outf:
            outf.write(json.dumps(model.model_json_schema(), indent=2, sort_keys=True))
            print(f"Wrote {filename}.")


@app.command()
def validate_settings(file: Path):
    """Validates a settings .json file against the Pydantic models."""

    with open(file) as f:
        config = f.read()
    try:
        XnginSettings.model_validate(from_json(config))
    except ValidationError as verr:
        print(f"{file} failed validation:", file=sys.stderr)
        print(verr, file=sys.stderr)
        raise typer.Exit(1) from verr


@app.command()
def bigquery_dataset_set_default_expiration(
    project_id: Annotated[
        str,
        typer.Option(..., help="The Google Cloud Project ID containing the dataset."),
    ],
    dataset_id: Annotated[str, typer.Option(..., help="The dataset name.")],
    days: Annotated[
        int,
        typer.Option(
            ...,
            help="The default expiration for new tables in the dataset (in days).",
            min=0,
        ),
    ] = 1,
):
    """Sets the default TTL (in days) of tables created in a dataset.

    Does not apply to existing tables. To remove the expiration time, specify --days 0.

    This is useful in testing environments that create BigQuery tables that are of minimal use when testing completes.
    """
    new_expiration_ms = days * 24 * 60 * 60 * 1000
    client = bigquery.Client()
    dataset = client.get_dataset(f"{project_id}.{dataset_id}")
    dataset.default_table_expiration_ms = new_expiration_ms
    dataset = client.update_dataset(dataset, ["default_table_expiration_ms"])
    print(
        f"Updated dataset {dataset.project}.{dataset.dataset_id} with new default table "
        f"expiration {dataset.default_table_expiration_ms}"
    )


if __name__ == "__main__":
    app()
