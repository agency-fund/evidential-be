"""Command line tool for various xngin-related operations."""

import base64
import csv
import json
import logging
import re
import sys
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import boto3
import gspread
import pandas as pd
import pandas_gbq
import psycopg
import psycopg2
import sqlalchemy
import tink
import tink.aead
import typer
import zstandard
from email_validator import EmailNotValidError, validate_email
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from gspread import GSpreadException
from gspread.worksheet import CellFormat
from pandas import DataFrame
from pydantic import ValidationError
from pydantic_core import from_json
from rich.console import Console
from sqlalchemy import create_engine, make_url
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.sql.compiler import IdentifierPreparer
from tink import secret_key_access

from xngin.apiserver import settings
from xngin.apiserver.dwh.reflect_schemas import create_schema_from_table
from xngin.apiserver.routers.stateless_api_types import DataType
from xngin.apiserver.settings import (
    CannotFindTableError,
    Datasource,
    SheetRef,
    XnginSettings,
)
from xngin.apiserver.testing import testing_dwh
from xngin.schema.schema_types import FieldDescriptor, ParticipantsSchema
from xngin.sheets.config_sheet import (
    InvalidSheetError,
    fetch_and_parse_sheet,
)
from xngin.sheets.gsheets import GSheetsPermissionError
from xngin.xsecrets import secretservice

SA_LOGGER_NAME_FOR_CLI = "cli_dwh"

REDSHIFT_HOSTNAME_SUFFIX = "redshift.amazonaws.com"

err_console = Console(stderr=True)
console = Console(stderr=False)
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
            sqlalchemy.create_engine(
                sqlalchemy.engine.make_url(dsn),
                logging_name=SA_LOGGER_NAME_FOR_CLI,
            ),
            table,
            use_reflection=use_reflection,
        )
    except CannotFindTableError as cfte:
        err_console.print(cfte.message)
        raise typer.Exit(1) from cfte
    return create_schema_from_table(dwh, unique_id_col=unique_id_col)


def df_to_ddl(
    df: DataFrame,
    *,
    table_name: str,
    quoter: IdentifierPreparer,
):
    """Helper to transform a DataFrame into a CREATE TABLE statement."""
    # TODO: replace these types with more generic/widely supported defaults.
    default_sql_type = "VARCHAR(255)"
    type_map = {
        "int64": "INTEGER",
        "float64": "DECIMAL",
        "object": "VARCHAR(255)",
        "datetime64[ns]": "TIMESTAMP",
        "bool": "BOOLEAN",
        "uuid": "UUID",  # Special case for UUIDs; not a real Pandas type.
    }
    # Check for UUID columns by trying to parse the first non-null value of string columns as UUID
    df_dtypes = {col: dtype for col, dtype in df.dtypes.items()}
    for col, dtype in df_dtypes.items():
        if dtype.name not in type_map:
            logging.warning(
                "Column '%s' has unknown SQL type for Pandas dtype '%s'. Using default: %s",
                col,
                dtype,
                default_sql_type,
            )
        if dtype == "object":
            nonnulls = df[col].dropna()
            first_val = nonnulls.iloc[0] if not nonnulls.empty else None
            try:
                uuid.UUID(first_val)
                df_dtypes[col] = "uuid"  # override with our special case
            except (ValueError, AttributeError, TypeError):
                pass  # Not a UUID
    # Now generate the DDL.
    columns = [
        f"{quoter.quote(col)} {type_map.get(str(dtype), default_sql_type)}"
        for col, dtype in df_dtypes.items()
    ]
    return f"""CREATE TABLE {table_name} ({",\n    ".join(columns)});"""


def validate_create_testing_dwh_src(v: Path):
    allowed_extensions = (".csv", ".csv.zst")
    for ext in allowed_extensions:
        if str(v).endswith(ext):
            return v
    raise typer.BadParameter("--src must end in .csv or .csv.zst")


@app.command()
def create_testing_dwh(
    dsn: Annotated[str, typer.Option(help="The SQLAlchemy URL for the database.")],
    src: Annotated[
        Path,
        typer.Option(
            help="Local path to the testing data warehouse CSV. This may be zstd-compressed and "
            "must end in .csv or .csv.zst.",
            callback=validate_create_testing_dwh_src,
        ),
    ] = testing_dwh.TESTING_DWH_RAW_DATA,
    nrows: Annotated[
        int | None,
        typer.Option(
            help="Limits the number of rows to load from the CSV. Does not apply to the data load "
            "of Redshift or Postgres connections, but will be applied to schema inference."
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
    create_db: Annotated[
        bool,
        typer.Option(
            help="Create the database if it does not yet exist (Postgres only)."
        ),
    ] = False,
    allow_existing: Annotated[
        bool,
        typer.Option(
            help="True if you only want to create the table if it does not exist."
        ),
    ] = False,
):
    """Loads the testing data warehouse CSV into a database.

    Any existing table will be replaced unless --allow-existing is used.

    For Postgres and Redshift (psycopg or psycopg2) connections, the table DDL will be read from a
    .{postgres|redshift}.ddl file in the same directory as the source CSV, or inferred via Pandas if
    that file does not exist.  The CSV file is parsed using their native server-side CSV parsers.

    Postgres connections may be specified with postgresql://, postgresql+psycopg://, or postgresql+psycopg2:// prefixes.

    Redshift connections must be specified with postgresql+psycopg2:// prefix.

    On BigQuery: CSV is parsed by Pandas. Table DDL is derived by pandas-gbq and written via to_gbq().

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
    print(f"Backend Name: {url.get_backend_name()}")
    print(f"Driver Name: {url.get_driver_name()}")
    print(f"Dialect: {url.get_dialect()}")

    if password is not None and url.username is not None:
        url = url.set(password=password)

    def read_csv():
        return pd.read_csv(src, nrows=nrows)

    def create_engine_and_database(url: sqlalchemy.URL):
        """Connects to a SQLAlchemy URL and creates the database if it doesn't exist.

        Only implemented for psycopg/psycopg2.
        """
        try:
            engine = create_engine(url, logging_name=SA_LOGGER_NAME_FOR_CLI)
            with engine.connect():
                print("Connected.")
        except OperationalError as exc:
            if "postgres" not in url.drivername or (
                # 1st case: psycopg2 driver
                "does not exist" not in str(exc)
                # 2nd case: psycopg driver
                and "Connection refused" not in str(exc)
            ):
                raise
            print(f"Creating database {url.database}...")
            engine = create_engine(
                url.set(database="postgres"),
                logging_name=SA_LOGGER_NAME_FOR_CLI,
            )
            with engine.connect().execution_options(
                isolation_level="AUTOCOMMIT"
            ) as conn:
                conn.execute(sqlalchemy.text(f"CREATE DATABASE {url.database}"))
            print("Reconnecting.")
            return create_engine(
                url,
                logging_name=SA_LOGGER_NAME_FOR_CLI,
            )
        else:
            return engine

    def drop_and_create(cur, create_table_ddl: str):
        cur.execute(drop_table_ddl)
        if schema_name is not None:
            cur.execute(create_schema_ddl)
        print(f"Creating table:\n{create_table_ddl}")
        cur.execute(create_table_ddl)

    def get_ddl_magic(quoter: IdentifierPreparer, flavor: str):
        """Gets the hard-coded DDL if available, or infers it using Pandas."""
        ddl_file = re.sub(r"[.]csv([.]zst)?$", f".{flavor}.ddl", str(src))
        if Path(ddl_file).exists():
            print(f"Using provided DDL from {ddl_file}")
            with open(ddl_file) as inp:
                ddl = inp.read().replace("{{table_name}}", full_table_name)
        else:
            print("Using inferred DDL (warning: may lose fidelity!)")
            ddl = df_to_ddl(
                read_csv(),
                table_name=full_table_name,
                quoter=quoter,
            )
        return ddl

    def count(cur):
        if url.drivername == "bigquery":
            cur.execute(f"SELECT COUNT(*) FROM `{url.database}.{table_name}`")
        else:
            cur.execute(f"SELECT COUNT(*) FROM {full_table_name}")
        ct = cur.fetchone()[0]
        print(f"{full_table_name} has {ct} rows.")
        return ct

    if allow_existing:
        if create_db:
            engine = create_engine_and_database(url)
        else:
            engine = create_engine(url, logging_name=SA_LOGGER_NAME_FOR_CLI)
        conn = engine.raw_connection()
        with conn.cursor() as cur:
            try:
                count(cur)
            except (
                psycopg.errors.UndefinedTable,
                psycopg2.errors.UndefinedTable,
                sqlalchemy.exc.OperationalError,
            ):
                print("🛠️ Table does not exist; creating...\n")
            else:
                print("📣 Table already exists; nothing to do.\n")
                return
        conn.close()
        engine.dispose()

    if url.host and url.host.endswith(REDSHIFT_HOSTNAME_SUFFIX):
        if not bucket:
            print("--bucket is required when importing into Redshift.")
            raise typer.Exit(2)
        if not iam_role:
            print("--iam-role is required when importing into Redshift.")
            raise typer.Exit(2)
        # Workaround: Despite using a direct psycopg2 connection for Redshift, we use SQLAlchemy's quoter.
        engine = create_engine(url, logging_name=SA_LOGGER_NAME_FOR_CLI)
        quoter = engine.dialect.identifier_preparer
        engine.dispose()
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
            ddl = get_ddl_magic(quoter, "redshift")
            drop_and_create(cur, ddl)
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
    elif url.drivername == "bigquery":
        df = read_csv()
        destination_table = f"{url.database}.{table_name}"
        print("Loading using an inferred schema (warning: may lose fidelity!)...")
        pandas_gbq.to_gbq(
            df, destination_table, project_id=url.host, if_exists="replace"
        )
    elif url.get_driver_name() in {"psycopg", "psycopg2"}:
        engine = create_engine_and_database(url)
        ddl = get_ddl_magic(engine.dialect.identifier_preparer, "postgres")
        with engine.begin() as conn:
            cursor = conn.connection.cursor()
            drop_and_create(cursor, ddl)
            opener = (lambda x: zstandard.open(x, "r")) if is_compressed else open
            if url.get_driver_name() == "psycopg":
                print("Loading via psycopg3 COPY FROM STDIN...")
                with opener(src) as reader:
                    cols = [h.strip() for h in reader.readline().split(",")]
                    sql = f"COPY {full_table_name} ({', '.join(cols)}) FROM STDIN (FORMAT CSV, DELIMITER ',')"
                    print(f"SQL: {sql}")
                    with cursor.copy(sql) as copy:
                        while data := reader.read(1 << 20):
                            copy.write(data)
            else:
                print("Loading via psycopg2 copy_expert...")
                with opener(src) as reader:
                    cols = [h.strip() for h in reader.readline().split(",")]
                    sql = f"COPY {full_table_name} ({', '.join(cols)}) FROM STDIN (FORMAT CSV, DELIMITER ',')"
                    print(f"SQL: {sql}")
                    cursor.copy_expert(sql, reader)

            count(cursor)

    else:
        err_console.print("Unrecognized database driver.")
        raise typer.Exit(2)


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
    field_names = [c for c in FieldDescriptor.model_fields if c != "extra"]
    rows = [field_names]

    def convert(v):
        if isinstance(v, bool):
            if v:
                return "true"
            return ""
        if isinstance(v, DataType):
            return str(v)
        return v

    for row in config.fields:
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
    worksheet = sheet.add_worksheet(table_name, rows=len(rows), cols=len(field_names))
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
    except GSheetsPermissionError as pe:
        err_console.print("You do not have permission to open this spreadsheet.")
        raise typer.Exit(1) from pe
    except InvalidSheetError as ise:
        err_console.print(f"Error(s):\n{ise}")
        raise typer.Exit(1) from ise


@app.command()
def export_json_schemas(output: Path = Path(".schemas")):
    """Generates JSON schemas for Xngin settings files."""
    if not output.exists():
        output.mkdir()
    for model in (XnginSettings, ParticipantsSchema, Datasource):
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
        for details in verr.errors():
            print(details, file=sys.stderr)
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


@app.command()
def bigquery_dataset_delete(
    project_id: Annotated[
        str,
        typer.Option(..., help="The Google Cloud Project ID containing the dataset."),
    ],
    dataset_id: Annotated[str, typer.Option(..., help="The dataset name.")],
):
    """Deletes a BigQuery dataset."""
    client = bigquery.Client()
    dataset_ref = f"{project_id}.{dataset_id}"
    try:
        client.delete_dataset(dataset_ref, delete_contents=True)
    except NotFound as exc:
        print(f"Dataset {dataset_ref} does not exist.")
        raise typer.Exit(1) from exc
    else:
        print(f"Dataset {dataset_ref} has been deleted.")


@app.command()
def bigquery_table_delete(
    project_id: Annotated[
        str,
        typer.Option(..., help="The Google Cloud Project ID containing the dataset."),
    ],
    dataset_id: Annotated[str, typer.Option(..., help="The dataset name.")],
    table_id: Annotated[str, typer.Option(..., help="The table name.")],
):
    """Deletes a BigQuery table."""
    client = bigquery.Client()
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    try:
        client.delete_table(table_ref)
    except NotFound as exc:
        print(f"Table {table_ref} does not exist.")
        raise typer.Exit(1) from exc
    else:
        print(f"Table {table_ref} has been deleted.")


def validate_arg_is_email(email: str):
    try:
        return validate_email(email, check_deliverability=False).normalized
    except EmailNotValidError as err:
        raise typer.BadParameter(str(err)) from err


@app.command()
def add_user(
    dsn: Annotated[
        str,
        typer.Option(
            help="The SQLAlchemy DSN of the database where the user should be added.",
            envvar="DATABASE_URL",
        ),
    ],
    email: Annotated[
        str | None,
        typer.Option(
            help="Email address of the user to add. If not provided, will prompt interactively.",
            callback=lambda v: validate_arg_is_email(v) if v else None,
            envvar="XNGIN_ADD_USER_EMAIL",
        ),
    ] = None,
    privileged: Annotated[
        bool,
        typer.Option(help="Whether the user should have privileged access."),
    ] = False,
    dwh: Annotated[
        str | None,
        typer.Option(
            help="The SQLAlchemy DSN of a DWH to be added to the user's organization.",
            envvar="XNGIN_DEVDWH_DSN",
        ),
    ] = None,
):
    """Adds a new user to the database.

    This command connects to the specified database and adds a new user with the given email address.
    If the --privileged flag is set, the user will be granted privileged access.

    If email is not provided via the --email flag, the command will prompt for it interactively.
    """
    console.print(f"DSN: [cyan]{dsn}[/cyan]")
    console.print(f"DWH: [cyan]{dwh}[/cyan]")

    if email is None:
        while True:
            email_input = typer.prompt("Enter email address")
            try:
                email = validate_email(
                    email_input, check_deliverability=False
                ).normalized
                break
            except EmailNotValidError as err:
                err_console.print(f"[bold red]Invalid email:[/bold red] {err!s}")

        privileged = typer.confirm(
            "Should this user have privileged access?", default=False
        )

    console.print(f"Adding user with email: [cyan]{email}[/cyan]")
    console.print(f"Privileged access: [cyan]{privileged}[/cyan]")

    if not dwh:
        console.print(
            "\n[bold yellow]Warning: Not adding a datasource for a data warehouse "
            "because the --dwh flag was not specified or environment variable "
            "XNGIN_DEVDWH_DSN is unset.[/bold yellow]"
        )

    engine = create_engine(dsn)
    with Session(engine) as session:
        try:
            user = testing_dwh.create_user_and_first_datasource(
                session, email=email, dsn=dwh, privileged=privileged
            )
            session.commit()
            console.print("\n[bold green]User added successfully:[/bold green]")
            console.print(f"User ID: [cyan]{user.id}[/cyan]")
            console.print(f"Email: [cyan]{user.email}[/cyan]")
            console.print(f"Privileged: [cyan]{user.is_privileged}[/cyan]")
            for organization in user.organizations:
                console.print(f"Organization ID: [cyan]{organization.id}[/cyan]")
                for datasource in organization.datasources:
                    console.print(f"  Datasource ID: [cyan]{datasource.id}[/cyan]")
        except IntegrityError as err:
            session.rollback()
            err_console.print(
                f"[bold red]Error:[/bold red] User with email '{email}' already exists."
            )
            raise typer.Exit(1) from err


class OutputFormat(StrEnum):
    base64 = "base64"
    json = "json"


@app.command()
def create_tink_key(
    output: Annotated[
        OutputFormat,
        typer.Option(
            help="Output format. Use base64 when generating a key for use in an environment variable."
        ),
    ] = OutputFormat.base64,
):
    """Generate an encryption key for the "local" secret storage backend.

    The encoded encryption keyset (specifically, a Tink keyset with a single key) will be written to stdout. This value
    is suitable for use as the XNGIN_SECRETS_TINK_KEYSET environment variable.
    """
    tink.aead.register()
    keyset_handle = tink.new_keyset_handle(tink.aead.aead_key_templates.AES128_GCM)
    # Remove superfluous spaces by decoding and re-encoding.
    keyset = json.dumps(
        json.loads(
            tink.json_proto_keyset_format.serialize(
                keyset_handle, secret_key_access.TOKEN
            )
        ),
        separators=(",", ":"),
    )
    if output == "base64":
        print(base64.standard_b64encode(keyset.encode("utf-8")).decode("utf-8"))
    else:
        print(keyset)


@app.command()
def encrypt(
    aad: Annotated[str, typer.Option()] = "cli",
):
    """Encrypts a string using the same encryption configuration that the API server does."""
    secretservice.setup()
    plaintext = sys.stdin.read()
    print(secretservice.get_symmetric().encrypt(plaintext, aad))


@app.command()
def decrypt(aad: Annotated[str, typer.Option()] = "cli"):
    """Decrypts a string using the same encryption configuration that the API server does."""
    secretservice.setup()
    ciphertext = sys.stdin.read()
    print(secretservice.get_symmetric().decrypt(ciphertext, aad))


if __name__ == "__main__":
    app()
