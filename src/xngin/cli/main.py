"""Command line tool for various xngin-related operations."""

import asyncio
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from compression import zstd
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import psycopg
import psycopg2
import sqlalchemy
import typer
from email_validator import EmailNotValidError, validate_email
from rich.console import Console
from sqlalchemy import create_engine, make_url
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.sql.compiler import IdentifierPreparer

from xngin.xsecrets import secretservice

if TYPE_CHECKING:
    from pandas import DataFrame

SA_LOGGER_NAME_FOR_CLI = "cli_dwh"
CLI_DB_APPLICATION_NAME = f"cli-{os.getpid()}"

REDSHIFT_HOSTNAME_SUFFIX = "redshift.amazonaws.com"
TESTING_DWH_RAW_DATA = Path(__file__).resolve().parent.parent / "apiserver/testdata/testing_dwh.csv.zst"

err_console = Console(stderr=True)
console = Console(stderr=False)
app = typer.Typer(help=__doc__)
snapshots_app = typer.Typer(help="Create and modify fake historical snapshots for development.")
app.add_typer(snapshots_app, name="snapshots")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

secretservice.setup(allow_noop=True)

async_command = lambda f: functools.wraps(f)(lambda *args, **kwargs: asyncio.run(f(*args, **kwargs)))  # noqa: E731


class Base64OrJson(StrEnum):
    base64 = "base64"
    json = "json"


def truncate_with_ellipsis(value: str) -> str:
    if len(value) > 250:
        return value[:247] + "..."
    return value


def create_engine_and_database(url: sqlalchemy.URL, *, connect_args: dict | None = None):
    """Connects to a SQLAlchemy URL and creates the database if it doesn't exist.

    Only implemented for psycopg/psycopg2.
    """
    connect_args = connect_args or {}
    try:
        engine = create_engine(url, connect_args=connect_args, logging_name=SA_LOGGER_NAME_FOR_CLI)
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
            url.set(database="postgres"), connect_args=connect_args, logging_name=SA_LOGGER_NAME_FOR_CLI
        )
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(sqlalchemy.text(f"CREATE DATABASE {url.database}"))
        print("Reconnecting.")
        return create_engine(url, connect_args=connect_args, logging_name=SA_LOGGER_NAME_FOR_CLI)
    else:
        return engine


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
            except ValueError, AttributeError, TypeError:
                pass  # Not a UUID
    # Now generate the DDL.
    columns = [
        f"{quoter.quote(str(col))} {type_map.get(str(dtype), default_sql_type)}" for col, dtype in df_dtypes.items()
    ]
    return f"""CREATE TABLE {table_name} ({",\n    ".join(columns)});"""


def validate_create_testing_dwh_src(v: Path):
    allowed_extensions = (".csv", ".csv.zst")
    for ext in allowed_extensions:
        if str(v).endswith(ext):
            return v
    raise typer.BadParameter("--src must end in .csv or .csv.zst")


def parse_iso_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


@app.command()
def create_apiserver_db(
    dsn: Annotated[
        str,
        typer.Option(help="The SQLAlchemy URL for the database.", envvar="DATABASE_URL"),
    ],
):
    from xngin.apiserver.sqla import tables  # noqa: PLC0415

    console.print(f"DSN: [cyan]{dsn}[/cyan]")
    engine = create_engine_and_database(make_url(dsn), connect_args={"application_name": CLI_DB_APPLICATION_NAME})
    tables.Base.metadata.create_all(bind=engine)


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
    ] = TESTING_DWH_RAW_DATA,
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
    password: Annotated[str | None, typer.Option(envvar="PGPASSWORD", help="The database password.")] = None,
    create_db: Annotated[
        bool,
        typer.Option(help="Create the database if it does not yet exist (Postgres only)."),
    ] = False,
    allow_existing: Annotated[
        bool,
        typer.Option(help="True if you only want to create the table if it does not exist."),
    ] = False,
    views: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated view names to create as aliases for the dwh table. "
            "Only supported on Postgres and Redshift."
        ),
    ] = None,
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

    create_schema_ddl = f"CREATE SCHEMA IF NOT EXISTS {schema_name}" if schema_name else ""
    full_table_name = f"{schema_name}.{table_name}" if schema_name else table_name
    drop_table_ddl = f"DROP TABLE IF EXISTS {full_table_name}"
    is_compressed = src.suffix == ".zst"

    url = make_url(dsn)
    print(f"create_testing_dwh for: {url.set(password=None)}")
    print(f"\tBackend Name: {url.get_backend_name()}")
    print(f"\tDriver Name: {url.get_driver_name()}")
    print(f"\tDialect: {url.get_dialect()}")

    if password is not None and url.username is not None:
        url = url.set(password=password)

    def read_csv():
        from pandas import read_csv as pd_read_csv  # noqa: PLC0415

        return pd_read_csv(src, nrows=nrows)

    def drop_and_create(cur, create_table_ddl: str):
        cur.execute(drop_table_ddl)
        if schema_name is not None:
            cur.execute(create_schema_ddl)
        print(f"Creating table:\n{truncate_with_ellipsis(create_table_ddl)}")
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

    def maybe_create_views(cur):
        if not views:
            return
        for view_name in views.split(","):
            qualified_view_name = f"{schema_name}.{view_name}" if schema_name else view_name
            print(f"Creating view {qualified_view_name}...")
            cur.execute(f"CREATE OR REPLACE VIEW {qualified_view_name} AS SELECT * FROM {full_table_name}")

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
        import boto3  # noqa: PLC0415

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
                compression_hint = "ZSTD" if is_compressed else ""
                cur.execute(
                    f"COPY {full_table_name} FROM 's3://{bucket}/{key}' "
                    f"IAM_ROLE '{iam_role}' FORMAT CSV IGNOREHEADER 1 {compression_hint};"
                )
                count(cur)
            finally:
                print("Deleting temporary file...")
                s3.delete_object(Bucket=bucket, Key=key)
            maybe_create_views(cur)
    elif url.drivername == "bigquery":
        import pandas_gbq  # noqa: PLC0415

        df = read_csv()
        destination_table = f"{url.database}.{table_name}"
        print("Loading using an inferred schema (warning: may lose fidelity!)...")
        pandas_gbq.to_gbq(df, destination_table, project_id=url.host, if_exists="replace")
    elif url.get_driver_name() in {"psycopg", "psycopg2"}:
        engine = create_engine_and_database(url)
        ddl = get_ddl_magic(engine.dialect.identifier_preparer, "postgres")
        with engine.begin() as conn:
            cursor = conn.connection.cursor()
            drop_and_create(cursor, ddl)
            opener = (lambda x: zstd.open(x, "rt")) if is_compressed else open
            if url.get_driver_name() == "psycopg":
                print("Loading via psycopg3 COPY FROM STDIN...")
                with opener(src) as reader:
                    cols = [h.strip() for h in reader.readline().split(",")]
                    sql = f"COPY {full_table_name} ({', '.join(cols)}) FROM STDIN (FORMAT CSV, DELIMITER ',')"
                    print(f"SQL: {truncate_with_ellipsis(sql)}")
                    with cursor.copy(sql) as copy:
                        while data := reader.read(1 << 20):
                            copy.write(data)
            else:
                print("Loading via psycopg2 copy_expert...")
                with opener(src) as reader:
                    cols = [h.strip() for h in reader.readline().split(",")]
                    sql = f"COPY {full_table_name} ({', '.join(cols)}) FROM STDIN (FORMAT CSV, DELIMITER ',')"
                    print(f"SQL: {truncate_with_ellipsis(sql)}")
                    cursor.copy_expert(sql, reader)

            count(cursor)
            maybe_create_views(cursor)

    else:
        err_console.print("Unrecognized database driver.")
        raise typer.Exit(2)


@app.command()
def export_json_schemas(output: Path = Path(".schemas")):
    """Generates JSON schemas for Xngin settings files."""
    from xngin.apiserver.dwh.inspection_types import ParticipantsSchema  # noqa: PLC0415
    from xngin.apiserver.settings import Datasource  # noqa: PLC0415

    if not output.exists():
        output.mkdir()
    for model in (ParticipantsSchema, Datasource):
        filename = output / (model.__name__ + ".schema.json")
        with open(filename, "w") as outf:
            outf.write(json.dumps(model.model_json_schema(), indent=2, sort_keys=True))
            print(f"Wrote {filename}.")


@app.command()
def export_openapi_spec(output: Path = Path("openapi.json")):
    """Writes the OpenAPI spec to the file specified by --output."""
    from fastapi import FastAPI  # noqa: PLC0415

    import xngin.apiserver.openapi  # noqa: PLC0415

    app = FastAPI()
    from xngin.apiserver import routes  # noqa: PLC0415

    routes.register(app)
    with open(output, "w") as outf:
        json.dump(xngin.apiserver.openapi.custom_openapi(app), outf, sort_keys=True, indent=2)


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
    from google.cloud import bigquery  # noqa: PLC0415

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
    from google.cloud import bigquery  # noqa: PLC0415
    from google.cloud.exceptions import NotFound  # noqa: PLC0415

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
    from google.cloud import bigquery  # noqa: PLC0415
    from google.cloud.exceptions import NotFound  # noqa: PLC0415

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
@async_command
async def add_user(
    database_url: Annotated[
        str,
        typer.Option(
            help="The application database where the user should be added.",
            envvar="DATABASE_URL",
        ),
    ],
    email: Annotated[
        str | None,
        typer.Argument(
            help="Email address of the user to add. If not provided, will prompt interactively.",
            callback=lambda v: validate_arg_is_email(v) if v else None,
            envvar="XNGIN_ADD_USER_EMAIL",
        ),
    ],
    privileged: Annotated[
        bool,
        typer.Option(help="Whether the user should have privileged access."),
    ] = False,
    dwh: Annotated[
        str | None,
        typer.Option(
            help="The SQLAlchemy URI of a DWH to be added to the user's organization.",
            envvar="XNGIN_DEVDWH_DSN",
        ),
    ] = None,
):
    """Adds a new user to the database.

    This command connects to the specified database and adds a new user with the given email address.
    If the --privileged flag is set, the user will be granted privileged access.

    If email is not provided as an argument, the command will prompt for it interactively.

    This command is only useful for local development databases; do not use it against production databases.
    """
    from xngin.apiserver.sqla import tables  # noqa: PLC0415
    from xngin.apiserver.storage.bootstrap import create_entities_for_first_time_user  # noqa: PLC0415

    console.print(f"Using application database: [cyan]{database_url}[/cyan]")
    console.print(f"Using data warehouse: [cyan]{dwh}[/cyan]")

    console.print(f"Adding user with email: [cyan]{email}[/cyan]")
    console.print(f"Privileged access: [cyan]{privileged}[/cyan]")

    if not dwh:
        console.print(
            "\n[bold yellow]Warning: Not adding a datasource for a data warehouse "
            "because the --dwh flag was not specified or environment variable "
            "XNGIN_DEVDWH_DSN is unset.[/bold yellow]"
        )

    engine = create_async_engine(database_url, connect_args={"application_name": CLI_DB_APPLICATION_NAME})
    async with AsyncSession(engine) as session:
        try:
            user = await create_entities_for_first_time_user(
                session, tables.User(email=email, is_privileged=privileged), dwh
            )
            await session.commit()
            await session.refresh(user)
            console.print("\n[bold green]User added successfully:[/bold green]")
            console.print(f"User ID: [cyan]{user.id}[/cyan]")
            console.print(f"Email: [cyan]{user.email}[/cyan]")
            console.print(f"Privileged: [cyan]{user.is_privileged}[/cyan]")
            api_keys = {}
            for organization in await user.awaitable_attrs.organizations:
                console.print(f"Organization: [cyan]{organization.name}[/cyan] (ID: {organization.id})")
                for datasource in await organization.awaitable_attrs.datasources:
                    from xngin.apiserver import apikeys  # noqa: PLC0415

                    label, key = apikeys.make_key()
                    key_hash = apikeys.hash_key_or_raise(key)
                    api_keys[datasource.id] = key
                    (await datasource.awaitable_attrs.api_keys).append(tables.ApiKey(id=label, key=key_hash))
                    console.print(
                        f"  Datasource: [cyan]{datasource.name}[/cyan] "
                        f"(ID: {datasource.id}) "
                        f"[blue](API Key: {key})[/blue]"
                    )
                    for experiment in await datasource.awaitable_attrs.experiments:
                        console.print(f"    Experiment: [cyan]{experiment.name}[/cyan] (ID: {experiment.id})")
        except IntegrityError as err:
            await session.rollback()
            err_console.print(f"[bold red]Error:[/bold red] {err}")
            raise typer.Exit(1) from err


@app.command()
def create_nacl_keyset(
    output: Annotated[
        Base64OrJson,
        typer.Option(help="Output format. Use base64 when generating a key for use in an environment variable."),
    ] = Base64OrJson.base64,
):
    """Generate an encryption keyset for the "nacl" secret provider.

    The encoded encryption key will be written to stdout.

    When --output=base64 (default), the output can be used as the XNGIN_SECRETS_NACL_KEYSET environment variable.
    """
    from xngin.xsecrets.nacl_provider import NaclProviderKeyset  # noqa: PLC0415

    keyset = NaclProviderKeyset.create()
    if output == Base64OrJson.base64:
        print(keyset.serialize_base64())
    else:
        print(keyset.serialize_json())


@app.command()
def encrypt(
    aad: Annotated[
        str,
        typer.Option(help="Bind the ciphertext to this additionally authenticated data (AAD)."),
    ] = "cli",
):
    """Encrypts a string using the same encryption configuration that the API server does."""
    plaintext = sys.stdin.read()
    print(secretservice.get_symmetric().encrypt(plaintext, aad))


@app.command()
def decrypt(
    aad: Annotated[str, typer.Option(help="The AAD specified when the ciphertext was encrypted.")] = "cli",
):
    """Decrypts a string using the same encryption configuration that the API server does."""
    ciphertext = sys.stdin.read()
    print(secretservice.get_symmetric().decrypt(ciphertext, aad))


@app.command()
def generate_typed_clients():
    """Generates strongly typed API clients from the FastAPI definitions."""
    # dev-only dependency
    import fastapi_typed_client  # noqa: PLC0415

    root = Path("src/xngin/apiserver/testing")
    eapi_path = root / "experiments_api_client.py"
    aapi_path = root / "admin_api_client.py"
    iadminapi_path = root / "admin_integrations_api_client.py"
    iapi_path = root / "integrations_api_client.py"

    print(f"Generating ExperimentsAPIClient: {eapi_path}")
    fastapi_typed_client.generate_fastapi_typed_client(
        "xngin.apiserver.routers.experiments.experiments_api:router",
        include_security_params=True,
        output_path=eapi_path,
        raise_if_not_default_status=True,
        title="ExperimentsAPIClient",
    )
    print(f"Generating AdminAPIClient: {aapi_path}")
    fastapi_typed_client.generate_fastapi_typed_client(
        "xngin.apiserver.routers.admin.admin_api:router",
        output_path=aapi_path,
        raise_if_not_default_status=True,
        title="AdminAPIClient",
    )
    print(f"Generating AdminIntegrationsAPIClient: {iadminapi_path}")
    fastapi_typed_client.generate_fastapi_typed_client(
        "xngin.apiserver.routers.admin_integrations.admin_integration_api:router",
        output_path=iadminapi_path,
        raise_if_not_default_status=True,
        title="AdminIntegrationsAPIClient",
    )

    print(f"Generating IntegrationsAPIClient: {iapi_path}")
    fastapi_typed_client.generate_fastapi_typed_client(
        "xngin.apiserver.routers.integrations.integrations_api:router",
        include_security_params=True,
        output_path=iapi_path,
        raise_if_not_default_status=True,
        title="IntegrationsAPIClient",
    )

    ruff_bin = shutil.which("ruff")
    if ruff_bin is None:
        return

    print("Formatting generated files...")
    try:
        subprocess.run(
            [
                ruff_bin,
                "format",
                eapi_path,
                aapi_path,
                iadminapi_path,
                iapi_path,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        err_console.print(f"[bold red]Error:[/bold red] ruff formatting failed: {exc}")
        raise typer.Exit(1) from exc


@snapshots_app.command("create-fake")
def snapshots_create_fake(
    dsn: Annotated[str, typer.Option("--dsn", "-d", help="Database connection string", envvar="DATABASE_URL")],
    exp_id: Annotated[str, typer.Option("--exp-id", help="Experiment ID")],
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", "-s", help="Start date in ISO format. Defaults to now."),
    ] = None,
    n: Annotated[int, typer.Option("--n", "-n", help="Number of daily snapshots to create.")] = 1,
    arm_id: Annotated[str | None, typer.Option("--arm-id", "-a", help="Arm ID to apply values to")] = None,
    metric: Annotated[str | None, typer.Option("--metric", "-m", help="Metric name to apply values to")] = None,
    field: Annotated[
        str | None,
        typer.Option("--field", "-f", help="Field name to override in generated analyses."),
    ] = None,
    values: Annotated[list[float] | None, typer.Argument(help="Optional values to cycle through")] = None,
    random_seed: Annotated[int | None, typer.Option("--random-seed", "-r", help="Random seed")] = None,
    echo: Annotated[bool, typer.Option("--echo", help="Echo SQL queries")] = False,
) -> None:
    """Create fake snapshots for a frequentist experiment."""
    from xngin.apiserver.snapshots.fake_data import (  # noqa: PLC0415
        VALID_SNAPSHOT_FIELDS,
        create_fake_snapshots,
        get_arm_ids,
        get_freq_experiment_for_cli,
        get_metric_names,
    )

    engine = create_engine(dsn, connect_args={"application_name": CLI_DB_APPLICATION_NAME}, echo=echo)

    with Session(engine) as session:
        try:
            experiment = get_freq_experiment_for_cli(session, exp_id)
        except ValueError as err:
            err_console.print(f"Error: {err}")
            raise typer.Exit(1) from err

        if metric and metric not in get_metric_names(experiment):
            err_console.print(
                f"Error: metric '{metric}' not found in experiment. Available: {get_metric_names(experiment)}"
            )
            raise typer.Exit(1)

        if arm_id and arm_id not in get_arm_ids(experiment):
            err_console.print(f"Error: arm_id '{arm_id}' not found in experiment. Available: {get_arm_ids(experiment)}")
            raise typer.Exit(1)

        if field and field not in VALID_SNAPSHOT_FIELDS:
            err_console.print(f"Error: field '{field}' not valid. Must be one of: {VALID_SNAPSHOT_FIELDS}")
            raise typer.Exit(1)

        snapshots = create_fake_snapshots(
            session,
            experiment,
            start_date=parse_iso_datetime(start_date),
            n=n,
            arm_id=arm_id,
            metric_name=metric,
            field=field,
            values=values,
            random_seed=random_seed,
        )
        session.commit()

    print(f"Successfully created {len(snapshots)} snapshots for experiment {exp_id}")


if __name__ == "__main__":
    app()
