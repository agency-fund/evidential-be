"""`create-testing-dwh` command: loads the testing CSV into a dev data warehouse."""

import logging
import re
import uuid
from compression import zstd
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import psycopg
import psycopg2
import sqlalchemy
import typer
from rich.console import Console
from sqlalchemy import create_engine, make_url

from xngin.apiserver.dwh import dwh_utils
from xngin.cli.common import SA_LOGGER_NAME_FOR_CLI, create_engine_and_database

if TYPE_CHECKING:
    from pandas import DataFrame
    from sqlalchemy.sql.compiler import IdentifierPreparer

_TESTING_DWH_RAW_DATA = Path(__file__).resolve().parent.parent.parent / "apiserver/testdata/testing_dwh.csv.zst"

_err_console = Console(stderr=True)


def _truncate_with_ellipsis(value: str) -> str:
    if len(value) > 250:
        return value[:247] + "..."
    return value


def _validate_src(v: Path) -> Path:
    allowed_extensions = (".csv", ".csv.zst")
    for ext in allowed_extensions:
        if str(v).endswith(ext):
            return v
    raise typer.BadParameter("--src must end in .csv or .csv.zst")


def _df_to_ddl(
    df: DataFrame,
    *,
    table_name: str,
    quoter: IdentifierPreparer,
) -> str:
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


@dataclass(frozen=True)
class _Source:
    """The CSV file being loaded."""

    path: Path
    is_compressed: bool
    nrows: int | None


@dataclass(frozen=True)
class _Dest:
    """The destination database, schema, and table."""

    url: sqlalchemy.URL
    schema_name: str | None
    table_name: str
    full_table_name: str
    create_schema_ddl: str
    drop_table_ddl: str
    views: str | None


def _build_source(*, src: Path, nrows: int | None) -> _Source:
    return _Source(path=src, is_compressed=src.suffix == ".zst", nrows=nrows)


def _build_dest(
    *,
    dsn: str,
    schema_name: str | None,
    table_name: str,
    password: str | None,
    views: str | None,
) -> _Dest:
    url = make_url(dsn)
    print(
        f"create_testing_dwh: {url.set(password=None)} backend={url.get_backend_name()} "
        f"driver={url.get_driver_name()} dialect={url.get_dialect().__name__}"
    )

    if password is not None and url.username is not None:
        url = url.set(password=password)

    full_table_name = f"{schema_name}.{table_name}" if schema_name else table_name
    return _Dest(
        url=url,
        schema_name=schema_name,
        table_name=table_name,
        full_table_name=full_table_name,
        create_schema_ddl=f"CREATE SCHEMA IF NOT EXISTS {schema_name}" if schema_name else "",
        drop_table_ddl=f"DROP TABLE IF EXISTS {full_table_name}",
        views=views,
    )


def _read_csv(source: _Source) -> DataFrame:
    from pandas import read_csv as pd_read_csv  # noqa: PLC0415

    return pd_read_csv(source.path, nrows=source.nrows)


def _get_ddl_magic(source: _Source, dest: _Dest, quoter: IdentifierPreparer, flavor: str) -> str:
    """Gets the hard-coded DDL if available, or infers it using Pandas."""
    ddl_file = re.sub(r"[.]csv([.]zst)?$", f".{flavor}.ddl", str(source.path))
    if Path(ddl_file).exists():
        print(f"Using provided DDL from {ddl_file}")
        with open(ddl_file) as inp:
            ddl = inp.read().replace("{{table_name}}", dest.full_table_name)
    else:
        print("Using inferred DDL (warning: may lose fidelity!)")
        ddl = _df_to_ddl(_read_csv(source), table_name=dest.full_table_name, quoter=quoter)
    return ddl


def _drop_and_create(cur, dest: _Dest, create_table_ddl: str) -> None:
    if dest.schema_name is not None:
        cur.execute(dest.create_schema_ddl)
    cur.execute(dest.drop_table_ddl)
    cur.execute(create_table_ddl)


def _exists(cur, dest: _Dest):
    if dwh_utils.is_bigquery(dest.url):
        cur.execute(f"SELECT 1 FROM `{dest.url.database}.{dest.table_name}` LIMIT 0")  # noqa: S608
    else:
        cur.execute(f"SELECT 1 FROM {dest.full_table_name} LIMIT 0")  # noqa: S608


def _count(cur, dest: _Dest):
    if dwh_utils.is_bigquery(dest.url):
        cur.execute(f"SELECT COUNT(*) FROM `{dest.url.database}.{dest.table_name}`")  # noqa: S608
    else:
        cur.execute(f"SELECT COUNT(*) FROM {dest.full_table_name}")  # noqa: S608
    return cur.fetchone()[0]


def _maybe_create_views(cur, dest: _Dest) -> None:
    if not dest.views:
        return
    view_names = [s.strip() for s in dest.views.split(",")]
    print(f"Creating views: {', '.join(sorted(view_names))}...")
    for view_name in view_names:
        qualified_view_name = f"{dest.schema_name}.{view_name}" if dest.schema_name else view_name
        cur.execute(
            f"CREATE OR REPLACE VIEW {qualified_view_name} AS SELECT * FROM {dest.full_table_name}"  # noqa: S608
        )


def _table_exists_and_skip(dest: _Dest, *, create_db: bool) -> bool:
    """Returns True iff the table already exists and we should skip the load."""
    if create_db:
        engine = create_engine_and_database(dest.url)
    else:
        engine = create_engine(dest.url, logging_name=SA_LOGGER_NAME_FOR_CLI)
    conn = engine.raw_connection()
    skip = False
    try:
        with conn.cursor() as cur:
            try:
                _exists(cur, dest)
                ct = _count(cur, dest)
            except (
                psycopg.errors.UndefinedTable,
                psycopg2.errors.UndefinedTable,
                sqlalchemy.exc.OperationalError,
            ):
                print(f"Table {dest.table_name} does not exist; creating...\n")
            else:
                print(f"Table {dest.table_name} already exists (nrows={ct}).\n")
                skip = True
    finally:
        conn.close()
        engine.dispose()
    return skip


def _load_redshift(source: _Source, dest: _Dest, *, bucket: str | None, iam_role: str | None) -> None:
    import boto3  # noqa: PLC0415

    if not bucket:
        print("--bucket is required when importing into Redshift.")
        raise typer.Exit(2)
    if not iam_role:
        print("--iam-role is required when importing into Redshift.")
        raise typer.Exit(2)
    # Workaround: Despite using a direct psycopg2 connection for Redshift, we use SQLAlchemy's quoter.
    engine = create_engine(dest.url, logging_name=SA_LOGGER_NAME_FOR_CLI)
    quoter = engine.dialect.identifier_preparer
    engine.dispose()
    with (
        psycopg2.connect(
            database=dest.url.database,
            host=dest.url.host,
            password=dest.url.password,
            port=dest.url.port,
            user=dest.url.username,
        ) as conn,
        conn.cursor() as cur,
    ):
        ddl = _get_ddl_magic(source, dest, quoter, "redshift")
        _drop_and_create(cur, dest, ddl)
        key = source.path.name
        print(f"Uploading to s3://{bucket}/{key}...")
        s3 = boto3.client("s3")
        s3.upload_file(source.path, bucket, f"{key}")
        try:
            print("Loading...")
            compression_hint = "ZSTD" if source.is_compressed else ""
            cur.execute(
                f"COPY {dest.full_table_name} FROM 's3://{bucket}/{key}' "
                f"IAM_ROLE '{iam_role}' FORMAT CSV IGNOREHEADER 1 {compression_hint};"
            )
            _count(cur, dest)
        finally:
            print("Deleting temporary file...")
            s3.delete_object(Bucket=bucket, Key=key)
        _maybe_create_views(cur, dest)


def _load_bigquery(source: _Source, dest: _Dest) -> None:
    import pandas_gbq  # noqa: PLC0415

    df = _read_csv(source)
    destination_table = f"{dest.url.database}.{dest.table_name}"
    print("Loading using an inferred schema (warning: may lose fidelity!)...")
    pandas_gbq.to_gbq(df, destination_table, project_id=dest.url.host, if_exists="replace")


def _load_postgres(source: _Source, dest: _Dest, *, create_db: bool) -> None:
    engine = (
        create_engine_and_database(dest.url)
        if create_db
        else create_engine(dest.url, logging_name=SA_LOGGER_NAME_FOR_CLI)
    )
    ddl = _get_ddl_magic(source, dest, engine.dialect.identifier_preparer, "postgres")
    with engine.begin() as conn:
        cursor = conn.connection.cursor()
        _drop_and_create(cursor, dest, ddl)
        opener = (lambda x: zstd.open(x, "rt")) if source.is_compressed else open
        if dest.url.get_driver_name() == "psycopg":
            print("Loading via psycopg3 COPY FROM STDIN...")
            with opener(source.path) as reader:
                cols = [h.strip() for h in reader.readline().split(",")]
                sql = f"COPY {dest.full_table_name} ({', '.join(cols)}) FROM STDIN (FORMAT CSV, DELIMITER ',')"
                print(f"SQL: {_truncate_with_ellipsis(sql)}")
                with cursor.copy(sql) as copy:
                    while data := reader.read(1 << 20):
                        copy.write(data)
        else:
            print("Loading via psycopg2 copy_expert...")
            with opener(source.path) as reader:
                cols = [h.strip() for h in reader.readline().split(",")]
                sql = f"COPY {dest.full_table_name} ({', '.join(cols)}) FROM STDIN (FORMAT CSV, DELIMITER ',')"
                print(f"SQL: {_truncate_with_ellipsis(sql)}")
                cursor.copy_expert(sql, reader)

        print(f"Loaded {cursor.rowcount} rows into {dest.full_table_name}.")
        _maybe_create_views(cursor, dest)


def create_testing_dwh(
    dsn: Annotated[str, typer.Option(help="The SQLAlchemy URL for the database.")],
    src: Annotated[
        Path,
        typer.Option(
            help="Local path to the testing data warehouse CSV. This may be zstd-compressed and "
            "must end in .csv or .csv.zst.",
            callback=_validate_src,
        ),
    ] = _TESTING_DWH_RAW_DATA,
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
            help="Name of the temporary S3 bucket that is readable by Redshift when --iam-role is "
            "assumed, and writable with your own AWS credentials to upload & delete the temp data. "
            "Required when connecting to Redshift."
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
        typer.Option(help="Create the database if it does not yet exist (Postgres and Redshift only)."),
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

    This command does not support schemas or non-trivial database names or table names consistently.
    """
    source = _build_source(src=src, nrows=nrows)
    dest = _build_dest(
        dsn=dsn,
        schema_name=schema_name,
        table_name=table_name,
        password=password,
        views=views,
    )

    if allow_existing and _table_exists_and_skip(dest, create_db=create_db):
        return

    if dwh_utils.is_redshift(dest.url):
        _load_redshift(source, dest, bucket=bucket, iam_role=iam_role)
    elif dwh_utils.is_bigquery(dest.url):
        _load_bigquery(source, dest)
    elif dwh_utils.is_postgres(dest.url):
        _load_postgres(source, dest, create_db=create_db)
    else:
        _err_console.print("Unrecognized database driver.")
        raise typer.Exit(2)


def register(app: typer.Typer) -> None:
    app.command()(create_testing_dwh)
