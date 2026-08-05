"""Helpers shared by `cli/main.py` and the per-command modules under `cli/commands/`."""

import os
from typing import NoReturn

import sqlalchemy
import typer
from rich.console import Console
from sqlalchemy import create_engine
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from xngin.apiserver.dwh import dwh_utils

SA_LOGGER_NAME_FOR_CLI = "cli_dwh"

CLI_DB_APPLICATION_NAME = f"cli-{os.getpid()}"

# Postgres: CREATE DATABASE and DROP DATABASE run against the "postgres" database because it reliably exists.
POSTGRES_MAINTENANCE_DATABASE = "postgres"

# Redshift: "dev" exists on every cluster.
REDSHIFT_MAINTENANCE_DATABASE = "dev"

# Postgres error codes
PG_DUPLICATE_DATABASE = "42P04"
PG_UNIQUE_VIOLATION = "23505"

console = Console()
err_console = Console(stderr=True)


def fail(message: str, *, code: int = 1) -> NoReturn:
    """Prints an error to stderr and exits.

    Use code 2 to indicate user error.
    """
    err_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code)


def sqlstate_of(exc: BaseException) -> str | None:
    """Returns the SQLSTATE of the DBAPI error underlying a SQLAlchemy exception, if there is one.

    psycopg exposes this as `sqlstate` and psycopg2 as `pgcode`; other dialects may expose neither. Accepts either a
    SQLAlchemy wrapper (whose `orig` holds the DBAPI error) or a bare DBAPI error.
    """
    orig = getattr(exc, "orig", exc)
    return getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)


def cli_connect_args(url: sqlalchemy.URL) -> dict:
    """Returns the connect_args identifying this process to the server, for dialects that accept them.

    Only the Postgres drivers understand application_name.
    """
    if dwh_utils.is_postgres(url):
        return {"application_name": CLI_DB_APPLICATION_NAME}
    return {}


def cli_engine(url: sqlalchemy.URL | str, *, connect_args: dict | None = None, echo: bool = False) -> sqlalchemy.Engine:
    """Creates an Engine with the logging name, application name, and dialect workarounds the CLI expects."""
    url = sqlalchemy.make_url(url)
    engine = create_engine(
        url, connect_args=cli_connect_args(url) | (connect_args or {}), logging_name=SA_LOGGER_NAME_FOR_CLI, echo=echo
    )
    dwh_utils.extra_engine_setup(engine)
    return engine


def cli_async_engine(url: sqlalchemy.URL | str) -> AsyncEngine:
    """Creates an Engine comparable to what cli_engine would create, but async."""
    url = sqlalchemy.make_url(url)
    engine = create_async_engine(url, connect_args=cli_connect_args(url), logging_name=SA_LOGGER_NAME_FOR_CLI)
    dwh_utils.extra_engine_setup(engine.sync_engine)
    return engine


def maintenance_database_for(url: sqlalchemy.URL) -> str:
    """Returns the name of the database to connect to in order to create or drop the database named in the URL."""
    return REDSHIFT_MAINTENANCE_DATABASE if dwh_utils.is_redshift(url) else POSTGRES_MAINTENANCE_DATABASE


def ensure_database_exists(url: sqlalchemy.URL) -> None:
    """Creates the database named in the URL if it does not already exist.

    Only Postgres-family targets (which includes Redshift) are supported; other dialects are left untouched, as
    their "databases" are provisioned out of band.
    """
    if not dwh_utils.is_postgres(url):
        return

    engine = cli_engine(url)
    try:
        with engine.connect():
            print("Connected.")
            return
    except OperationalError as exc:
        # A missing database surfaces as a bare OperationalError carrying no SQLSTATE, so the message is the only
        # signal available.
        if "does not exist" not in str(exc):
            raise
    finally:
        engine.dispose()

    print(f"Creating database {url.database}...")
    maintenance_database = maintenance_database_for(url)
    maintenance_engine = cli_engine(url.set(database=maintenance_database))
    try:
        # CREATE DATABASE cannot run inside a transaction block.
        with maintenance_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            try:
                conn.execute(sqlalchemy.text(f"CREATE DATABASE {url.database}"))
            except DBAPIError as exc:
                # Losing a race with a concurrent creator is not an error. Redshift is not guaranteed to report
                # Postgres' SQLSTATEs, so fall back to matching the message.
                if sqlstate_of(exc) not in {PG_DUPLICATE_DATABASE, PG_UNIQUE_VIOLATION} and "already exists" not in str(
                    exc
                ):
                    raise
    except OperationalError as exc:
        fail(f'could not create database {url.database} via the "{maintenance_database}" database: {exc}')
    finally:
        maintenance_engine.dispose()


def create_engine_and_database(url: sqlalchemy.URL, *, connect_args: dict | None = None) -> sqlalchemy.Engine:
    """Creates the database named in the URL if needed, then returns an Engine connected to it."""
    ensure_database_exists(url)
    return cli_engine(url, connect_args=connect_args)
