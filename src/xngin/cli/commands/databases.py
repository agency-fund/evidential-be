"""`create-database` and `drop-database` commands."""

import contextlib
from collections.abc import Iterator
from typing import Annotated

import psycopg
import typer

from xngin.cli.common import CLI_DB_APPLICATION_NAME, MAINTENANCE_DATABASE, console, fail

DsnOption = Annotated[
    str,
    typer.Option(
        envvar="XNGIN_LOCALPG_DSN",
        help="libpq connection URI of the Postgres server to modify. The database named in the URI is ignored; "
        f'the connection is made to the "{MAINTENANCE_DATABASE}" database.',
    ),
]

NamesArgument = Annotated[list[str], typer.Argument(metavar="DBNAME...", help="Names of the databases.")]


@contextlib.contextmanager
def _maintenance_connection(dsn: str) -> Iterator[psycopg.Connection]:
    """Connects to the server's maintenance database with autocommit enabled.

    CREATE DATABASE and DROP DATABASE cannot run inside a transaction block.
    """
    try:
        conn = psycopg.connect(
            dsn,
            application_name=CLI_DB_APPLICATION_NAME,
            autocommit=True,
            dbname=MAINTENANCE_DATABASE,
        )
    except (psycopg.OperationalError, psycopg.ProgrammingError) as exc:
        fail(f"could not connect: {exc}")
    try:
        yield conn
    finally:
        conn.close()


def create_database(
    names: NamesArgument,
    dsn: DsnOption,
    allow_existing: Annotated[
        bool,
        typer.Option(help="Leave databases that already exist alone instead of failing."),
    ] = False,
):
    """Creates one or more empty databases.

    Fails if a database already exists, unless --allow-existing is passed.
    """
    with _maintenance_connection(dsn) as conn:
        for name in names:
            try:
                conn.execute(t"CREATE DATABASE {name:i}")  # type: ignore[misc]
            except psycopg.errors.DuplicateDatabase:
                if not allow_existing:
                    fail(f"database {name} already exists.")
                console.print(f"Database [cyan]{name}[/cyan] already exists.")
            else:
                console.print(f"Created database [cyan]{name}[/cyan].")


def drop_database(names: NamesArgument, dsn: DsnOption):
    """Drops one or more databases, ignoring those that do not exist.

    Sessions connected to a database being dropped are terminated.
    """
    with _maintenance_connection(dsn) as conn:
        for name in names:
            try:
                conn.execute(t"DROP DATABASE IF EXISTS {name:i} WITH (FORCE)")  # type: ignore[misc]
            except psycopg.errors.ObjectInUse as exc:
                fail(f"database {name} could not be dropped: {exc}")
            console.print(f"Dropped database [cyan]{name}[/cyan], if it existed.")


def register(app: typer.Typer) -> None:
    app.command()(create_database)
    app.command()(drop_database)
