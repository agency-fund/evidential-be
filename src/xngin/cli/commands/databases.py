"""`create-database` and `drop-database` commands, and the database creation helper other commands share."""

import contextlib
from collections.abc import Iterator
from typing import Annotated

import psycopg
import psycopg.conninfo
import psycopg2
import psycopg2.sql
import sqlalchemy
import typer

from xngin.apiserver.dwh import dwh_utils
from xngin.cli.common import CLI_DB_APPLICATION_NAME, console, fail

# CREATE DATABASE and DROP DATABASE cannot run against the database they operate on, so they are issued against a
# database that reliably exists: "postgres" on Postgres, and "dev" on Redshift, which has no "postgres" database.
POSTGRES_MAINTENANCE_DATABASE = "postgres"
REDSHIFT_MAINTENANCE_DATABASE = "dev"

DsnOption = Annotated[
    str,
    typer.Option(
        envvar="XNGIN_LOCALPG_DSN",
        help="libpq connection URI of the Postgres server to modify. The database named in the URI is ignored; "
        f'the connection is made to the "{POSTGRES_MAINTENANCE_DATABASE}" database.',
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
            dbname=POSTGRES_MAINTENANCE_DATABASE,
        )
    except (psycopg.OperationalError, psycopg.ProgrammingError) as exc:
        fail(f"could not connect: {exc}")
    try:
        yield conn
    finally:
        conn.close()


def _create_database(conn: psycopg.Connection, name: str) -> bool:
    """Creates a database, returning False if it already existed."""
    try:
        conn.execute(t"CREATE DATABASE {name:i}")  # type: ignore[misc]
    except psycopg.errors.DuplicateDatabase:
        return False
    return True


def _create_redshift_database(url: sqlalchemy.URL, name: str) -> bool:
    """Creates a database on the Redshift cluster named in the URL, returning False if it already existed."""
    try:
        conn = psycopg2.connect(
            application_name=CLI_DB_APPLICATION_NAME,
            database=REDSHIFT_MAINTENANCE_DATABASE,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
        )
    except psycopg2.OperationalError as exc:
        fail(f"could not connect: {exc}")
    conn.autocommit = True
    with contextlib.closing(conn), conn.cursor() as cur:
        try:
            cur.execute(psycopg2.sql.SQL("CREATE DATABASE {}").format(psycopg2.sql.Identifier(name)))
        except psycopg2.errors.DuplicateDatabase:
            return False
    return True


def create_database_if_absent(url: sqlalchemy.URL) -> None:
    """Creates the database named in a SQLAlchemy URL if it does not already exist.

    Only Postgres-family targets (which includes Redshift) are supported; other dialects are left untouched, as
    their "databases" are provisioned out of band.
    """
    if not dwh_utils.is_postgres(url):
        return
    if not url.database:
        fail("the DSN must name a database.", code=2)

    if dwh_utils.is_redshift(url):
        created = _create_redshift_database(url, url.database)
    else:
        # The URL is taken apart rather than rendered back to a string because SQLAlchemy does not escape the values
        # libpq requires escaped: a space in a password would produce an unparseable URI. Query parameters (sslmode
        # and friends) are passed through, except repeated ones, which SQLAlchemy represents as tuples and libpq
        # cannot express.
        dsn = psycopg.conninfo.make_conninfo(
            host=url.host,
            port=url.port,
            user=url.username,
            password=url.password,
            **{key: value for key, value in url.query.items() if isinstance(value, str)},
        )
        with _maintenance_connection(dsn) as conn:
            created = _create_database(conn, url.database)

    if created:
        console.print(f"Created database [cyan]{url.database}[/cyan].")


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
            if _create_database(conn, name):
                console.print(f"Created database [cyan]{name}[/cyan].")
            elif allow_existing:
                console.print(f"Database [cyan]{name}[/cyan] already exists.")
            else:
                fail(f"database {name} already exists.")


def drop_database(names: NamesArgument, dsn: DsnOption):
    """Drops one or more databases, ignoring those that do not exist.

    Sessions connected to a database being dropped are terminated.
    """
    with _maintenance_connection(dsn) as conn:
        for name in names:
            try:
                conn.execute(t"DROP DATABASE {name:i} WITH (FORCE)")  # type: ignore[misc]
            except psycopg.errors.InvalidCatalogName:
                console.print(f"Database [cyan]{name}[/cyan] does not exist.")
            except psycopg.errors.ObjectInUse as exc:
                fail(f"database {name} could not be dropped: {exc}")
            else:
                console.print(f"Dropped database [cyan]{name}[/cyan].")


def register(app: typer.Typer) -> None:
    app.command()(create_database)
    app.command()(drop_database)
