"""Creates and drops the Postgres databases used by tests.

Each test run and pytest-xdist worker gets its own databases, named "<configured name>_<started>_<run>_<worker>" (e.g.
test_xngin_20261022t153000_5e6f7a8b_gw0, started at 15:30:00 UTC), so that concurrent runs, including runs in other
checkouts or on other machines sharing one Postgres instance, never touch each other's databases. A process drops the
databases it created when its tests pass and leaves them behind for inspection when they fail.

Whatever a run leaves behind, including the databases of a pytest-xdist worker that crashed, is dropped a day after the
run started by whichever run next creates an application database (task drop-databases drops everything sooner).
No test run lasts a day, so a database that old is no longer in use.

See xngin.apiserver.testing.database_plugin for the pytest hooks that use this.
"""

import hashlib
import os
import re
import secrets
from datetime import UTC, datetime, timedelta

import psycopg as psycopg3
import sqlalchemy
from sqlalchemy import MetaData, create_mock_engine
from sqlalchemy.dialects.postgresql import psycopg

# Arbitrary key for the advisory lock that serializes creating and dropping test databases, which may race between
# pytest-xdist workers and between checkouts sharing one Postgres instance.
_LOCK_KEY = 0x786E67696E

TEMPLATE_PREFIX = "xngin_tmpl_"

# Each schema gets its own template, named after a hash of its DDL, so a branch with a new migration gets a new one.
# Besides the current schema's template, keep this many of the newest others, so that switching between branches, or
# running checkouts of different branches against one Postgres instance, reuses their templates instead of rebuilding
# them. Older templates are dropped so that they don't pile up.
_OTHER_TEMPLATES_KEPT = 4

# How long after a run starts its leftover databases are dropped.
_LEFTOVER_MAX_AGE = timedelta(days=1)

# Lowercase, as Postgres folds unquoted names to lowercase: the names can be typed in SQL without quoting them.
_STARTED_FORMAT = "%Y%m%dt%H%M%S"

# pytest-xdist gives every worker of a run the same random id; a run without pytest-xdist makes up its own.
_RUN_ID = os.environ.get("PYTEST_XDIST_TESTRUNUID", "")[:8] or secrets.token_hex(4)

# Makes this process's test database names unique to its run and worker, and records when it started.
_DATABASE_SUFFIX = (
    f"_{datetime.now(UTC).strftime(_STARTED_FORMAT)}_{_RUN_ID}_{os.environ.get('PYTEST_XDIST_WORKER', 'main')}"
)

# Matches the names of test databases, which end with a _DATABASE_SUFFIX, capturing when their process started.
_TEST_DATABASE_NAME = re.compile(r"_(?P<started>\d{8}t\d{6})_[0-9a-f]{8}_(main|gw\d+)$")

# The URLs of the databases this process created, which it drops when its tests pass.
_CREATED: dict[str, sqlalchemy.URL] = {}


def with_name_suffix(name: str) -> str:
    """Returns name suffixed to be unique to this run and worker, for objects that runs and workers share a server for.

    The testing warehouse is one database for every run and worker, so schemas and roles that tests create in it need
    this suffix, as test databases do.
    """
    return f"{name}{_DATABASE_SUFFIX}"


def with_database_suffix(connect_url: str) -> str:
    """Returns connect_url with its database name suffixed to be unique to this run and worker."""
    url = sqlalchemy.engine.url.make_url(connect_url)
    return url.set(database=with_name_suffix(f"{url.database}")).render_as_string(hide_password=False)


def _server_dsn(url: sqlalchemy.URL) -> str:
    """Returns the DSN of the server's always-present postgres database, from which databases are made and dropped."""
    return url.set(database="postgres", drivername="postgres").render_as_string(hide_password=False)


def _connect_to_server(url: sqlalchemy.URL) -> psycopg3.Connection:
    return psycopg3.connect(_server_dsn(url), autocommit=True)


def _check_is_test_database(database: str):
    """Refuses to (re)create a database whose name lacks this process's suffix, which would drop someone's data."""
    if not database.endswith(_DATABASE_SUFFIX):
        raise ValueError(f"Refusing to recreate {database!r}: test database names must end with {_DATABASE_SUFFIX!r}.")


def _drop_database(conn: psycopg3.Connection, name: str):
    conn.execute(t"DROP DATABASE IF EXISTS {name:i} WITH (FORCE)")  # type: ignore[misc]


def _drop_template(conn: psycopg3.Connection, name: str):
    """Drops a template database, which Postgres requires to be unmarked as a template first."""
    conn.execute(t"ALTER DATABASE {name:i} IS_TEMPLATE false")  # type: ignore[misc]
    _drop_database(conn, name)


def _drop_leftovers(conn: psycopg3.Connection, current_template: str, now: datetime):
    """Drops test databases older than _LEFTOVER_MAX_AGE before now, and all but the
    newest templates for other schemas.
    """
    # Newest first: database OIDs are assigned in creation order.
    rows = conn.execute("SELECT datname FROM pg_database ORDER BY oid DESC").fetchall()
    other_templates = [name for (name,) in rows if name.startswith(TEMPLATE_PREFIX) and name != current_template]
    # Unlike test databases, which are dropped a day after their run started, templates are reused across runs, so they
    # are dropped by count instead: all but the newest few, whatever their age.
    for name in other_templates[_OTHER_TEMPLATES_KEPT:]:
        _drop_template(conn, name)
    for (name,) in rows:
        if (match := _TEST_DATABASE_NAME.search(name)) and (
            now - datetime.strptime(match["started"], _STARTED_FORMAT).replace(tzinfo=UTC) > _LEFTOVER_MAX_AGE
        ):
            _drop_database(conn, name)


def _schema_digest(metadata: MetaData) -> str:
    """Hashes the DDL that metadata.create_all() emits on Postgres, so templates are rebuilt when the schema changes."""
    statements: list[str] = []
    dialect = psycopg.dialect()
    engine = create_mock_engine(
        "postgresql+psycopg://",
        lambda sql, *_args, **_kwargs: statements.append(str(sql.compile(dialect=dialect))),
    )
    metadata.create_all(engine, checkfirst=False)
    # Sorted because create_all() emits each table's indexes in set order, which can vary between processes.
    return hashlib.sha256("\n".join(sorted(statements)).encode()).hexdigest()[:12]


def create_test_database_pg(connect_url: str, metadata: MetaData):
    """Creates the database named by connect_url as an empty copy of metadata's schema.

    The schema is built once into a template database named after a hash of its DDL and then cloned, which is much
    faster than running create_all() for every test session. The template is shared by every pytest-xdist worker and
    by checkouts whose schemas match. Leftovers from earlier runs are dropped first; see _drop_leftovers().

    The advisory lock serializing this is released when the connection closes.
    """
    url = sqlalchemy.engine.url.make_url(connect_url)
    database = url.database or ""
    _check_is_test_database(database)
    template = f"{TEMPLATE_PREFIX}{_schema_digest(metadata)}"
    with _connect_to_server(url) as conn:
        conn.execute(t"SELECT pg_advisory_lock({_LOCK_KEY})")  # type: ignore[misc]
        _drop_leftovers(conn, template, datetime.now(UTC))
        row = conn.execute(t"SELECT datistemplate FROM pg_database WHERE datname = {template}").fetchone()  # type: ignore[misc]
        if row is None or not row[0]:
            # A template that exists but was never marked as one is left over from an interrupted build.
            _drop_database(conn, template)
            conn.execute(t"CREATE DATABASE {template:i}")  # type: ignore[misc]
            engine = sqlalchemy.create_engine(url.set(database=template), poolclass=sqlalchemy.NullPool)
            try:
                with engine.begin() as template_conn:
                    metadata.create_all(template_conn)
            finally:
                engine.dispose()
            conn.execute(t"ALTER DATABASE {template:i} IS_TEMPLATE true")  # type: ignore[misc]
        _drop_database(conn, database)
        conn.execute(t"CREATE DATABASE {database:i} TEMPLATE {template:i} STRATEGY FILE_COPY")  # type: ignore[misc]
        _CREATED[database] = url


def create_empty_test_database_pg(connect_url: sqlalchemy.URL | str):
    """(Re)creates the empty database named by connect_url."""
    url = sqlalchemy.engine.url.make_url(connect_url)
    database = url.database or ""
    _check_is_test_database(database)
    with _connect_to_server(url) as conn:
        _drop_database(conn, database)
        conn.execute(t"CREATE DATABASE {database:i}")  # type: ignore[misc]
    _CREATED[database] = url


def finish_test_databases_pg(*, keep: bool) -> list[str]:
    """Drops the databases this process created unless keep is true, and returns their names."""
    names = sorted(_CREATED)
    if not keep:
        by_server: dict[str, list[str]] = {}
        for name, url in _CREATED.items():
            by_server.setdefault(_server_dsn(url), []).append(name)
        for dsn, server_names in by_server.items():
            with psycopg3.connect(dsn, autocommit=True) as conn, conn.pipeline():
                for name in sorted(server_names):
                    _drop_database(conn, name)
        _CREATED.clear()
    return names
