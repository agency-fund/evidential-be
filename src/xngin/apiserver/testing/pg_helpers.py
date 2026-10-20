"""Creates and drops the Postgres databases used by tests.

Each test run and pytest-xdist worker gets its own databases, named "<configured name>_<checkout>_p<run pid>_<worker>"
(e.g. test_xngin_1a2b_p4242_gw0), so that concurrent runs, including runs in other checkouts or on other machines
sharing one Postgres instance, never touch each other's databases. A process drops the databases it created when its
tests pass and leaves them behind for inspection when they fail. Whatever a run leaves behind, including the
databases of a pytest-xdist worker that crashed, is dropped by the next run of the same checkout on the same machine.

See xngin.apiserver.testing.database_plugin for the pytest hooks that use this.
"""

import hashlib
import os
import re
import socket
from pathlib import Path

import machineid
import psycopg as psycopg3
import sqlalchemy
from sqlalchemy import MetaData, create_mock_engine
from sqlalchemy.dialects.postgresql import psycopg

# Arbitrary key for the advisory lock that serializes creating and dropping test databases, which may race between
# pytest-xdist workers and between checkouts sharing one Postgres instance.
_LOCK_KEY = 0x786E67696E

TEMPLATE_PREFIX = "xngin_tmpl_"


def _machine_id() -> str:
    try:
        return machineid.id()
    except machineid.MachineIdNotFound:
        return socket.gethostname()


# Identifies this checkout on this machine. A run only cleans up after earlier runs with the same id, because the pids
# in their database names can only be checked for liveness on the machine that ran them.
_CHECKOUT_ID = hashlib.sha256(f"{_machine_id()}:{Path(__file__).resolve().parents[4]}".encode()).hexdigest()[:4]

# The URLs of the databases this process created, which it drops when its tests pass.
_CREATED: dict[str, sqlalchemy.URL] = {}


def _database_suffix() -> str:
    """Returns the suffix that makes this process's test database names unique to its checkout, run, and worker."""
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    # The pid of the outermost pytest process identifies the run: it is the parent of every pytest-xdist worker.
    run_pid = os.getppid() if worker else os.getpid()
    return f"_{_CHECKOUT_ID}_p{run_pid}_{worker or 'main'}"


def with_database_suffix(connect_url: str) -> str:
    """Returns connect_url with its database name suffixed to be unique to this checkout, run, and worker."""
    url = sqlalchemy.engine.url.make_url(connect_url)
    return url.set(database=f"{url.database}{_database_suffix()}").render_as_string(hide_password=False)


def _connect_to_server(url: sqlalchemy.URL) -> psycopg3.Connection:
    """Connects to the server's always-present postgres database, from which databases can be created and dropped."""
    dsn = url.set(database="postgres", drivername="postgres").render_as_string(hide_password=False)
    return psycopg3.connect(dsn, autocommit=True)


def _drop_database(conn: psycopg3.Connection, name: str):
    conn.execute(t"DROP DATABASE IF EXISTS {name:i} WITH (FORCE)")  # type: ignore[misc]


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # It exists but belongs to someone else.
    return True


def _drop_leftovers(conn: psycopg3.Connection, current_template: str):
    """Drops this checkout's test databases whose run has exited, and templates for schemas other than the current one.

    Rebuilding a template takes about a second, so a checkout using another schema just rebuilds its own next time.
    """
    pattern = re.compile(rf"_{_CHECKOUT_ID}_p(\d+)_(main|gw\d+)$")
    for (name,) in conn.execute("SELECT datname FROM pg_database").fetchall():
        if name.startswith(TEMPLATE_PREFIX) and name != current_template:
            conn.execute(t"ALTER DATABASE {name:i} IS_TEMPLATE false")  # type: ignore[misc]
            _drop_database(conn, name)
        elif (match := pattern.search(name)) and not _is_running(int(match[1])):
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
    """
    url = sqlalchemy.engine.url.make_url(connect_url)
    database = url.database or ""
    template = f"{TEMPLATE_PREFIX}{_schema_digest(metadata)}"
    with _connect_to_server(url) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
        try:
            _drop_leftovers(conn, template)
            row = conn.execute("SELECT datistemplate FROM pg_database WHERE datname = %s", (template,)).fetchone()
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
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))


def create_empty_test_database_pg(connect_url: sqlalchemy.URL | str):
    """(Re)creates the empty database named by connect_url."""
    url = sqlalchemy.engine.url.make_url(connect_url)
    database = url.database or ""
    with _connect_to_server(url) as conn:
        _drop_database(conn, database)
        conn.execute(t"CREATE DATABASE {database:i}")  # type: ignore[misc]
    _CREATED[database] = url


def finish_test_databases_pg(*, keep: bool) -> list[str]:
    """Drops the databases this process created unless keep is true, and returns their names."""
    names = sorted(_CREATED)
    if not keep:
        for name, url in sorted(_CREATED.items()):
            with _connect_to_server(url) as conn:
                _drop_database(conn, name)
        _CREATED.clear()
    return names
