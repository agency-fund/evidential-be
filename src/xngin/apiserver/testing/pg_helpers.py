"""Creates and drops the Postgres databases used by tests.

Each test run and pytest-xdist worker gets its own databases, named "<configured name>_<checkout>_p<run pid>_<worker>"
(e.g. test_xngin_1a2b_p4242_gw0), so that concurrent runs, including runs in sibling checkouts sharing one Postgres
instance, never touch each other's databases. A run drops its databases when it passes and leaves them behind for
inspection when it fails; the next run in the same checkout drops them.
"""

import hashlib
import os
import re
from pathlib import Path

import psycopg as psycopg3
import sqlalchemy
from sqlalchemy import MetaData, create_mock_engine
from sqlalchemy.dialects.postgresql import psycopg

# Arbitrary key for the advisory lock that serializes creating and dropping test databases, which may race between
# pytest-xdist workers and between checkouts sharing one Postgres instance.
_LOCK_KEY = 0x786E67696E

# Identifies this checkout, so that a run only ever cleans up after earlier runs in the same checkout.
_CHECKOUT_ID = hashlib.sha256(str(Path(__file__).resolve().parents[4]).encode()).hexdigest()[:4]


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


def _management_dsn(connect_url: sqlalchemy.URL) -> str:
    """Returns a psycopg3-compatible DSN for the server's always-present postgres database."""
    return connect_url.set(database="postgres", drivername="postgres").render_as_string(hide_password=False)


def finish_test_databases_pg(connect_url: str, *, keep: bool) -> list[str]:
    """Drops this process's test databases unless keep is true, and returns their names.

    connect_url may name any database on the server.
    """
    url = sqlalchemy.engine.url.make_url(connect_url)
    with psycopg3.connect(_management_dsn(url), autocommit=True) as conn:
        rows = conn.execute(
            "SELECT datname FROM pg_database WHERE right(datname, %s) = %s ORDER BY datname",
            (
                len(_database_suffix()),
                _database_suffix(),
            ),
        ).fetchall()
        names = [name for (name,) in rows]
        if not keep:
            for name in names:
                conn.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
    return names


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # It exists but belongs to someone else.
    return True


def _drop_databases_of_finished_runs(conn: psycopg3.Connection):
    """Drops this checkout's test databases whose run has exited, such as those a failed run left behind."""
    pattern = re.compile(rf"_{_CHECKOUT_ID}_p(\d+)_(main|gw\d+)$")
    for (name,) in conn.execute("SELECT datname FROM pg_database").fetchall():
        if (match := pattern.search(name)) and not _is_running(int(match[1])):
            conn.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


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
    by checkouts whose schemas match. Databases left behind by earlier, finished runs in this checkout are dropped.
    """
    url = sqlalchemy.engine.url.make_url(connect_url)
    template = f"xngin_tmpl_{_schema_digest(metadata)}"
    with psycopg3.connect(_management_dsn(url), autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
        try:
            _drop_databases_of_finished_runs(conn)
            row = conn.execute("SELECT datistemplate FROM pg_database WHERE datname = %s", (template,)).fetchone()
            if row is None or not row[0]:
                # A template that exists but was never marked as one is left over from an interrupted build.
                conn.execute(f"DROP DATABASE IF EXISTS {template} WITH (FORCE)")
                conn.execute(f"CREATE DATABASE {template}")
                engine = sqlalchemy.create_engine(url.set(database=template), poolclass=sqlalchemy.NullPool)
                try:
                    with engine.begin() as template_conn:
                        metadata.create_all(template_conn)
                finally:
                    engine.dispose()
                conn.execute(f"ALTER DATABASE {template} WITH IS_TEMPLATE true")
            conn.execute(f"DROP DATABASE IF EXISTS {url.database} WITH (FORCE)")
            conn.execute(f"CREATE DATABASE {url.database} TEMPLATE {template} STRATEGY FILE_COPY")
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
