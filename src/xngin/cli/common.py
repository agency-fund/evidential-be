"""Helpers shared by `cli/main.py` and the per-command modules under `cli/commands/`."""

import os
import stat
from pathlib import Path

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError

from xngin.apiserver.dwh import dwh_utils

SA_LOGGER_NAME_FOR_CLI = "cli_dwh"

OWNER_ONLY_MODE = stat.S_IRUSR | stat.S_IWUSR


def write_file_atomically(path: Path, content: str, *, private: bool = False) -> None:
    """Writes content to path atomically.

    Pass private=True for files holding secrets: the file is restricted to the current user, as if it had been
    created and then `chmod og=`ed. This only has effect on Unix-like filesystems.
    """
    if path.exists() and not path.is_file():
        # Character devices like /dev/null can't be renamed over, and atomicity means nothing for them.
        path.write_text(content)
        return

    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        if private:
            temp_path.touch(mode=OWNER_ONLY_MODE)
            temp_path.chmod(OWNER_ONLY_MODE)
        temp_path.write_text(content)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def create_engine_and_database(url: sqlalchemy.URL, *, connect_args: dict | None = None):
    """Connects to a SQLAlchemy URL and creates the database if it doesn't exist.

    Only implemented for psycopg/psycopg2.
    """
    import psycopg.errors  # noqa: PLC0415
    import psycopg2.errors  # noqa: PLC0415

    connect_args = connect_args or {}

    try:
        engine = create_engine(url, connect_args=connect_args, logging_name=SA_LOGGER_NAME_FOR_CLI)
        dwh_utils.extra_engine_setup(engine)
        with engine.connect():
            print("Connected.")
    except OperationalError as exc:
        if not dwh_utils.is_postgres(url) or (
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
        dwh_utils.extra_engine_setup(engine)
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            # Creating a database can fail in many ways.
            try:
                conn.execute(sqlalchemy.text(f"CREATE DATABASE {url.database}"))
            except psycopg.errors.DuplicateDatabase, psycopg2.errors.DuplicateDatabase:
                pass
            except psycopg.errors.IntegrityError, psycopg2.errors.IntegrityError:
                pass
            except ProgrammingError as exc:
                if "already exists" not in str(exc):
                    raise
            except IntegrityError as exc:
                if "pg_database_datname_index" not in str(exc):
                    raise
        return create_engine(url, connect_args=connect_args, logging_name=SA_LOGGER_NAME_FOR_CLI)
    else:
        return engine
