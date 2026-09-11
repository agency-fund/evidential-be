"""Handles SQLAlchemy connections to the application database."""

import contextlib
import dataclasses
import os

from loguru import logger
from sqlalchemy import Engine, create_engine, make_url
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker as sync_sessionmaker_factory

from xngin.apiserver import flags

# SQLAlchemy's logger will append this to the name of its loggers used for the application database; e.g.
# sqlalchemy.engine.Engine.xngin_app.
SA_LOGGER_NAME_FOR_APP = "xngin_app"

APP_DB_APPLICATION_NAME = f"api-{os.getpid()}"

DEFAULT_POSTGRES_DIALECT = "postgresql+psycopg"


class DatabaseSetupRequiredError(Exception):
    pass


def generic_url_to_sa_url(database_url):
    """Converts postgres:// to a SQLAlchemy-compatible value that includes a dialect."""
    if database_url.startswith(("postgres://", "postgresql://")):
        database_url = DEFAULT_POSTGRES_DIALECT + "://" + database_url[database_url.find("://") + 3 :]
    return database_url


def get_server_database_url():
    """Gets a SQLAlchemy-compatible URL string from the environment."""
    if database_url := flags.DATABASE_URL:
        with_dialect = generic_url_to_sa_url(database_url)
        safe_url = make_url(with_dialect).set(password="redacted")  # noqa: S106
        logger.info(f"Using application database DSN: {safe_url}")
        return with_dialect
    raise ValueError("DATABASE_URL is not set")


@dataclasses.dataclass(slots=True, frozen=True)
class DatabaseState:
    """Contains application-wide application database connection."""

    database_url: str
    sync_engine: Engine
    sync_sessionmaker: sync_sessionmaker_factory[Session]


_GLOBAL_STATE: DatabaseState | None = None


def get_sqlalchemy_database_url():
    if _GLOBAL_STATE is None:
        raise DatabaseSetupRequiredError()
    return _GLOBAL_STATE.database_url


def get_sync_engine():
    if _GLOBAL_STATE is None:
        raise DatabaseSetupRequiredError()
    return _GLOBAL_STATE.sync_engine


def sync_session():
    """Returns a new synchronous Session for the application database."""
    if _GLOBAL_STATE is None:
        raise DatabaseSetupRequiredError()
    return _GLOBAL_STATE.sync_sessionmaker()


@contextlib.contextmanager
def setup():
    global _GLOBAL_STATE

    database_url = get_server_database_url()

    sync_engine = create_engine(
        database_url,
        connect_args={"application_name": APP_DB_APPLICATION_NAME},
        execution_options={"logging_token": "app_sync"},
        logging_name=SA_LOGGER_NAME_FOR_APP,
    )

    # We use expire_on_commit for reasons described in docs/SQLALCHEMY.md.
    sync_sessionmaker = sync_sessionmaker_factory(bind=sync_engine, expire_on_commit=False)
    _GLOBAL_STATE = DatabaseState(database_url, sync_engine, sync_sessionmaker)
    try:
        yield
    finally:
        sync_engine.dispose()
