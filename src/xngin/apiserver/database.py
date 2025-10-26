"""Handles SQLAlchemy connections to the application database."""

import contextlib
import dataclasses

from loguru import logger
from sqlalchemy import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.asyncio.engine import AsyncEngine

from xngin.apiserver import flags

# SQLAlchemy's logger will append this to the name of its loggers used for the application database; e.g.
# sqlalchemy.engine.Engine.xngin_app.
SA_LOGGER_NAME_FOR_APP = "xngin_app"

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
        safe_url = make_url(with_dialect).set(password="redacted")
        logger.info(f"Using application database DSN: {safe_url}")
        return with_dialect
    raise ValueError("DATABASE_URL is not set")


@dataclasses.dataclass(slots=True, frozen=True)
class DatabaseState:
    """Contains application-wide application database connection."""

    database_url: str
    async_engine: AsyncEngine
    sessionmaker: async_sessionmaker


_GLOBAL_STATE: DatabaseState | None = None


def get_sqlalchemy_database_url():
    if _GLOBAL_STATE is None:
        raise DatabaseSetupRequiredError()
    return _GLOBAL_STATE.database_url


def get_async_engine():
    if _GLOBAL_STATE is None:
        raise DatabaseSetupRequiredError()
    return _GLOBAL_STATE.async_engine


def async_session():
    """Returns a new AsyncSession for the application database."""
    if _GLOBAL_STATE is None:
        raise DatabaseSetupRequiredError()
    return _GLOBAL_STATE.sessionmaker()


@contextlib.asynccontextmanager
async def setup():
    global _GLOBAL_STATE

    database_url = get_server_database_url()

    async_engine = create_async_engine(
        database_url, execution_options={"logging_token": "app_async"}, logging_name=SA_LOGGER_NAME_FOR_APP
    )

    # We use expire_on_commit for reasons described in docs/SQLALCHEMY.md.
    sessionmaker = async_sessionmaker(bind=async_engine, expire_on_commit=False)

    _GLOBAL_STATE = DatabaseState(database_url, async_engine, sessionmaker)
    try:
        yield
    finally:
        await async_engine.dispose()
