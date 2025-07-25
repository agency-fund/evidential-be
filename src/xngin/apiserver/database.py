"""Handles SQLAlchemy connections to the application database."""

import contextlib
import dataclasses
from pathlib import Path

from loguru import logger
from sqlalchemy import event
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
        database_url = (
            DEFAULT_POSTGRES_DIALECT
            + "://"
            + database_url[database_url.find("://") + 3 :]
        )
    return database_url


def get_server_database_url():
    """Gets a SQLAlchemy-compatible URL string from the environment."""
    if database_url := flags.DATABASE_URL:
        with_dialect = generic_url_to_sa_url(database_url)
        logger.info(f"Using application database DSN: {with_dialect}")
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
        database_url,
        execution_options={"logging_token": "app_async"},
        logging_name=SA_LOGGER_NAME_FOR_APP,
        echo=flags.ECHO_SQL_APP_DB,
    )

    # We use expire_on_commit for reasons described in docs/SQLALCHEMY.md.
    sessionmaker = async_sessionmaker(bind=async_engine, expire_on_commit=False)

    if flags.LOG_SQL_APP_DB:
        import inspect  # noqa: PLC0415

        @event.listens_for(
            async_engine.sync_engine, "before_cursor_execute", retval=True
        )
        def _apply_comment(
            _connection, _cursor, statement, parameters, _context, _executemany
        ):
            annotation = "unknown"
            frame = inspect.stack()
            # Find the first frame that is likely to be in our project, but skip the current frame.
            for f in frame[1:]:
                if Path(__file__).is_relative_to(Path(f.filename).parent.parent):
                    annotation = f"{f.filename}:{f.lineno}"
                    break
            statement = statement + " " + f"\n--- {annotation}"
            return statement, parameters

    _GLOBAL_STATE = DatabaseState(database_url, async_engine, sessionmaker)
    try:
        yield
    finally:
        await async_engine.dispose()
