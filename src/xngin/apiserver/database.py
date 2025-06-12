import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from xngin.apiserver import flags
from xngin.apiserver.models import tables

# SQLAlchemy's logger will append this to the name of its loggers used for the application database; e.g.
# sqlalchemy.engine.Engine.xngin_app.
SA_LOGGER_NAME_FOR_APP = "xngin_app"

DEFAULT_POSTGRES_DIALECT = "postgresql+psycopg"


def get_server_database_url():
    """Gets a SQLAlchemy-compatible URL string from the environment."""
    # Hosting providers may set hosted database URL as DATABASE_URL.
    if database_url := os.environ.get("DATABASE_URL"):
        return generic_url_to_sa_url(database_url)
    if xngin_db := os.environ.get("XNGIN_DB"):
        return xngin_db
    raise ValueError("DATABASE_URL or XNGIN_DB not set")


def generic_url_to_sa_url(database_url):
    """Converts postgres:// to a SQLAlchemy-compatible value that includes a dialect."""
    if database_url.startswith("postgres://"):
        database_url = (
            DEFAULT_POSTGRES_DIALECT + "://" + database_url[len("postgres://") :]
        )
    return database_url


SQLALCHEMY_DATABASE_URL = get_server_database_url()


engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    execution_options={"logging_token": "app"},
    logging_name=SA_LOGGER_NAME_FOR_APP,
    echo=flags.ECHO_SQL_APP_DB,
)

async_engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    execution_options={"logging_token": "app_async"},
    logging_name=SA_LOGGER_NAME_FOR_APP,
    echo=flags.ECHO_SQL_APP_DB,
)

# We use expire_on_commit for reasons described in docs/SQLALCHEMY.md.
AsyncSessionLocal = async_sessionmaker(bind=async_engine, expire_on_commit=False)


def setup():
    tables.Base.metadata.create_all(bind=engine)


if flags.LOG_SQL_APP_DB:
    import inspect

    @event.listens_for(engine, "before_cursor_execute", retval=True)
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

    @event.listens_for(async_engine.sync_engine, "before_cursor_execute", retval=True)
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
