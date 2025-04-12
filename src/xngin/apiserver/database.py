import os
import sqlite3
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.orm import sessionmaker
from xngin.apiserver import flags
from xngin.apiserver.models.tables import Base

DEFAULT_POSTGRES_DIALECT = "postgresql+psycopg"

# TODO: replace with something that looks upwards until it finds pyproject.toml.
DEFAULT_SQLITE_DB = Path(__file__).parent.parent.parent.parent / "xngin.db"


def get_server_database_url():
    """Gets a SQLAlchemy-compatible URL string from the environment."""
    # Hosting providers may set hosted database URL as DATABASE_URL.
    if database_url := os.environ.get("DATABASE_URL"):
        return generic_url_to_sa_url(database_url)
    if xngin_db := os.environ.get("XNGIN_DB"):
        return xngin_db
    return f"sqlite:///{DEFAULT_SQLITE_DB}"


def generic_url_to_sa_url(database_url):
    """Converts postgres:// to a SQLAlchemy-compatible value that includes a dialect."""
    if database_url.startswith("postgres://"):
        database_url = (
            DEFAULT_POSTGRES_DIALECT + "://" + database_url[len("postgres://") :]
        )
    return database_url


SQLALCHEMY_DATABASE_URL = get_server_database_url()


def get_connect_args():
    default = {}
    if SQLALCHEMY_DATABASE_URL.startswith("sqlite:"):
        default.update({"check_same_thread": False})
    return default


engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args=get_connect_args(),
    execution_options={"logging_token": "app"},
    logging_name="xngin_app",
    echo=flags.ECHO_SQL_APP_DB,
)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection: DBAPIConnection, _):
    if not isinstance(dbapi_connection, sqlite3.Connection):
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")  # for API key cascading deletes
    cursor.close()


SessionLocal = sessionmaker(bind=engine)


def setup():
    Base.metadata.create_all(bind=engine)
