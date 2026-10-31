import pytest

from xngin.apiserver import database


@pytest.fixture
def tq_dsn() -> str:
    """Converts a SQLAlchemy DSN to a Psycopg-compatible DSN."""
    return database.get_psycopg_dsn(database.get_sqlalchemy_database_url())
