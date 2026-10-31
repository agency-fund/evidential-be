import pytest
from sqlalchemy import make_url

from xngin.apiserver import database


@pytest.fixture
def tq_dsn() -> str:
    """Converts a SQLAlchemy DSN to a Psycopg-compatible DSN."""
    url = make_url(database.get_sqlalchemy_database_url()).set(drivername="postgresql")
    return url.render_as_string(hide_password=False)
