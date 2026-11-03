import pytest

from xngin.apiserver import database


@pytest.mark.parametrize(
    "database_url",
    [
        "postgres://u:p@host:5432/db?sslmode=require",
        "postgresql://u:p@host:5432/db?sslmode=require",
        "postgresql+psycopg://u:p@host:5432/db?sslmode=require",
    ],
)
def test_get_psycopg_dsn(database_url):
    assert database.get_psycopg_dsn(database_url) == "postgresql://u:p@host:5432/db?sslmode=require"
