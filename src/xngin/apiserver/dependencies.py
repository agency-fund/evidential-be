import httpx2

from xngin.apiserver import database


class CannotFindDatasourceError(Exception):
    """Error raised when an invalid Datasource-ID is provided in a request."""


def random_seed_dependency():
    """Returns None; to be overridden by tests."""
    return


def xngin_sync_db_session():
    """Returns a synchronous database connection to the xngin app database."""
    with database.sync_session() as session:
        yield session


def retrying_httpx_dependency():
    """Returns a new httpx2 client that will retry on connection errors"""
    transport = httpx2.HTTPTransport(retries=2)
    with httpx2.Client(transport=transport, timeout=15.0) as client:
        yield client
