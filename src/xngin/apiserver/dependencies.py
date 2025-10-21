import httpx

from xngin.apiserver import database


class CannotFindDatasourceError(Exception):
    """Error raised when an invalid Datasource-ID is provided in a request."""


def random_seed_dependency():
    """Returns None; to be overridden by tests."""
    return


async def xngin_db_session():
    """Returns a database connection to the xngin app database (not customer data warehouse)."""
    async with database.async_session() as session:
        yield session


async def retrying_httpx_dependency():
    """Returns a new httpx client that will retry on connection errors"""
    transport = httpx.AsyncHTTPTransport(retries=2)
    async with httpx.AsyncClient(transport=transport, timeout=15.0) as client:
        yield client
