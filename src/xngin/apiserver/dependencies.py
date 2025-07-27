from typing import Annotated

import httpx
from fastapi import Depends, Header
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import constants, database
from xngin.apiserver.apikeys import require_valid_api_key
from xngin.apiserver.models import tables
from xngin.apiserver.settings import (
    Datasource,
)


class CannotFindDatasourceError(Exception):
    """Error raised when an invalid Datasource-ID is provided in a request."""


def random_seed_dependency():
    """Returns None; to be overridden by tests."""
    return


async def xngin_db_session():
    """Returns a database connection to the xngin app database (not customer data warehouse)."""
    async with database.async_session() as session:
        yield session


async def datasource_dependency(
    datasource_id: Annotated[
        str,
        Header(
            example="testing",
            alias=constants.HEADER_CONFIG_ID,
            description="The ID of the datasource to operate on.",
        ),
    ],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
    api_key: Annotated[
        str | None,
        Depends(APIKeyHeader(name=constants.HEADER_API_KEY, auto_error=False)),
    ],
):
    """Returns the configuration for the current request, as determined by the Datasource-ID HTTP request header."""
    if not datasource_id:
        # TODO: 400, replace header parameters with header model for cleaner validation
        raise CannotFindDatasourceError(f"{constants.HEADER_CONFIG_ID} is required.")

    if from_db := await xngin_session.get(tables.Datasource, datasource_id):
        await require_valid_api_key(xngin_session, api_key, datasource_id)
        dsconfig = from_db.get_config()
        return Datasource(id=datasource_id, config=dsconfig)

    raise CannotFindDatasourceError("Datasource not found.")


async def retrying_httpx_dependency():
    """Returns a new httpx client that will retry on connection errors"""
    transport = httpx.AsyncHTTPTransport(retries=2)
    async with httpx.AsyncClient(transport=transport, timeout=15.0) as client:
        yield client
