from typing import Annotated

import httpx
from fastapi import Depends, Header
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import constants
from xngin.apiserver.apikeys import require_valid_api_key
from xngin.apiserver.database import AsyncSessionLocal
from xngin.apiserver.models import tables
from xngin.apiserver.settings import (
    Datasource,
    DatasourceConfig,
    XnginSettings,
    get_settings_for_server,
)


class CannotFindDatasourceError(Exception):
    """Error raised when an invalid Datasource-ID is provided in a request."""


def random_seed_dependency():
    """Returns None; to be overridden by tests."""
    return


def settings_dependency():
    """Provides the settings for the server.

    This may be overridden by tests using the FastAPI dependency override features.
    """
    return get_settings_for_server()


async def xngin_db_session():
    """Returns a database connection to the xngin app database (not customer data warehouse)."""
    async with AsyncSessionLocal() as session:
        yield session


async def datasource_dependency(
    settings: Annotated[XnginSettings, Depends(settings_dependency)],
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
        return None

    # Datasource configs can be in the static JSON settings or in the database.
    from_json = settings.get_datasource(datasource_id)

    # Datasources from the database always require an API key.
    if from_json is None and (
        from_db := await xngin_session.get(tables.Datasource, datasource_id)
    ):
        await require_valid_api_key(xngin_session, api_key, datasource_id)
        dsconfig = from_db.get_config()
        return Datasource(id=datasource_id, config=dsconfig)

    # Datasources from the static JSON settings optionally require an API key.
    if from_json and from_json.require_api_key:
        await require_valid_api_key(xngin_session, api_key, datasource_id)

    if from_json is None:
        raise CannotFindDatasourceError("Invalid datasource.")

    return from_json


def datasource_config_required(
    ds: Annotated[Datasource | None, Depends(datasource_dependency)],
) -> DatasourceConfig:
    """Returns the connection-specific implementation for this datasource."""
    if ds is None:
        raise CannotFindDatasourceError("Invalid datasource.")
    return ds.config


async def retrying_httpx_dependency():
    """Returns a new httpx client that will retry on connection errors"""
    transport = httpx.AsyncHTTPTransport(retries=2)
    async with httpx.AsyncClient(transport=transport, timeout=15.0) as client:
        yield client
