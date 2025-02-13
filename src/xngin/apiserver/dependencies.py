from typing import Annotated

import httpx
from fastapi import Depends, Header
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from xngin.apiserver import constants
from xngin.apiserver.apikeys import require_valid_api_key
from xngin.apiserver.database import SessionLocal
from xngin.apiserver.gsheet_cache import GSheetCache
from xngin.apiserver.settings import (
    get_settings_for_server,
    XnginSettings,
    Datasource,
    DatasourceConfigUnion,
)
from xngin.apiserver.models.tables import Datasource as DatasourceTable


def settings_dependency():
    """Provides the settings for the server.

    This may be overridden by tests using the FastAPI dependency override features.
    """
    return get_settings_for_server()


def xngin_db_session():
    """Returns a database connection to the xngin sqlite database (not customer data warehouse)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def datasource_dependency(
    settings: Annotated[XnginSettings, Depends(settings_dependency)],
    datasource_id: Annotated[
        str, Header(example="testing", alias=constants.HEADER_CONFIG_ID)
    ],
    xngin_db: Annotated[Session, Depends(xngin_db_session)],
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
    if from_json is None and (from_db := xngin_db.get(DatasourceTable, datasource_id)):
        dsconfig = from_db.get_config()
        require_valid_api_key(xngin_db, api_key, datasource_id)
        return Datasource(id=datasource_id, config=dsconfig)

    # Datasources from the static JSON settings optionally require an API key.
    if from_json and from_json.require_api_key:
        require_valid_api_key(xngin_db, api_key, datasource_id)
    return from_json


def datasource_config_required(
    ds: Annotated[Datasource, Depends(datasource_dependency)],
) -> DatasourceConfigUnion:
    """Returns the connection-specific implementation for this datasource."""
    return ds.config


def gsheet_cache(xngin_db: Annotated[Session, Depends(xngin_db_session)]):
    return GSheetCache(xngin_db)


async def httpx_dependency():
    """Returns a new httpx client with default configuration, to be used with each request"""
    async with httpx.AsyncClient(timeout=15.0) as client:
        yield client
