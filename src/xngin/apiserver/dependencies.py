from typing import Annotated

import httpx
from fastapi import Depends, Header
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from xngin.apiserver.apikeys import require_valid_api_key
from xngin.apiserver.database import SessionLocal
from xngin.apiserver.gsheet_cache import GSheetCache
from xngin.apiserver.settings import get_settings_for_server, XnginSettings


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


def config_dependency(
    settings: Annotated[XnginSettings, Depends(settings_dependency)],
    config_id: Annotated[str, Header(example="testing")],
    xngin_db: Annotated[Session, Depends(xngin_db_session)],
    api_key: Annotated[
        str | None, Depends(APIKeyHeader(name="X-API-Key", auto_error=False))
    ],
):
    """Returns the configuration for the current request, as determined by the Config-ID HTTP request header."""
    if not config_id:
        return None
    config = settings.get_client_config(config_id)
    if config and config.require_api_key:
        require_valid_api_key(xngin_db, api_key, config_id)
    return config


def gsheet_cache(xngin_db: Annotated[Session, Depends(xngin_db_session)]):
    return GSheetCache(xngin_db)


async def httpx_dependency():
    """Returns a new httpx client with default configuration, to be used with each request"""
    async with httpx.AsyncClient(timeout=15.0) as client:
        yield client
