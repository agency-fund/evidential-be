from contextlib import asynccontextmanager
from typing import Annotated

import sqlalchemy
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import constants
from xngin.apiserver.dependencies import async_xngin_db_session, settings_dependency
from xngin.apiserver.settings import XnginSettings


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting router: {__name__} (prefix={router.prefix})")
    yield


router = APIRouter(lifespan=lifespan, prefix="/_healthchecks", dependencies=[])


@router.get("/db")
async def healthcheck_db(
    session: Annotated[AsyncSession, Depends(async_xngin_db_session)],
):
    """Endpoint to confirm that we can make a connection to the database and issue a query."""
    now = (
        await session.execute(sqlalchemy.select(sqlalchemy.sql.func.now()))
    ).scalar_one_or_none()
    return {"status": "ok", "db_time": now}


@router.get("/settings")
def debug_settings(
    request: Request,
    settings: Annotated[XnginSettings, Depends(settings_dependency)],
):
    """Endpoint for testing purposes. Returns the current server configuration and optionally the config ID."""
    # Secrets will not be returned because they are stored as SecretStrs, but nonetheless this method
    # should only be invoked from trusted IP addresses.
    if request.client is None or request.client.host not in settings.trusted_ips:
        raise HTTPException(403)
    response: dict[str, str | XnginSettings] = {"settings": settings}
    if config_id := request.headers.get(constants.HEADER_CONFIG_ID):
        response["config_id"] = config_id
    return response
