from contextlib import asynccontextmanager
from typing import Annotated

import sqlalchemy
from fastapi import APIRouter, Depends, FastAPI
from loguru import logger
from sqlalchemy.orm import Session

from xngin.apiserver.dependencies import xngin_sync_db_session


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting router: {__name__} (prefix={router.prefix})")
    yield


router = APIRouter(lifespan=lifespan, prefix="/_healthchecks", dependencies=[])


@router.get("/db")
def healthcheck_db(
    session: Annotated[Session, Depends(xngin_sync_db_session)],
):
    """Endpoint to confirm that we can make a connection to the database and issue a query."""
    now = session.execute(sqlalchemy.select(sqlalchemy.sql.func.now())).scalar_one_or_none()
    return {"status": "ok", "db_time": now}
