from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI
from sqlalchemy.orm.session import Session
import sqlalchemy
from xngin.apiserver.dependencies import xngin_db_session
from loguru import logger


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting router: {__name__}")
    yield


router = APIRouter(lifespan=lifespan, prefix="/_healthchecks", dependencies=[])


@router.get("/db", include_in_schema=False)
def healthcheck_db(session: Annotated[Session, Depends(xngin_db_session)]):
    """Endpoint to confirm that we can make a connection to the database and issue a query."""
    now = session.execute(
        sqlalchemy.select(sqlalchemy.sql.func.now())
    ).scalar_one_or_none()
    return {"status": "ok", "db_time": now}
