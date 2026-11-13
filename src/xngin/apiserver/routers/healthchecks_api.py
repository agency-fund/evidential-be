from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated

import sqlalchemy
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from loguru import logger
from sqlalchemy.orm import Session

from xngin.apiserver import database, flags
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


@router.get("/deadlock")
def healthcheck_deadlock(
    session: Annotated[Session, Depends(xngin_sync_db_session)],
):
    """Deadlock this request's transaction with another one, to exercise the API's handling of deadlocks.

    Available only in dev environments. Postgres notices the deadlock after its deadlock_timeout and rolls back one of
    the two transactions, raising an error that exceptionhandlers.py transforms into a 409.
    """
    if not flags.is_dev_environment():
        raise HTTPException(status_code=404)

    def lock(target: Session, key: int):
        target.execute(sqlalchemy.select(sqlalchemy.func.pg_advisory_xact_lock(0xDEAD, key)))

    with database.get_session() as other, other.begin():
        lock(session, 1)
        lock(other, 2)
        # Each transaction now waits for the other transaction to release a lock.
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(lock, other, 1)
            lock(session, 2)
            waiting.result()
    raise RuntimeError("The database did not detect the deadlock.")
