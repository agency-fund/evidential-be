"""Implements a basic Admin API."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, FastAPI, Depends, Path, Body, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from xngin.apiserver.apikeys import make_key, hash_key
from xngin.apiserver.models.tables import (
    ApiKey as ApiKeyDB,
    ApiKeyDatasource as ApiKeyDatasourceDB,
)
from xngin.apiserver.dependencies import settings_dependency, xngin_db_session
from xngin.apiserver.routers.oidc_dependencies import require_oidc_token
from xngin.apiserver.settings import XnginSettings

logger = logging.getLogger(__name__)


def is_enabled():
    """Feature flag: Returns true iff OIDC is enabled."""
    return os.environ.get("ENABLE_ADMIN", "").lower() in ("true", 1)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


router = APIRouter(
    lifespan=lifespan,
    prefix="/m",
    dependencies=[
        Depends(require_oidc_token)
    ],  # All routes in this router require authentication.
)


@router.get("/caller-identity")
def caller_identity(oidc: Annotated[dict, Depends(require_oidc_token)]):
    """Returns basic metadata about the authenticated caller of this method."""
    return {
        "iat": oidc.get("iat"),
        "email": oidc.get("email"),
        "hd": oidc.get("hd"),
    }


@router.get("/datasources")
async def datasources(
    settings: Annotated[XnginSettings, Depends(settings_dependency)],
):
    """Returns a list of organizations."""
    return [
        {"id": s.id, "type": s.config.type, "secured": s.require_api_key is True}
        for s in settings.client_configs
    ]


class AdminApiBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ApiKey(AdminApiBaseModel):
    id: Annotated[str, Field(...)]
    datasource_ids: Annotated[list[str], Field(...)]
    key: Annotated[str | None, Field(...)] = None


class UpdateApiKeyRequest(AdminApiBaseModel):
    datasource_ids: Annotated[list[str], Field(...)]


@router.get("/apikeys")
async def apikeys_list(
    session: Annotated[Session, Depends(xngin_db_session)],
) -> list[ApiKey]:
    stmt = select(ApiKeyDB).options(selectinload(ApiKeyDB.datasources))
    result = session.execute(stmt)
    api_keys = result.scalars().all()
    return [
        ApiKey(
            id=api_key.id,
            datasource_ids=[ds.datasource_id for ds in api_key.datasources],
            key=None,  # Omit key in list response
        )
        for api_key in api_keys
    ]


@router.post("/apikeys")
async def apikeys_create(
    session: Annotated[Session, Depends(xngin_db_session)],
    body: Annotated[UpdateApiKeyRequest, Body(...)],
) -> ApiKey:
    """Creates an API key for the requested datasources."""
    label, key = make_key()
    key_hash = hash_key(key)
    api_key = ApiKeyDB(id=label, key=key_hash)
    api_key.datasources = [
        ApiKeyDatasourceDB(datasource_id=ds_id) for ds_id in body.datasource_ids
    ]
    session.add(api_key)
    session.commit()
    return ApiKey(id=label, datasource_ids=body.datasource_ids, key=key)


@router.delete("/apikeys/{api_key_id}")
async def apikeys_delete(
    session: Annotated[Session, Depends(xngin_db_session)],
    api_key_id: Annotated[str, Path(...)],
):
    """Deletes the specified API key."""
    stmt = delete(ApiKeyDB).where(ApiKeyDB.id == api_key_id)
    session.execute(stmt)
    session.commit()
    return {"status": "success"}


@router.patch("/apikeys/{api_key_id}")
async def apikeys_update(
    session: Annotated[Session, Depends(xngin_db_session)],
    api_key_id: Annotated[str, Path(...)],
    body: Annotated[UpdateApiKeyRequest, Body()],
) -> ApiKey:
    """Updates the list of datasources for the specified API key."""
    stmt = (
        select(ApiKeyDB)
        .options(selectinload(ApiKeyDB.datasources))
        .where(ApiKeyDB.id == api_key_id)
    )
    result = session.execute(stmt)
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    api_key.datasources = [
        ApiKeyDatasourceDB(datasource_id=ds_id) for ds_id in body.datasource_ids
    ]
    session.commit()
    return ApiKey(id=api_key_id, datasource_ids=body.datasource_ids)
