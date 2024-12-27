"""Implements a basic Admin API."""

import logging
import os
import secrets
import string
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, FastAPI, Depends, Path, Body
from pydantic import BaseModel, Field

from xngin.apiserver.dependencies import settings_dependency
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

KEY_ALPHABET = [*string.ascii_lowercase, *string.ascii_uppercase, *string.digits]


def make_key():
    label = "".join([secrets.choice(KEY_ALPHABET) for _ in range(6)])
    rnd = "".join([secrets.choice(KEY_ALPHABET) for _ in range(38)])
    key = f"xat_{label}_{rnd}"
    return label, key


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
    return [{"id": s.id, "type": s.config.type} for s in settings.client_configs]


class ApiKey(BaseModel):
    id: Annotated[str, Field(...)]
    datasource_ids: Annotated[list[str], Field(...)]
    key: Annotated[str | None, Field(...)] = None


class CreateApiKeyRequest(BaseModel):
    datasource_ids: Annotated[list[str], Field(...)]


@router.get("/apikeys")
async def apikeys_list() -> list[ApiKey]:
    return [ApiKey(id=make_key()[0], datasource_ids=["a", "b", "c"]) for _ in range(10)]


@router.post("/apikeys")
async def apikeys_create(body: Annotated[CreateApiKeyRequest, Body(...)]) -> ApiKey:
    label, key = make_key()
    return ApiKey(id=label, key=key, datasource_ids=body.datasource_ids)


@router.delete("/apikeys/{api_key_id}")
async def apikeys_delete(api_key_id: Annotated[str, Path(...)]):
    return {}


@router.patch("/apikeys/{api_key_id}")
async def apikeys_update(
    api_key_id: Annotated[str, Path(...)], body: Annotated[CreateApiKeyRequest, Body()]
):
    return {}
