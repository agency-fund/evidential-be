"""Implements a basic Google OIDC RP."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Query, FastAPI
from fastapi.security import OpenIdConnect
from starlette import status
from starlette.responses import FileResponse

logger = logging.getLogger(__name__)


def is_enabled():
    """Feature flag: Returns true iff OIDC is enabled."""
    return os.environ.get("ENABLE_OIDC", "").lower() in ("true", 1)


# TODO: refresh these occasionally
google_jwks = None
google_config = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await asyncio.gather(get_google_jwks())
    yield


router = APIRouter(
    lifespan=lifespan,
    prefix="/a/oidc",
)


GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
oidc_google = OpenIdConnect(openIdConnectUrl=GOOGLE_DISCOVERY_URL)
CLIENT_ID = os.environ.get("GOOGLE_OIDC_CLIENT_ID")
CLIENT_SECRET = os.environ.get("GOOGLE_OIDC_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8000/a/oidc"  # used for testing UI only
SCOPES = ["openid", "email", "profile"]


async def get_google_configuration():
    """Fetch and cache Google's OpenID configuration"""
    global google_config
    if google_config is None:
        async with httpx.AsyncClient() as client:
            response = await client.get(GOOGLE_DISCOVERY_URL)
            response.raise_for_status()
            google_config = response.json()
    return google_config


async def get_google_jwks():
    """Fetch and cache Google's JWKS"""
    global google_jwks
    if google_jwks is None:
        config = await get_google_configuration()
        async with httpx.AsyncClient() as client:
            response = await client.get(config["jwks_uri"])
            response.raise_for_status()
            google_jwks = response.json()
    return google_jwks


# TODO: remove this once integration is confirmed to work
@router.get("/")
def index():
    return FileResponse("static/oidctest.html")


@router.get("/login")
async def login(
    code_challenge: Annotated[str, Query(...)],
    code_challenge_method: Annotated[str, Query(...)],
):
    """Generates a login URL given a PKCE code challenge."""
    config = await get_google_configuration()
    if code_challenge_method not in {"S256"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only S256 PKCE method is supported",
        )

    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
    }

    return {"auth_url": f"{config['authorization_endpoint']}?{urlencode(params)}"}


@router.get("/callback")
async def auth_callback(
    code: Annotated[str, Query(...)], code_verifier: Annotated[str, Query(...)]
):
    """OAuth callback endpoint that exchanges the authorization code for tokens."""
    config = await get_google_configuration()
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            config["token_endpoint"],
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "code": code,
                "code_verifier": code_verifier,  # Include PKCE verifier
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )
        token_response.raise_for_status()
        return token_response.json()
