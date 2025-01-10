"""Implements a basic Google OIDC RP."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Query, FastAPI
from fastapi.security import OpenIdConnect
from starlette.responses import FileResponse

ENV_GOOGLE_OIDC_CLIENT_ID = "GOOGLE_OIDC_CLIENT_ID"
ENV_GOOGLE_OIDC_CLIENT_SECRET = "GOOGLE_OIDC_CLIENT_SECRET"
ENV_GOOGLE_OIDC_REDIRECT_URI = "GOOGLE_OIDC_REDIRECT_URI"

CLIENT_ID = os.environ.get(ENV_GOOGLE_OIDC_CLIENT_ID)
CLIENT_SECRET = os.environ.get(ENV_GOOGLE_OIDC_CLIENT_SECRET)
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
DEFAULT_REDIRECT_URI = "http://localhost:8000/a/oidc"  # default value should match OIDC_BASE_URL in pkcetest.html
REDIRECT_URI = os.environ.get(
    ENV_GOOGLE_OIDC_REDIRECT_URI, DEFAULT_REDIRECT_URI
)  # used for testing UI only
SCOPES = ["openid", "email", "profile"]
oidc_google = OpenIdConnect(openIdConnectUrl=GOOGLE_DISCOVERY_URL)

logger = logging.getLogger(__name__)


class OidcMisconfiguredError(Exception):
    pass


def is_enabled():
    """Feature flag: Returns true iff OIDC is enabled."""
    enabled = os.environ.get("ENABLE_OIDC", "").lower() in ("true", "1")
    if enabled:
        if not os.environ.get(ENV_GOOGLE_OIDC_CLIENT_ID):
            raise OidcMisconfiguredError(
                f"{ENV_GOOGLE_OIDC_CLIENT_ID} environment variable is not set."
            )
        if not os.environ.get(ENV_GOOGLE_OIDC_CLIENT_SECRET):
            logger.warning(
                f"{ENV_GOOGLE_OIDC_CLIENT_SECRET} environment variable is not set."
            )
    return enabled


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
    return FileResponse("static/pkcetest.html")


@router.get("/login")
async def login(
    code_challenge: Annotated[str, Query(...)],
    code_challenge_method: Annotated[
        Literal["S256"],
        Query(
            ...,
        ),
    ],
):
    """Generates a login URL given a PKCE code challenge.

    Only relevant for PKCE.
    """
    config = await get_google_configuration()

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
    code: Annotated[str, Query(...)],
    code_verifier: Annotated[
        str, Query(min_length=43, max_length=128, pattern=r"^[A-Za-z0-9._~-]+$")
    ],
):
    """OAuth callback endpoint that exchanges the authorization code for tokens.

    Only relevant for PKCE.
    """
    config = await get_google_configuration()
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            config["token_endpoint"],
            data={
                "client_id": CLIENT_ID,
                # client_secret is not strictly required by PKCE spec but Google requires it.
                "client_secret": CLIENT_SECRET,
                "code": code,
                "code_verifier": code_verifier,
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )
        logger.info(f"token exchange response = {token_response.content}")
        token_response.raise_for_status()
        return token_response.json()
