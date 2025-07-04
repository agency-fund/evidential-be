"""Implements a basic Google OIDC RP."""

from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from loguru import logger

from xngin.apiserver import constants, flags
from xngin.apiserver.dependencies import retrying_httpx_dependency
from xngin.apiserver.routers.auth.auth_api_types import CallbackResponse
from xngin.apiserver.routers.auth.auth_dependencies import (
    GoogleOidcConfig,
    get_google_configuration,
)


class OidcMisconfiguredError(Exception):
    pass


def validate_environment_variables():
    """Raises informative exceptions if environment variables critical for OIDC functioning are not set."""
    if not flags.CLIENT_ID:
        raise OidcMisconfiguredError(
            f"{flags.ENV_GOOGLE_OIDC_CLIENT_ID} environment variable is not set."
        )
    if not flags.CLIENT_SECRET:
        logger.warning(
            f"{flags.ENV_GOOGLE_OIDC_CLIENT_SECRET} environment variable is not set."
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting router: {__name__} (prefix={router.prefix})")
    validate_environment_variables()
    yield


router = APIRouter(
    lifespan=lifespan,
    prefix=constants.API_PREFIX_V1 + "/a/oidc",
)


@router.get("/callback")
async def auth_callback(
    code: Annotated[str, Query(...)],
    code_verifier: Annotated[
        str, Query(min_length=43, max_length=128, pattern=r"^[A-Za-z0-9._~-]+$")
    ],
    oidc_config: Annotated[GoogleOidcConfig, Depends(get_google_configuration)],
    httpx_client: Annotated[httpx.AsyncClient, Depends(retrying_httpx_dependency)],
) -> CallbackResponse:
    """OAuth callback endpoint that exchanges the authorization code for tokens, and returns the id_token to the client.

    Only relevant for PKCE.
    """
    token_endpoint = oidc_config.config["token_endpoint"]
    token_response = await httpx_client.post(
        token_endpoint,
        data={
            "client_id": flags.CLIENT_ID,
            # client_secret is not strictly required by PKCE spec but Google requires it.
            "client_secret": flags.CLIENT_SECRET,
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": flags.OIDC_REDIRECT_URI,
            "grant_type": "authorization_code",
        },
    )
    if token_response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected status code from token endpoint: {token_response.status_code}",
        )
    response = token_response.json()
    if not isinstance(response, dict) or response.get("id_token") is None:
        raise HTTPException(
            status_code=500, detail=f"Unexpected response from {token_endpoint}"
        )
    return CallbackResponse(id_token=response.get("id_token"))
