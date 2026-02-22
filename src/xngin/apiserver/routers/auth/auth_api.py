"""Implements a basic Google OIDC RP."""

from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from jose import JWTError, jwt
from loguru import logger
from starlette import status

from xngin.apiserver import constants, flags
from xngin.apiserver.dependencies import retrying_httpx_dependency
from xngin.apiserver.routers.auth import auth_dependencies
from xngin.apiserver.routers.auth.auth_api_types import CallbackResponse
from xngin.apiserver.routers.auth.auth_dependencies import (
    GoogleOidcConfig,
    SessionTokenCryptor,
    get_google_configuration,
)
from xngin.apiserver.routers.auth.principal import Principal


class OidcMisconfiguredError(Exception):
    pass


def validate_environment_variables():
    """Raises informative exceptions if environment variables critical for OIDC functioning are not set."""
    if flags.AIRPLANE_MODE or auth_dependencies.TESTING_TOKENS_ENABLED:
        return

    if not flags.CLIENT_ID:
        raise OidcMisconfiguredError(f"{flags.ENV_GOOGLE_OIDC_CLIENT_ID} environment variable is not set.")
    if not flags.CLIENT_SECRET:
        logger.warning(f"{flags.ENV_GOOGLE_OIDC_CLIENT_SECRET} environment variable is not set.")


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
    code_verifier: Annotated[str, Query(min_length=43, max_length=128, pattern=r"^[A-Za-z0-9._~-]+$")],
    oidc_config: Annotated[GoogleOidcConfig, Depends(get_google_configuration)],
    httpx_client: Annotated[httpx.AsyncClient, Depends(retrying_httpx_dependency)],
    session_cryptor: Annotated[SessionTokenCryptor, Depends()],
) -> CallbackResponse:
    """Exchanges the OIDC authorization code and verifier for an identity token (JWT), and then creates a session token.

    This is the final step in acquiring a JWT from Google promising that the user successfully authenticated. After
    verifying the identity token from Google, we return a signed application-specific token that the frontend can
    use to authenticate the user for the remainder of their session.
    """
    id_token = await _exchange_code_for_idtoken(oidc_config, httpx_client, code, code_verifier)
    decoded = _validate_idtoken(oidc_config, id_token)
    session_token = session_cryptor.encode(
        Principal(
            email=decoded["email"],
            hd=decoded.get("hd", ""),  # optional claim only on Google hosted domains
            iat=decoded["iat"],
            iss=decoded["iss"],
            sub=decoded["sub"],
        )
    )
    return CallbackResponse(session_token=session_token)


async def _exchange_code_for_idtoken(
    oidc_config: GoogleOidcConfig, httpx_client: httpx.AsyncClient, code: str, code_verifier: str
):
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
    if not isinstance(response, dict):
        raise HTTPException(status_code=500, detail=f"Unexpected response from {token_endpoint}")
    id_token = response.get("id_token")
    if id_token is None or not isinstance(id_token, str):
        raise HTTPException(status_code=500, detail=f"Unexpected response from {token_endpoint}")
    return id_token


def _validate_idtoken(oidc_config: GoogleOidcConfig, id_token: str) -> dict:
    """Validates a Google ID token (JWT) and returns the claims as a Python dictionary."""
    try:
        header = jwt.get_unverified_header(id_token)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {e}",
        ) from e
    key = next((jwk for jwk in oidc_config.jwks["keys"] if jwk["kid"] == header["kid"]), None)
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unable to find appropriate key",
        )
    try:
        decoded = jwt.decode(
            id_token,
            key,
            algorithms=["RS256"],
            audience=flags.CLIENT_ID,
            issuer=oidc_config.config.get("issuer"),
            options={
                "require_iss": True,
                "require_aud": True,
                "require_iat": True,
                "require_exp": True,
                "verify_at_hash": False,  # PKCE flow sends at_hash but we don't need to verify it.
            },
        )
        # Confirming that authorized party (azp) and audience (aud) match is not strictly necessary but if Google ever
        # issues a token where azp an aud don't match then we would like to know about it.
        if decoded["azp"] != decoded["aud"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid azp/aud")
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {e}",
        ) from e
    return decoded
