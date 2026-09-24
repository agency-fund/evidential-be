"""Implements an OIDC relying party for a single configured identity provider (authorization code flow with PKCE)."""

import asyncio
import datetime
from contextlib import asynccontextmanager
from typing import Annotated

import httpx2
import jwt
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from loguru import logger
from starlette import status

from xngin.apiserver import constants, flags
from xngin.apiserver.dependencies import retrying_httpx_dependency
from xngin.apiserver.routers.auth import auth_dependencies
from xngin.apiserver.routers.auth.auth_api_types import CallbackRequest, CallbackResponse
from xngin.apiserver.routers.auth.auth_dependencies import SessionTokenCryptor
from xngin.apiserver.routers.auth.discovery import OidcDiscovery, get_oidc_discovery
from xngin.apiserver.routers.auth.oidc_settings import OidcSettings, get_oidc_settings
from xngin.apiserver.routers.auth.principal import Principal

# The identity provider and this server may disagree slightly about the wall clock. PyJWT applies this leeway to the
# iat, nbf, and exp claims.
CLOCK_SKEW_LEEWAY = datetime.timedelta(seconds=15)

# OpenID Connect Core requires iss, sub, aud, exp, and iat in every ID token. We also require email because it is the
# key used to look up invited users.
REQUIRED_CLAIMS = ["iss", "aud", "iat", "exp", "sub", "email"]


def validate_environment_variables():
    """Raises informative exceptions if environment variables critical for OIDC functioning are not set."""
    if flags.AIRPLANE_MODE or auth_dependencies.TESTING_TOKENS_ENABLED:
        return

    settings = get_oidc_settings()
    if settings.client_secret is None:
        logger.info(f"{flags.ENV_XNGIN_OIDC_CLIENT_SECRET} is not set; the token exchange will use a public client.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting router: {__name__} (prefix={router.prefix})")
    validate_environment_variables()
    if flags.AIRPLANE_MODE or auth_dependencies.TESTING_TOKENS_ENABLED:
        yield
        return

    # Construction fetches the discovery document and signing keys, so keep that blocking I/O off the event loop.
    discovery = await asyncio.to_thread(OidcDiscovery, get_oidc_settings())
    app.state.oidc_discovery = discovery
    try:
        yield
    finally:
        discovery.close()
        del app.state.oidc_discovery


router = APIRouter(
    lifespan=lifespan,
    prefix=constants.API_PREFIX_V1 + "/a/oidc",
)


@router.post("/callback")
def auth_callback(
    body: CallbackRequest,
    settings: Annotated[OidcSettings, Depends(get_oidc_settings)],
    discovery: Annotated[OidcDiscovery, Depends(get_oidc_discovery)],
    httpx_client: Annotated[httpx2.Client, Depends(retrying_httpx_dependency)],
    session_cryptor: Annotated[SessionTokenCryptor, Depends()],
) -> CallbackResponse:
    """Exchanges the OIDC authorization code and verifier for an identity token (JWT), and then creates a session token.

    This is the final step in acquiring a JWT from the identity provider promising that the user successfully
    authenticated. After verifying the identity token, we return a signed application-specific token that the
    frontend can use to authenticate the user for the remainder of their session.
    """
    id_token = _exchange_code_for_idtoken(
        settings, discovery, httpx_client, code=body.code, code_verifier=body.code_verifier
    )
    signing_key = _get_signing_key(discovery, id_token=id_token)
    claims = _validate_idtoken(settings, signing_key, id_token=id_token, nonce=body.nonce)
    session_token = session_cryptor.encode(
        Principal(
            email=claims["email"],
            hd=claims.get("hd", ""),  # optional claim only on Google hosted domains
            iat=claims["iat"],
            iss=claims["iss"],
            sub=claims["sub"],
        )
    )
    return CallbackResponse(session_token=session_token)


def _exchange_code_for_idtoken(
    settings: OidcSettings,
    discovery: OidcDiscovery,
    httpx_client: httpx2.Client,
    *,
    code: str,
    code_verifier: str,
) -> str:
    token_endpoint = discovery.token_endpoint()
    data = {
        "client_id": settings.client_id,
        "code": code,
        "code_verifier": code_verifier,
        "redirect_uri": settings.redirect_uri,
        "grant_type": "authorization_code",
    }
    if settings.client_secret is not None:
        # PKCE does not require a client secret, but Google does.
        data["client_secret"] = settings.client_secret
    token_response = httpx_client.post(token_endpoint, data=data)
    if token_response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected status code from token endpoint: {token_response.status_code}",
        )
    response = token_response.json()
    if not isinstance(response, dict):
        raise HTTPException(status_code=500, detail=f"Unexpected response from {token_endpoint}")
    id_token = response.get("id_token")
    if not isinstance(id_token, str):
        raise HTTPException(status_code=500, detail=f"Unexpected response from {token_endpoint}")
    return id_token


def _get_signing_key(
    discovery: OidcDiscovery,
    *,
    id_token: str,
) -> dict:
    """Selects the token's signing key, allowing discovery to refresh JWKS when needed."""
    try:
        header = jwt.get_unverified_header(id_token)
    except jwt.PyJWTError as exc:
        logger.warning(f"JWT header parsing failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        ) from exc
    if header.get("alg") != "RS256":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    key = discovery.get_signing_key(kid=header.get("kid"), algorithm=header.get("alg"))
    if key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unable to find appropriate key")
    return key


def _validate_idtoken(settings: OidcSettings, signing_key: dict, *, id_token: str, nonce: str) -> dict:
    """Validates an ID token (JWT) from the configured identity provider and returns the claims as a dictionary."""
    try:
        claims = jwt.decode(
            id_token,
            jwt.PyJWK(signing_key, algorithm="RS256"),
            algorithms=["RS256"],
            audience=settings.client_id,
            issuer=settings.issuer,
            leeway=CLOCK_SKEW_LEEWAY,
            # OpenID Connect Core 3.1.3.7 requires rejecting ID tokens that carry audiences we do not trust. We trust
            # no audience other than our own client ID, so we use strict mode to ensure the value is a single
            # string equal to the client ID. PyJWT's default test is containment.
            options={"require": REQUIRED_CLAIMS, "strict_aud": True},
        )
        # The authorized party (azp) is optional (OpenID Connect Core 3.1.3.7) and absent from tokens generated by
        # some providers. When present, it must be our client ID.
        if "azp" in claims and claims["azp"] != settings.client_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid azp/aud")
        if claims.get("nonce") != nonce:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid nonce")
        if not isinstance(claims["email"], str) or not isinstance(claims["sub"], str):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    except jwt.PyJWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials"
        ) from e
    return claims
