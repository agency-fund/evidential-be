"""Implements an OIDC relying party for a single configured identity provider (authorization code flow with PKCE)."""

import asyncio
import datetime
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

import httpx2
import jwt
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Response
from loguru import logger
from starlette import status

from xngin.apiserver import constants, flags
from xngin.apiserver.dependencies import retrying_httpx_dependency
from xngin.apiserver.routers.auth import auth_dependencies
from xngin.apiserver.routers.auth.auth_api_types import CallbackRequest, CallbackResponse, OidcClientConfigResponse
from xngin.apiserver.routers.auth.auth_dependencies import SessionTokenCryptor
from xngin.apiserver.routers.auth.discovery import (
    OidcDiscovery,
    OidcDiscoveryClient,
    OidcProviderTimeoutError,
    OidcTokenExchangeError,
    OidcUserinfoError,
    get_oidc_discovery,
)
from xngin.apiserver.routers.auth.oidc_settings import SCOPE, OidcSettings, get_oidc_settings
from xngin.apiserver.routers.auth.principal import Principal

# The identity provider and this server may disagree slightly about the wall clock. PyJWT applies this leeway to the
# iat, nbf, and exp claims.
CLOCK_SKEW_LEEWAY = datetime.timedelta(seconds=15)

# OpenID Connect Core requires iss, sub, aud, exp, and iat in every ID token. We also require email because it is the
# key used to look up invited users. email_verified is also required, but some providers publish it only from the
# userinfo endpoint, so it is checked separately.
REQUIRED_CLAIMS = ["iss", "aud", "iat", "exp", "sub", "email"]

# Bounds the length of provider-supplied OAuth error fields and headers copied into log messages.
MAX_OAUTH_ERROR_FIELD_LENGTH = 200

# Google's OIDC issuer string, only used in informative error messages.
GOOGLE_ISSUER = "https://accounts.google.com"

OIDC_PROVIDER_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    502: {"description": "The identity provider returned an invalid or unavailable response."},
    504: {"description": "The identity provider request timed out."},
}


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
    app.state.oidc_discovery = await asyncio.to_thread(OidcDiscovery.startup, get_oidc_settings())
    try:
        yield
    finally:
        app.state.oidc_discovery.close()
        del app.state.oidc_discovery


router = APIRouter(
    lifespan=lifespan,
    prefix=constants.API_PREFIX_V1 + "/a/oidc",
)


@router.get(
    "/config",
    responses=OIDC_PROVIDER_ERROR_RESPONSES,
)
def oidc_client_config(
    settings: Annotated[OidcSettings, Depends(get_oidc_settings)],
    discovery: Annotated[OidcDiscoveryClient, Depends(get_oidc_discovery)],
) -> OidcClientConfigResponse:
    """Returns the identity provider settings the frontend needs to begin the login flow.

    The frontend builds the authorization request from these values, so the backend environment is the only place
    the identity provider is configured.
    """
    return OidcClientConfigResponse(
        authorization_endpoint=discovery.authorization_endpoint(),
        client_id=settings.client_id,
        redirect_uri=settings.redirect_uri,
        scope=SCOPE,
    )


@router.post(
    "/callback",
    responses=OIDC_PROVIDER_ERROR_RESPONSES,
)
def auth_callback(
    body: CallbackRequest,
    settings: Annotated[OidcSettings, Depends(get_oidc_settings)],
    discovery: Annotated[OidcDiscoveryClient, Depends(get_oidc_discovery)],
    httpx_client: Annotated[httpx2.Client, Depends(retrying_httpx_dependency)],
    session_cryptor: Annotated[SessionTokenCryptor, Depends()],
    response: Response,
) -> CallbackResponse:
    """Exchanges the OIDC authorization code and verifier for an identity token (JWT), and then creates a session token.

    This is the final step in acquiring a JWT from the identity provider promising that the user successfully
    authenticated. After verifying the identity token, we return a signed application-specific token that the
    frontend can use to authenticate the user for the remainder of their session.
    """
    tokens = _exchange_code_for_tokens(
        settings, discovery, httpx_client, code=body.code, code_verifier=body.code_verifier
    )
    signing_key = _get_signing_key(discovery, id_token=tokens.id_token)
    claims = _validate_idtoken(settings, signing_key, id_token=tokens.id_token, nonce=body.nonce)
    if "email_verified" not in claims:
        _require_email_verified_by_userinfo(discovery, httpx_client, claims=claims, access_token=tokens.access_token)
    session_token = session_cryptor.encode(_principal_from_claims(settings, claims))
    # The session token is a bearer credential, so no cache may keep a copy of this response.
    response.headers["Cache-Control"] = "no-store"
    return CallbackResponse(session_token=session_token)


@dataclass(frozen=True, slots=True)
class _TokenResponse:
    id_token: str
    # Used only to query the userinfo endpoint; None if the provider did not return one.
    access_token: str | None


def _exchange_code_for_tokens(
    settings: OidcSettings,
    discovery: OidcDiscoveryClient,
    httpx_client: httpx2.Client,
    *,
    code: str,
    code_verifier: str,
) -> _TokenResponse:
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
    try:
        token_response = httpx_client.post(token_endpoint, data=data)
    except httpx2.TimeoutException as exc:
        raise OidcProviderTimeoutError(f"Token request to {token_endpoint} timed out.") from exc
    except httpx2.RequestError as exc:
        raise OidcTokenExchangeError(f"Token request to {token_endpoint} failed: {type(exc).__name__}.") from exc
    if token_response.status_code != 200:
        if settings.issuer == GOOGLE_ISSUER and settings.client_secret is None:
            logger.warning(f"This OIDC IDP may require {flags.ENV_XNGIN_OIDC_CLIENT_SECRET} to be set.")
        raise OidcTokenExchangeError(
            f"Token endpoint {token_endpoint} returned status code {token_response.status_code}"
            f"{_describe_oauth_error(token_response)}."
        )
    try:
        response = token_response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OidcTokenExchangeError(f"Token endpoint {token_endpoint} returned invalid JSON.") from exc
    if not isinstance(response, dict):
        raise OidcTokenExchangeError(f"Token endpoint {token_endpoint} returned a non-dictionary response.")
    id_token = response.get("id_token")
    if not isinstance(id_token, str):
        raise OidcTokenExchangeError(f"Token endpoint {token_endpoint} did not return a string id_token.")
    access_token = response.get("access_token")
    return _TokenResponse(id_token=id_token, access_token=access_token if isinstance(access_token, str) else None)


def _describe_oauth_error(token_response: httpx2.Response) -> str:
    """Summarizes the error and error_description fields of an RFC 6749 section 5.2 error response, if present.

    Only these two fields are reported because they are intended for the client developer; the rest of the body is
    not logged.
    """
    try:
        body = token_response.json()
    except json.JSONDecodeError, UnicodeDecodeError:
        return ""
    if not isinstance(body, dict):
        return ""
    fields = [
        f"{name}={body[name][:MAX_OAUTH_ERROR_FIELD_LENGTH]!r}"
        for name in ("error", "error_description")
        if isinstance(body.get(name), str)
    ]
    return f" ({', '.join(fields)})" if fields else ""


def _get_signing_key(
    discovery: OidcDiscoveryClient,
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
        # A missing email_verified claim is looked up at the userinfo endpoint by the caller.
        if "email_verified" in claims and not _is_email_verified(claims):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Email address is not verified")
        if not isinstance(claims["email"], str) or not isinstance(claims["sub"], str):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    except jwt.PyJWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials"
        ) from e
    return claims


def _is_email_verified(claims: dict) -> bool:
    """Reports whether the email address in ID token claims or a userinfo response can be trusted.

    Users are looked up by email, so an email the identity provider has not verified cannot be trusted. Some
    providers incorrectly send the boolean as the string "true".
    """
    email_verified = claims.get("email_verified")
    return email_verified is True or email_verified == "true"


def _require_email_verified_by_userinfo(
    discovery: OidcDiscoveryClient,
    httpx_client: httpx2.Client,
    *,
    claims: dict,
    access_token: str | None,
) -> None:
    """Checks email_verified at the userinfo endpoint, for providers that omit the claim from ID tokens.

    Raises HTTPException unless the provider reports that the ID token's email address is verified.
    """
    userinfo_endpoint = discovery.userinfo_endpoint()
    if userinfo_endpoint is None or access_token is None:
        logger.warning(
            "The ID token has no email_verified claim, and it cannot be looked up because the identity provider "
            f"{'advertises no userinfo endpoint' if userinfo_endpoint is None else 'returned no access token'}."
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Email address is not verified")
    userinfo = _fetch_userinfo(httpx_client, userinfo_endpoint, access_token=access_token)
    # OpenID Connect Core 5.3.4: the response must not be used unless its sub matches the ID token's.
    if userinfo.get("sub") != claims["sub"]:
        logger.warning("The userinfo response's sub does not match the ID token's sub.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    # email_verified describes the email in the same response, so it must be the address we look users up by.
    if userinfo.get("email") != claims["email"]:
        logger.warning("The userinfo response's email does not match the ID token's email.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    if not _is_email_verified(userinfo):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Email address is not verified")


def _fetch_userinfo(httpx_client: httpx2.Client, userinfo_endpoint: str, *, access_token: str) -> dict:
    """Fetches the claims about the authenticated user from the userinfo endpoint (OpenID Connect Core 5.3)."""
    try:
        response = httpx_client.get(
            userinfo_endpoint,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        )
    except httpx2.TimeoutException as exc:
        raise OidcProviderTimeoutError(f"Userinfo request to {userinfo_endpoint} timed out.") from exc
    except httpx2.RequestError as exc:
        raise OidcUserinfoError(f"Userinfo request to {userinfo_endpoint} failed: {type(exc).__name__}.") from exc
    if response.status_code != 200:
        # RFC 6750 section 3 reports bearer token errors in the WWW-Authenticate header rather than the body.
        www_authenticate = response.headers.get("WWW-Authenticate")
        detail = f" (WWW-Authenticate={www_authenticate[:MAX_OAUTH_ERROR_FIELD_LENGTH]!r})" if www_authenticate else ""
        raise OidcUserinfoError(
            f"Userinfo endpoint {userinfo_endpoint} returned status code {response.status_code}{detail}."
        )
    content_type = response.headers.get("Content-Type", "").partition(";")[0].strip().lower()
    if content_type == "application/jwt":
        raise OidcUserinfoError(
            f"Userinfo endpoint {userinfo_endpoint} returned a signed or encrypted response, which is not supported."
        )
    try:
        userinfo = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OidcUserinfoError(f"Userinfo endpoint {userinfo_endpoint} returned invalid JSON.") from exc
    if not isinstance(userinfo, dict):
        raise OidcUserinfoError(f"Userinfo endpoint {userinfo_endpoint} returned a non-dictionary response.")
    return userinfo


def _principal_from_claims(settings: OidcSettings, claims: dict) -> Principal:
    """Builds a Principal from validated ID token claims, applying the configured claim map to auxiliary fields."""
    auxiliary: dict[str, str] = {}
    for claim, field_name in settings.claim_map.items():
        value = claims.get(claim, "")
        if not isinstance(value, str):
            logger.warning(
                f"ID token claim {claim!r} mapped to {field_name!r} is a {type(value).__name__}, not a string"
            )
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
        auxiliary[field_name] = value
    return Principal(
        email=claims["email"],
        hd=auxiliary.get("hd", ""),
        # JWT NumericDate values may be fractional; PyJWT has already verified that iat is numeric.
        iat=int(claims["iat"]),
        iss=claims["iss"],
        sub=claims["sub"],
    )
