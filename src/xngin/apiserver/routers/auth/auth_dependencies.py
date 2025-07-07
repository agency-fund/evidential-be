import asyncio
import datetime
import secrets
from dataclasses import dataclass
from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OpenIdConnect
from jose import JWTError, jwt
from loguru import logger

from xngin.apiserver import flags
from xngin.apiserver.routers.auth.principal import Principal

GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# Set TESTING_TOKENS_ENABLED to allow statically defined bearer tokens to skip the JWT validation.
AIRPLANE_TOKEN = "airplane-mode-token"
PRIVILEGED_EMAIL = "testing-privileged@example.com"
PRIVILEGED_TOKEN_FOR_TESTING = secrets.token_urlsafe(32)
TESTING_TOKENS_ENABLED = False
UNPRIVILEGED_EMAIL = "testing-unprivileged@example.com"
UNPRIVILEGED_TOKEN_FOR_TESTING = secrets.token_urlsafe(32)
TESTING_TOKENS = {
    AIRPLANE_TOKEN: Principal(
        email="testing@example.com", iss="airplane", sub="airplane", hd="example.com"
    ),
    UNPRIVILEGED_TOKEN_FOR_TESTING: Principal(
        email=UNPRIVILEGED_EMAIL, iss="testing", sub="testing", hd="example.com"
    ),
    PRIVILEGED_TOKEN_FOR_TESTING: Principal(
        email=PRIVILEGED_EMAIL,
        iss="testing",
        sub="testing",
        hd="example.com",
    ),
}


class GoogleOidcError(Exception):
    pass


class ServerAppearsOfflineError(Exception):
    pass


@dataclass
class GoogleOidcConfig:
    last_refreshed: datetime.datetime
    config: dict
    jwks: dict

    def should_refresh(self):
        return self.last_refreshed < datetime.datetime.now() - datetime.timedelta(
            hours=1
        )


# _google_config and _google_config_stampede_lock are managed by get_google_configuration().
_google_config: GoogleOidcConfig | None = None
_google_config_stampede_lock = asyncio.Lock()


async def _fetch_object_200(client: httpx.AsyncClient, url: str):
    """Fetches a URL using the given httpx client, parses the response as a JSON dictionary.

    Raises GoogleOidcError when the response is not a 200 status or when the response is not a dict.
    """
    response = await client.get(url)
    if response.status_code != 200:
        raise GoogleOidcError(
            f"Fetching {url} failed with an unexpected status code: {response.status_code}"
        )
    parsed = response.json()
    if not isinstance(parsed, dict):
        raise GoogleOidcError(f"{url} returned a non-dictionary response")
    return parsed


async def get_google_configuration() -> GoogleOidcConfig:
    """Fetch and cache Google's OpenID configuration."""
    global _google_config
    # When config is fresh, we can use it immediately.
    if _google_config and not _google_config.should_refresh():
        return _google_config

    # Send only one outbound request even if there are many waiting.
    async with _google_config_stampede_lock:
        if _google_config and not _google_config.should_refresh():
            return _google_config

        logger.info("Fetching Google OpenID configuration")
        try:
            transport = httpx.AsyncHTTPTransport(retries=2)
            async with httpx.AsyncClient(transport=transport, timeout=15.0) as client:
                config = await _fetch_object_200(client, GOOGLE_DISCOVERY_URL)
                jwks_url = config.get("jwks_uri")
                if not jwks_url:
                    raise GoogleOidcError(
                        "config object does not have a jwks_uri field"
                    )
                jwks_response = await _fetch_object_200(client, jwks_url)
                if not jwks_response.get("keys"):
                    raise GoogleOidcError(
                        "JWKS response does not contain keys in expected format"
                    )
                _google_config = GoogleOidcConfig(
                    last_refreshed=datetime.datetime.now(),
                    config=config,
                    jwks=jwks_response,
                )
        except httpx.ConnectError as exc:
            raise ServerAppearsOfflineError("We appear to be offline.") from exc
        else:
            return _google_config


async def require_oidc_token(
    token: Annotated[
        str, Security(OpenIdConnect(openIdConnectUrl=GOOGLE_DISCOVERY_URL))
    ],
    oidc_config: Annotated[GoogleOidcConfig, Depends(get_google_configuration)],
) -> Principal:
    """Dependency for validating that the Authorization: header is a valid Google JWT.

    This method may raise a 400 or 401, and the oidc_google dependency may raise a 403.

    Returns:
        TokenInfo containing the validated token's claims.
    """
    # FastAPI's OpenIDConnect helper only checks that the header exists. It does not verify that the header has the
    # expected prefix.
    expected_prefix = "Bearer "
    if not token.startswith(expected_prefix):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Authorization header must start with "{expected_prefix}".',
        )
    token = token[len(expected_prefix) :]
    if TESTING_TOKENS_ENABLED and token in TESTING_TOKENS:
        return TESTING_TOKENS[token]
    if flags.AIRPLANE_MODE and token == AIRPLANE_TOKEN:
        return TESTING_TOKENS[AIRPLANE_TOKEN]
    try:
        header = jwt.get_unverified_header(token)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {e}",
        ) from e
    key = next(
        (jwk for jwk in oidc_config.jwks["keys"] if jwk["kid"] == header["kid"]), None
    )
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unable to find appropriate key",
        )
    try:
        decoded = jwt.decode(
            token,
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
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid azp/aud"
            )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {e}",
        ) from e
    return Principal(
        email=decoded["email"],
        iss=decoded["iss"],
        sub=decoded["sub"],
        hd=decoded.get("hd", ""),
    )


def enable_testing_tokens():
    """Configures the authentication system to enable tokens used in unit tests."""
    global TESTING_TOKENS_ENABLED
    TESTING_TOKENS_ENABLED = True


def disable(app):
    """Disables interaction with internet-dependent authentication resources."""

    def noop():
        pass

    # Disable fetching of OIDC configuration data.
    app.dependency_overrides[get_google_configuration] = noop


def setup(app):
    """Configures FastAPI dependencies for OIDC."""

    # If we are not in airplane mode, there is no setup to do.
    if not flags.AIRPLANE_MODE:
        return

    logger.warning("AIRPLANE_MODE is set.")

    disable(app)
