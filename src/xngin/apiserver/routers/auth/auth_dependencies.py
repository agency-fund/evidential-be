import asyncio
import datetime
import secrets
import time
from dataclasses import dataclass
from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.functions import count

from xngin.apiserver import flags
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.routers.auth.principal import Principal
from xngin.apiserver.routers.auth.token_crypter import TokenCrypter
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.bootstrap import create_entities_for_first_time_user
from xngin.xsecrets import chafernet

# The length of time that a session token is considered valid.
SESSION_TOKEN_LIFETIME = datetime.timedelta(hours=12).seconds
SESSION_TOKEN_LOCAL_KEYSET_FILE = ".xngin_session_token_keyset"
SESSION_TOKEN_PREFIX = "xa_"

GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# Set TESTING_TOKENS_ENABLED to allow statically defined bearer tokens to skip the JWT validation.
AIRPLANE_TOKEN = "airplane-mode-token"
PRIVILEGED_EMAIL = "testing-privileged@example.com"
PRIVILEGED_TOKEN_FOR_TESTING = secrets.token_urlsafe(32)
TESTING_TOKENS_ENABLED = False
UNPRIVILEGED_EMAIL = "testing-unprivileged@example.com"
UNPRIVILEGED_TOKEN_FOR_TESTING = secrets.token_urlsafe(32)
TESTING_TOKENS: dict[str, Principal] = {
    AIRPLANE_TOKEN: Principal(
        email="testing@example.com",
        hd="example.com",
        iat=int(time.time()),
        iss="airplane",
        sub="airplane",
    ),
    UNPRIVILEGED_TOKEN_FOR_TESTING: Principal(
        email=UNPRIVILEGED_EMAIL,
        hd="example.com",
        iat=int(time.time()),
        iss="testing",
        sub="testing",
    ),
    PRIVILEGED_TOKEN_FOR_TESTING: Principal(
        email=PRIVILEGED_EMAIL,
        hd="example.com",
        iat=int(time.time()),
        iss="testing",
        sub="testing",
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
        return self.last_refreshed < datetime.datetime.now() - datetime.timedelta(hours=1)


# _google_config and _google_config_stampede_lock are managed by get_google_configuration().
_google_config: GoogleOidcConfig | None = None
_google_config_stampede_lock = asyncio.Lock()


async def _fetch_object_200(client: httpx.AsyncClient, url: str):
    """Fetches a URL using the given httpx client, parses the response as a JSON dictionary.

    Raises GoogleOidcError when the response is not a 200 status or when the response is not a dict.
    """
    response = await client.get(url)
    if response.status_code != 200:
        raise GoogleOidcError(f"Fetching {url} failed with an unexpected status code: {response.status_code}")
    parsed = response.json()
    if not isinstance(parsed, dict):
        raise GoogleOidcError(f"{url} returned a non-dictionary response")
    return parsed


async def get_google_configuration() -> GoogleOidcConfig:
    """Dependency providing Google's OpenID configuration."""
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
                    raise GoogleOidcError("config object does not have a jwks_uri field")
                jwks_response = await _fetch_object_200(client, jwks_url)
                if not jwks_response.get("keys"):
                    raise GoogleOidcError("JWKS response does not contain keys in expected format")
                _google_config = GoogleOidcConfig(
                    last_refreshed=datetime.datetime.now(),
                    config=config,
                    jwks=jwks_response,
                )
        except httpx.ConnectError as exc:
            raise ServerAppearsOfflineError("We appear to be offline.") from exc
        else:
            return _google_config


def session_token_crypter_dependency(*, ttl: int = SESSION_TOKEN_LIFETIME):
    """Dependency that provides a configured session token crypter."""
    return TokenCrypter(
        ttl=ttl,
        keyset_env_var=flags.ENV_SESSION_TOKEN_KEYSET,
        local_keyset_filename=SESSION_TOKEN_LOCAL_KEYSET_FILE,
        prefix=SESSION_TOKEN_PREFIX,
    )


async def require_valid_session_token(
    authorization: Annotated[
        HTTPAuthorizationCredentials,
        Depends(HTTPBearer(description="Session token obtained from the auth_callback operation.")),
    ],
    tokencryptor: Annotated[TokenCrypter, Depends(session_token_crypter_dependency)],
) -> Principal:
    """Dependency for decoding the session token and retrieving a Principal.

    Returns:
        Principal containing the validated claims from the ID token.
    """
    token = authorization.credentials
    del authorization
    if principal := get_special_principal(token):
        return principal
    try:
        decrypted = tokencryptor.decrypt(token)
        return Principal.model_validate_json(decrypted)
    except chafernet.InvalidTokenError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token invalid or expired.",
        ) from err


async def require_user_from_token(
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    principal: Annotated[Principal, Depends(require_valid_session_token)],
) -> tables.User:
    """Dependency for fetching the User record matching the authenticated user's email."""
    user = await _lookup_or_create(session, principal)
    if user:
        return user
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Expired session, or user not found.",
    )


async def _lookup_or_create(session: AsyncSession, principal: Principal) -> tables.User | None:
    """Lookup or create a user based on email, iss, sub, and iat.

    To support initial deployment, a User will be created if we are in airplane mode or if there are no users in the
    database yet.
    """
    result = await session.scalars(select(tables.User).where(tables.User.email == principal.email))
    user = result.first()
    if user:
        if user.last_logout.timestamp() > principal.iat:
            return None
        if user.iss == principal.iss and user.sub == principal.sub:
            return user
        # Users that are invited but who have not yet logged in will have (iss, sub) == (None, None), meaning that they
        # can authenticate with any IDP. After logging in the first time, they will only ever be able to log in with
        # that IDP.
        if user.iss is None:
            user.iss = principal.iss
            user.sub = principal.sub
            await session.commit()
            return user
        return None

    # There are two cases when we create a user on an authenticated request:
    # 1. Airplane mode: We are in airplane mode and the request is coming from the UI in airplane mode.
    # 2. First use of installation by a developer: There are no users in the database, and this is the first request.
    user_count = await session.scalar(select(count(tables.User.id)))
    if user_count == 0 or (flags.AIRPLANE_MODE and principal.iss == "airplane"):
        user = tables.User(email=principal.email, iss=principal.iss, sub=principal.sub, is_privileged=True)
        user = await create_entities_for_first_time_user(session, user, flags.XNGIN_DEVDWH_DSN)
        await session.commit()
        return user
    return None


def enable_testing_tokens():
    """Configures the authentication system to enable tokens used in unit tests."""
    global TESTING_TOKENS_ENABLED
    TESTING_TOKENS_ENABLED = True


def get_special_principal(token: str) -> Principal | None:
    """In testing or airplane mode, specific tokens map to specific principals."""
    if TESTING_TOKENS_ENABLED and token in TESTING_TOKENS:
        return TESTING_TOKENS[token]
    if flags.AIRPLANE_MODE and token == AIRPLANE_TOKEN:
        return TESTING_TOKENS[AIRPLANE_TOKEN]
    return None


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
