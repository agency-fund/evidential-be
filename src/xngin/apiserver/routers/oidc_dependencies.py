import secrets
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OpenIdConnect
import httpx
from jose import JWTError, jwt
from loguru import logger
from xngin.apiserver import flags

# JWTs generated for domains other than @agency.fund are considered untrusted.
PRIVILEGED_DOMAINS = ("agency.fund",)

GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
oidc_google = OpenIdConnect(openIdConnectUrl=GOOGLE_DISCOVERY_URL)

# TODO: refresh these occasionally
google_jwks = None
google_config = None


class ServerAppearsOfflineError(Exception):
    pass


@dataclass
class TokenInfo:
    """Information extracted from a validated OIDC token."""

    email: str
    iss: str  # issuer
    sub: str  # subject identifier
    hd: str  # hosted domain

    def is_privileged(self):
        return self.hd in PRIVILEGED_DOMAINS


# Set TESTING_TOKENS_ENABLED to allow statically defined bearer tokens to skip the JWT validation.
PRIVILEGED_EMAIL = "testing@agency.fund"
PRIVILEGED_TOKEN_FOR_TESTING = secrets.token_urlsafe(32)
TESTING_TOKENS_ENABLED = False
UNPRIVILEGED_EMAIL = "testing@agencyfund.org"
UNPRIVILEGED_TOKEN_FOR_TESTING = secrets.token_urlsafe(32)
TESTING_TOKENS = {
    UNPRIVILEGED_TOKEN_FOR_TESTING: TokenInfo(
        email=UNPRIVILEGED_EMAIL, iss="testing", sub="testing", hd="agencyfund.org"
    ),
    PRIVILEGED_TOKEN_FOR_TESTING: TokenInfo(
        email=PRIVILEGED_EMAIL,
        iss="testing",
        sub="testing",
        hd="agency.fund",
    ),
}


async def get_google_configuration():
    """Fetch and cache Google's OpenID configuration"""
    global google_config
    if google_config is None:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(GOOGLE_DISCOVERY_URL)
                response.raise_for_status()
                google_config = response.json()
        except httpx.ConnectError as exc:
            raise ServerAppearsOfflineError("We appear to be offline.") from exc

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


async def require_oidc_token(
    token: Annotated[str, Security(oidc_google)],
    oidc_config: Annotated[dict, Depends(get_google_configuration)],
    jwks: Annotated[dict, Depends(get_google_jwks)],
) -> TokenInfo:
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
    try:
        header = jwt.get_unverified_header(token)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {e}",
        ) from e
    key = next((jwk for jwk in jwks["keys"] if jwk["kid"] == header["kid"]), None)
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
            issuer=oidc_config.get("issuer"),
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
    return TokenInfo(
        email=decoded["email"],
        iss=decoded["iss"],
        sub=decoded["sub"],
        hd=decoded.get("hd", ""),
    )


def setup(app):
    """Configures FastAPI dependencies for OIDC."""

    # If we are not in airplane mode, there is no setup to do.
    if not flags.AIRPLANE_MODE:
        return

    logger.warning("AIRPLANE_MODE is set.")

    def noop():
        pass

    def get_privileged_token():
        return TESTING_TOKENS[PRIVILEGED_TOKEN_FOR_TESTING]

    app.dependency_overrides[get_google_configuration] = noop
    app.dependency_overrides[get_google_jwks] = noop
    app.dependency_overrides[require_oidc_token] = get_privileged_token
