from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException
from jose import jwt, JWTError
from fastapi import status

from xngin.apiserver.routers.oidc import (
    oidc_google,
    get_google_jwks,
    CLIENT_ID,
    get_google_configuration,
)
import logging
import secrets


# JWTs generated for domains other than @agency.fund are considered invalid.
ALLOWED_HOSTED_DOMAINS = ("agency.fund",)

logger = logging.getLogger(__name__)


@dataclass
class TokenInfo:
    """Information extracted from a validated OIDC token."""

    email: str
    iss: str  # issuer
    sub: str  # subject identifier
    hd: str  # hosted domain

    def is_privileged(self):
        return self.hd in ALLOWED_HOSTED_DOMAINS


# Set TESTING_TOKENS_ENABLED to allow statically-defined bearer tokens
# to skip the JWT validation.
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


async def require_oidc_token(
    token: Annotated[str, Depends(oidc_google)],
    oidc_config: Annotated[dict, Depends(get_google_configuration)],
    jwks: Annotated[dict, Depends(get_google_jwks)],
) -> TokenInfo:
    """Dependency for validating that the Authorization: header is a valid Google JWT.

    Returns:
        TokenInfo containing the validated token's claims.
    """
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
            audience=CLIENT_ID,
            issuer=oidc_config.get("issuer"),
            options={
                "require_iss": True,
                "require_aud": True,
                "require_iat": True,
                "require_exp": True,
                "verify_at_hash": False,  # PKCE flow sends at_hash but we don't need to verify it.
            },
        )
        if (
            decoded["hd"] not in ALLOWED_HOSTED_DOMAINS
            or decoded["azp"] != decoded["aud"]
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid hd"
            )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {e}",
        ) from e
    return TokenInfo(
        email=decoded["email"], iss=decoded["iss"], sub=decoded["sub"], hd=decoded["hd"]
    )
