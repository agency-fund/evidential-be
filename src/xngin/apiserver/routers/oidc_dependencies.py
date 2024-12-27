from typing import Annotated

from fastapi import Depends, HTTPException
from jose import jwt, JWTError
from starlette import status

from xngin.apiserver.routers.oidc import (
    oidc_google,
    get_google_jwks,
    CLIENT_ID,
    get_google_configuration,
)
import logging

logger = logging.getLogger(__name__)


async def require_oidc_token(
    token: Annotated[str, Depends(oidc_google)],
    oidc_config: Annotated[dict, Depends(get_google_configuration)],
    jwks: Annotated[dict, Depends(get_google_jwks)],
):
    """Dependency for validating that the Authorization: header is a valid Google JWT."""
    expected_prefix = "Bearer "
    if not token.startswith(expected_prefix):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Authorization header must start with "{expected_prefix}".',
        )
    token = token[len(expected_prefix) :]
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
        return jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=CLIENT_ID,
            issuer=oidc_config.get("issuer"),
            options={
                "require_iss": True,
                "require_aud": True,
                "require_iat": True,
                "verify_at_hash": False,
            },
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {e}",
        ) from e
