from typing import Annotated

from pydantic import BaseModel, Field


class CallbackRequest(BaseModel):
    """Contains the OIDC authorization code and PKCE verifier for the token exchange."""

    code: str
    # Pattern specified in https://datatracker.ietf.org/doc/html/rfc7636#section-4.1
    code_verifier: Annotated[str, Field(min_length=43, max_length=128, pattern=r"^[A-Za-z0-9._~-]+$")]


class CallbackResponse(BaseModel):
    """Contains the credentials the SPA will use for interacting with the Admin API.

    This is only returned when the SPA has successfully completed the OIDC PKCE flow.
    """

    # This contains an encrypted and serialized Principal. See SessionTokenCryptor.
    session_token: Annotated[str, Field(description="Bearer token for use on Admin API endpoints.")]


# This is similar to Principal, except that Principal is an internal type and we can store sensitive information on
# it whereas CallerIdentity is designed to inform the frontend about the user's identity and privileges.
class CallerIdentity(BaseModel):
    """Describes the user's identity in a format suitable for use in the frontend."""

    email: str
    iss: str  # issuer
    sub: str  # subject identifier
    hd: str  # hosted domain
    is_privileged: bool
