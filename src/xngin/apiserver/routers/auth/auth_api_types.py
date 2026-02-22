from typing import Annotated

from pydantic import BaseModel, Field


class CallbackResponse(BaseModel):
    """Contains the credentials the SPA will use for interacting with the Admin API.

    This is only returned when the SPA has successfully completed the OIDC PKCE flow.
    """

    # This contains an encrypted and serialized Principal. See TokenCryptor.
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
