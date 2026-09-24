from pydantic import BaseModel


class Principal(BaseModel):
    """Describes a user within the API server.

    The fields on this type should be sufficient for uniquely identifying a user using any IDP.
    """

    email: str  # user email
    hd: str  # auxiliary field (e.g. Google hosted domain), populated from ID token claims via XNGIN_OIDC_CLAIM_MAP
    iat: int  # issued-at timestamp
    iss: str  # issuer
    sub: str  # subject identifier
