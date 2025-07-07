from dataclasses import dataclass


@dataclass
class Principal:
    """Describes a currently authenticated user within the API server.

    The fields on this type should be sufficient for uniquely identifying a user using any IDP.
    """

    email: str
    iss: str  # issuer
    sub: str  # subject identifier
    hd: str  # hosted domain
