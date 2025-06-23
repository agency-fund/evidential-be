from dataclasses import dataclass

# JWTs generated for domains other than @agency.fund are considered untrusted.
PRIVILEGED_DOMAINS = ("agency.fund",)


@dataclass
class Principal:
    """Describes a currently authenticated user within the API server.

    The fields on this type should be sufficient for uniquely identifying a user using any IDP.
    """

    email: str
    iss: str  # issuer
    sub: str  # subject identifier
    hd: str  # hosted domain

    def is_privileged(self):
        return self.hd in PRIVILEGED_DOMAINS
