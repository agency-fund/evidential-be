"""Configuration of the OIDC relying party, parsed from environment variables.

One identity provider (IDP) is configured at a time. Any provider implementing OpenID Connect Discovery and the
authorization code flow with PKCE can be used; provider differences are handled by spec-conformant defaults rather
than provider-specific switches.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from xngin.apiserver import flags

# Principal fields that XNGIN_OIDC_CLAIM_MAP may populate from ID token claims.
ALLOWED_CLAIM_MAP_TARGETS = frozenset({"hd"})
# The scope request parameter. openid requests an ID token, and email adds the email claim used to look up users.
SCOPE = "openid email"
DISCOVERY_PATH = "/.well-known/openid-configuration"


class OidcMisconfiguredError(Exception):
    pass


def parse_claim_map(raw: str) -> dict[str, str]:
    """Parses "claim:field,claim:field" into a mapping of ID token claim name to Principal field name."""
    claim_map: dict[str, str] = {}
    for entry in (item.strip() for item in raw.split(",")):
        if not entry:
            continue
        claim, separator, target = (part.strip() for part in entry.partition(":"))
        if not separator or not claim or not target:
            raise OidcMisconfiguredError(
                f"{flags.ENV_XNGIN_OIDC_CLAIM_MAP} entry '{entry}' is not in claim:field form."
            )
        if target not in ALLOWED_CLAIM_MAP_TARGETS:
            raise OidcMisconfiguredError(
                f"{flags.ENV_XNGIN_OIDC_CLAIM_MAP} target '{target}' is not a Principal auxiliary field. "
                f"Allowed: {', '.join(sorted(ALLOWED_CLAIM_MAP_TARGETS))}."
            )
        if target in claim_map.values():
            raise OidcMisconfiguredError(f"{flags.ENV_XNGIN_OIDC_CLAIM_MAP} maps more than one claim to '{target}'.")
        claim_map[claim] = target
    return claim_map


def check_absolute_https_url(value: str, *, allow_http: bool = False, allow_query: bool = True) -> str | None:
    """Explains why value is not an absolute https URL that is safe to use, or returns None when it is.

    Credentials and fragments are never allowed: OAuth endpoint and redirect URIs must not carry a fragment
    (RFC 6749 sections 3.1 and 3.1.2), and a URL with credentials is never something to hand to a browser.
    """
    if any(character.isspace() or ord(character) < 0x20 or character == "\x7f" for character in value):
        return "must not contain whitespace or control characters"
    parsed = urlsplit(value)
    if parsed.scheme not in (("https", "http") if allow_http else ("https",)):
        return "must be an https:// URL"
    try:
        _ = parsed.port
    except ValueError:
        return "has an invalid port"
    if not parsed.hostname:
        return "must include a hostname"
    if parsed.username is not None or parsed.password is not None:
        return "must not include credentials"
    if parsed.fragment or value.endswith("#"):
        return "must not include a fragment"
    if not allow_query and (parsed.query or value.endswith("?")):
        return "must not include a query"
    return None


def _require(value: str, env_var: str) -> str:
    value = value.strip()
    if not value:
        raise OidcMisconfiguredError(f"{env_var} environment variable is not set.")
    return value


@dataclass(frozen=True, slots=True)
class OidcSettings:
    """Describes the identity provider this server trusts."""

    issuer: str
    client_id: str
    redirect_uri: str
    # Maps ID token claim names to Principal auxiliary field names (e.g. {"hd": "hd"}).
    claim_map: Mapping[str, str] = field(default_factory=dict)
    # Unset for public clients, which authenticate with PKCE alone.
    client_secret: str | None = None

    def discovery_url(self) -> str:
        # OpenID Connect Discovery 1.0 section 4.1: a terminating slash on the issuer is removed before appending the
        # well-known path. The issuer itself is compared exactly as configured (authentik's issuers end with a slash).
        return self.issuer.rstrip("/") + DISCOVERY_PATH

    @classmethod
    def parse(
        cls,
        *,
        issuer: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        claim_map: str,
        allow_http: bool,
    ) -> OidcSettings:
        """Builds settings from raw environment variable values, raising OidcMisconfiguredError on invalid input."""
        issuer = _require(issuer, flags.ENV_XNGIN_OIDC_ISSUER)
        # OpenID Connect Discovery 1.0 section 3: the issuer is an https URL with no query or fragment.
        if reason := check_absolute_https_url(issuer, allow_http=allow_http, allow_query=False):
            raise OidcMisconfiguredError(f"{flags.ENV_XNGIN_OIDC_ISSUER} {reason} (got {issuer!r}).")
        redirect_uri = _require(redirect_uri, flags.ENV_XNGIN_OIDC_REDIRECT_URI)
        if reason := check_absolute_https_url(redirect_uri, allow_http=allow_http):
            raise OidcMisconfiguredError(f"{flags.ENV_XNGIN_OIDC_REDIRECT_URI} {reason} (got {redirect_uri!r}).")
        return cls(
            issuer=issuer,
            client_id=_require(client_id, flags.ENV_XNGIN_OIDC_CLIENT_ID),
            redirect_uri=redirect_uri,
            claim_map=parse_claim_map(claim_map),
            client_secret=client_secret.strip() or None,
        )

    @classmethod
    def from_flags(cls) -> OidcSettings:
        return cls.parse(
            issuer=flags.OIDC_ISSUER,
            client_id=flags.OIDC_CLIENT_ID,
            client_secret=flags.OIDC_CLIENT_SECRET,
            redirect_uri=flags.OIDC_REDIRECT_URI,
            claim_map=flags.OIDC_CLAIM_MAP,
            allow_http=flags.is_dev_environment(),
        )


# _settings is managed by get_oidc_settings().
_settings: OidcSettings | None = None


def get_oidc_settings() -> OidcSettings:
    """Dependency providing the OIDC settings parsed from the environment."""
    global _settings
    if _settings is None:
        _settings = OidcSettings.from_flags()
    return _settings
