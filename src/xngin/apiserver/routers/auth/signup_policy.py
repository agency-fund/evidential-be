"""Decides whether a user without an existing users row may sign up when they first sign in."""

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from xngin.apiserver.limits import MAX_LENGTH_OF_DOMAIN_VALUE
from xngin.apiserver.routers.common_enums import SignupMode
from xngin.apiserver.sqla import tables

# A lowercase ASCII hostname with at least two labels. Each label is 1-63 characters of letters, digits and hyphens
# that does not begin or end with a hyphen; the whole name is at most MAX_LENGTH_OF_DOMAIN_VALUE characters.
# Internationalized domain names must be given in punycode (xn--...). Wildcards, trailing dots, schemes, ports and paths
# are not accepted.
_DOMAIN_RE = re.compile(
    rf"^(?=.{{1,{MAX_LENGTH_OF_DOMAIN_VALUE}}}$)([a-z0-9]([a-z0-9-]{{0,61}}[a-z0-9])?\.)+[a-z0-9]([a-z0-9-]{{0,61}}[a-z0-9])?$"
)


@dataclass(frozen=True)
class SignupPolicy:
    mode: SignupMode
    signup_domains: frozenset[str]


def load_signup_policy(session: Session) -> SignupPolicy:
    """Reads the signup policy from the database. A missing settings row means the existing mode."""
    settings = session.get(tables.ServerSettings, tables.SERVER_SETTINGS_ID)
    if settings is None:
        return SignupPolicy(mode=SignupMode.EXISTING, signup_domains=frozenset())
    mode = SignupMode(settings.signup_mode)
    domains = (
        session.scalars(select(tables.DomainSettings.domain).where(tables.DomainSettings.allow_signup))
        if mode == SignupMode.EXISTING_AND_DOMAINS
        else ()
    )
    return SignupPolicy(mode=mode, signup_domains=frozenset(domains))


def email_domain(email: str) -> str | None:
    """Returns the lowercased domain of an email address, or None if it does not have a local part and an ASCII domain.

    The domain is the part after the last "@".
    """
    local, sep, domain = email.rpartition("@")
    # Check for ASCII before lowercasing because some non-ASCII characters lowercase to ASCII ones (e.g. KELVIN SIGN).
    if not sep or not local or not domain or not domain.isascii():
        return None
    return domain.lower()


def signup_allowed(policy: SignupPolicy, email: str) -> bool:
    """Returns true when the policy permits creating a user for this email address."""
    match policy.mode:
        case SignupMode.EXISTING:
            return False
        case SignupMode.ANYONE:
            return email_domain(email) is not None
        case SignupMode.EXISTING_AND_DOMAINS:
            domain = email_domain(email)
            # Exact match only: subdomains of an allowed domain are not allowed.
            return domain is not None and domain in policy.signup_domains


def normalize_domain(raw: str) -> str:
    """Normalizes a domain for the signup allow-list, raising ValueError if it is not a valid hostname.

    Surrounding whitespace and one leading "@" are removed and the result is lowercased.
    """
    stripped = raw.strip().removeprefix("@")
    # Check for ASCII before lowercasing because some non-ASCII characters lowercase to ASCII ones (e.g. KELVIN SIGN).
    domain = stripped.lower()
    if not stripped.isascii() or not _DOMAIN_RE.fullmatch(domain):
        raise ValueError(
            f"Invalid domain {raw!r}: expected a hostname such as example.org. Internationalized domain names must be "
            "given in punycode."
        )
    return domain
