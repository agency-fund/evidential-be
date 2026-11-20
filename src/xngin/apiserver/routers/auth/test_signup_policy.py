import pytest

from xngin.apiserver.limits import MAX_LENGTH_OF_DOMAIN_VALUE
from xngin.apiserver.routers.auth.signup_policy import (
    SignupPolicy,
    email_domain,
    normalize_domain,
    signup_allowed,
)
from xngin.apiserver.routers.common_enums import SignupMode


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Example.ORG ", "example.org"),
        ("@example.org", "example.org"),
        ("example.org", "example.org"),
        ("a.b.example.net", "a.b.example.net"),
        ("my-domain.example.com", "my-domain.example.com"),
        ("xn--bcher-kva.example.com", "xn--bcher-kva.example.com"),
        ("XN--BCHER-KVA.example.com", "xn--bcher-kva.example.com"),
    ],
)
def test_normalize_domain_accepts(raw, expected):
    assert normalize_domain(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        " ",
        "@",
        "example",
        "*.example.org",
        "example.org.",
        ".example.org",
        "example..org",
        "http://example.org",
        "example.org/a",
        "example.org:443",
        "a b.example.org",
        "-a.example.org",
        "a-.example.org",
        "@@example.org",
        "user@example.org",
        "bücher.example.com",
        "\u212a.example.org",  # KELVIN SIGN lowercases to an ASCII "k".
        "a" * 63 + "." + "a" * 53 + ".example.org",  # 129 characters, one more than the limit.
        "a" * 64 + ".example.org",  # A label longer than 63 characters.
    ],
)
def test_normalize_domain_rejects(raw):
    with pytest.raises(ValueError, match="Invalid domain"):
        normalize_domain(raw)


def test_normalize_domain_accepts_maximum_length():
    domain = "a" * 63 + "." + "a" * 52 + ".example.org"
    assert len(domain) == MAX_LENGTH_OF_DOMAIN_VALUE
    assert normalize_domain(domain) == domain


@pytest.mark.parametrize(
    ("email", "expected"),
    [
        ("a@example.org", "example.org"),
        ("U@EXAMPLE.org", "example.org"),
        ("a@example.org@example.com", "example.com"),
        ("a@\u212a.example.org", None),  # KELVIN SIGN lowercases to an ASCII "k".
        ("example.org", None),
        ("@example.org", None),
        ("a@", None),
        ("", None),
    ],
)
def test_email_domain(email, expected):
    assert email_domain(email) == expected


EXISTING = SignupPolicy(mode=SignupMode.EXISTING, signup_domains=frozenset({"example.org"}))
ANYONE = SignupPolicy(mode=SignupMode.ANYONE, signup_domains=frozenset())
ALLOWED = SignupPolicy(mode=SignupMode.EXISTING_AND_DOMAINS, signup_domains=frozenset({"example.org", "example.com"}))
ALLOWED_EXAMPLE_ONLY = SignupPolicy(mode=SignupMode.EXISTING_AND_DOMAINS, signup_domains=frozenset({"example.org"}))
KELVIN = SignupPolicy(mode=SignupMode.EXISTING_AND_DOMAINS, signup_domains=frozenset({"k.example.org"}))
ALLOWED_NONE = SignupPolicy(mode=SignupMode.EXISTING_AND_DOMAINS, signup_domains=frozenset())


@pytest.mark.parametrize(
    ("policy", "email", "expected"),
    [
        pytest.param(EXISTING, "a@example.org", False, id="existing-ignores-domains"),
        pytest.param(ANYONE, "a@example.org", True, id="anyone"),
        pytest.param(ANYONE, "someone@example.net", True, id="anyone-any-domain"),
        pytest.param(ANYONE, "no-at-sign", False, id="anyone-without-at"),
        pytest.param(ANYONE, "a@", False, id="anyone-without-domain"),
        pytest.param(ANYONE, "@example.org", False, id="anyone-without-local-part"),
        pytest.param(ALLOWED, "a@example.org", True, id="allowed-match"),
        pytest.param(ALLOWED, "U@EXAMPLE.org", True, id="allowed-match-case-insensitive"),
        pytest.param(ALLOWED, "a@sub.example.org", False, id="allowed-no-subdomains"),
        pytest.param(ALLOWED, "a@example.org.example.net", False, id="allowed-no-prefix-match"),
        pytest.param(ALLOWED, "a@example.org.", False, id="allowed-trailing-dot"),
        pytest.param(ALLOWED, "a@example.net", False, id="allowed-miss"),
        pytest.param(ALLOWED, "a@example.org@example.com", True, id="allowed-judged-on-last-at-allowed"),
        pytest.param(ALLOWED_EXAMPLE_ONLY, "a@example.org@example.net", False, id="allowed-judged-on-last-at-denied"),
        pytest.param(ALLOWED, "example.org", False, id="allowed-without-at"),
        pytest.param(KELVIN, "a@\u212a.example.org", False, id="allowed-non-ascii-domain"),
        pytest.param(ALLOWED_NONE, "a@example.org", False, id="allowed-empty-list"),
    ],
)
def test_signup_allowed(policy, email, expected):
    assert signup_allowed(policy, email) is expected
