"""Unit tests for the SSRF guard in safe_resolve.

The session-wide ``safe_resolve_testing_mode`` fixture sets
``ALLOW_CONNECTING_TO_PRIVATE_IPS = True``; these tests override it where they
need the production safety checks.
"""

import pytest

from xngin.apiserver.dns import safe_resolve

# IPv4 multicast is never a valid unicast target.
_MULTICAST_LITERAL = "224.0.0.1"

# IPv6 is unsupported and always rejected -- including public addresses and the evasion-prone encodings of
# internal targets (NAT64 64:ff9b::/96, 6to4 2002::/16, IPv4-mapped ::ffff:0:0/96).
_IPV6_LITERALS = [
    "2606:4700:4700::1111",
    "::1",
    "::ffff:127.0.0.1",
    "64:ff9b::169.254.169.254",
    "2002:a9fe:a9fe::",
]

_IP_CLASSIFICATION_CASES = [
    ("", False),
    ("0.0.0.0", False),  # noqa: S104
    ("1.2.3.4", True),
    ("10.0.0.1", False),
    ("100.64.0.1", False),
    ("127.0.0.1", False),
    ("169.254.169.254", False),
    ("172.16.0.1", False),
    ("192.168.1.1", False),
    ("192.30.252.1", True),
    ("8.8.8.8", True),
    ("not-an-ip", False),
    (_MULTICAST_LITERAL, False),
    *((addr, False) for addr in _IPV6_LITERALS),
]


@pytest.fixture
def enforce_ip_safety(monkeypatch):
    monkeypatch.setattr(safe_resolve, "ALLOW_CONNECTING_TO_PRIVATE_IPS", False)


def _fail_lookup_v4(host: str):
    raise AssertionError(f"lookup_v4 must not be called for an IP literal: {host}")


@pytest.mark.parametrize(("addr", "expected"), _IP_CLASSIFICATION_CASES)
def test_is_safe_ip_classifies_ipv4_and_non_ip_values(enforce_ip_safety, addr, expected):
    assert safe_resolve._is_safe_ip(addr) is expected


@pytest.mark.parametrize("allow_private_ips", [False, True])
def test_safe_resolve_rejects_multicast_literals(monkeypatch, allow_private_ips):
    # ipaddress.is_global considers IPv4 multicast "global"; it is still never a valid unicast target.
    monkeypatch.setattr(safe_resolve, "ALLOW_CONNECTING_TO_PRIVATE_IPS", allow_private_ips)
    monkeypatch.setattr(safe_resolve, "lookup_v4", _fail_lookup_v4)
    with pytest.raises(safe_resolve.DnsLookupUnsafeError):
        safe_resolve.safe_resolve(_MULTICAST_LITERAL)


@pytest.mark.parametrize("addr", _IPV6_LITERALS)
def test_safe_resolve_rejects_ipv6_literals(monkeypatch, addr):
    monkeypatch.setattr(safe_resolve, "lookup_v4", _fail_lookup_v4)
    with pytest.raises(safe_resolve.DnsLookupUnsafeError):
        safe_resolve.safe_resolve(addr)


def test_safe_resolve_returns_safe_ipv4_literal_without_resolving(enforce_ip_safety, monkeypatch):
    monkeypatch.setattr(safe_resolve, "lookup_v4", _fail_lookup_v4)
    assert safe_resolve.safe_resolve("8.8.8.8") == "8.8.8.8"


def test_safe_resolve_rejects_host_without_ipv4_address(monkeypatch):
    # A host that resolves to no A records (e.g. an AAAA-only host) is rejected.
    monkeypatch.setattr(safe_resolve, "lookup_v4", lambda host: None)
    with pytest.raises(safe_resolve.DnsLookupError):
        safe_resolve.safe_resolve("ipv6only.example.com")


def test_safe_resolve_rejects_unsafe_ipv4_literal_without_resolving(enforce_ip_safety, monkeypatch):
    monkeypatch.setattr(safe_resolve, "lookup_v4", _fail_lookup_v4)
    with pytest.raises(safe_resolve.DnsLookupUnsafeError):
        safe_resolve.safe_resolve("169.254.169.254")
