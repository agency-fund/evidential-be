"""Unit tests for the SSRF guard in safe_resolve.

The session-wide ``safe_resolve_testing_mode`` fixture sets
``ALLOW_CONNECTING_TO_PRIVATE_IPS = True``; these tests flip it back off via the
``enforce_ip_safety`` fixture so the real safety checks run.
"""

import pytest

from xngin.apiserver.dns import safe_resolve


@pytest.fixture
def enforce_ip_safety(monkeypatch):
    monkeypatch.setattr(safe_resolve, "ALLOW_CONNECTING_TO_PRIVATE_IPS", False)


def test_is_safe_ip_allows_public_ipv4(enforce_ip_safety):
    assert safe_resolve.is_safe_ip("8.8.8.8")
    assert safe_resolve.is_safe_ip("1.2.3.4")
    # Regression: globally-routable 192.x addresses (e.g. github.com's range) must not be blocked.
    assert safe_resolve.is_safe_ip("192.30.252.1")


def test_is_safe_ip_blocks_private_and_special_ipv4(enforce_ip_safety):
    unsafe = ["127.0.0.1", "10.0.0.1", "192.168.1.1", "172.16.0.1", "169.254.169.254", "0.0.0.0", "100.64.0.1"]  # noqa: S104
    for addr in unsafe:
        assert not safe_resolve.is_safe_ip(addr), addr


def test_is_safe_ip_rejects_non_ip_strings(enforce_ip_safety):
    assert not safe_resolve.is_safe_ip("not-an-ip")
    assert not safe_resolve.is_safe_ip("")


def test_is_safe_ip_rejects_multicast(enforce_ip_safety):
    # ipaddress.is_global considers IPv4 multicast "global"; we must reject it.
    assert not safe_resolve.is_safe_ip("224.0.0.1")
    assert not safe_resolve.is_safe_ip("239.255.255.250")  # SSDP


def test_safe_resolve_rejects_multicast_even_when_private_allowed():
    # No enforce_ip_safety fixture: the autouse fixture leaves ALLOW_CONNECTING_TO_PRIVATE_IPS=True, yet multicast
    # must still be rejected (it is never a valid unicast target).
    with pytest.raises(safe_resolve.DnsLookupUnsafeError):
        safe_resolve.safe_resolve("224.0.0.1")


# IPv6 is unsupported and always rejected -- including public addresses and the evasion-prone encodings of
# internal targets (NAT64 64:ff9b::/96, 6to4 2002::/16, IPv4-mapped ::ffff:0:0/96).
_IPV6_LITERALS = ["2606:4700:4700::1111", "::1", "::ffff:127.0.0.1", "64:ff9b::169.254.169.254", "2002:a9fe:a9fe::"]


def test_is_safe_ip_rejects_all_ipv6(enforce_ip_safety):
    for addr in _IPV6_LITERALS:
        assert not safe_resolve.is_safe_ip(addr), addr


def test_safe_resolve_rejects_ipv6_literals():
    # Rejection must happen before any DNS resolution; otherwise the autouse intercept maps them to 127.0.0.1.
    # DnsLookupUnsafeError subclasses DnsLookupError.
    for addr in _IPV6_LITERALS:
        with pytest.raises(safe_resolve.DnsLookupUnsafeError):
            safe_resolve.safe_resolve(addr)


def test_safe_resolve_returns_safe_ipv4_literal(enforce_ip_safety):
    assert safe_resolve.safe_resolve("8.8.8.8") == "8.8.8.8"


def test_safe_resolve_rejects_host_without_ipv4_address(monkeypatch):
    # A host that resolves to no A records (e.g. an AAAA-only host) is rejected.
    monkeypatch.setattr(safe_resolve, "lookup_v4", lambda host: None)
    with pytest.raises(safe_resolve.DnsLookupError):
        safe_resolve.safe_resolve("ipv6only.example.com")


def test_safe_resolve_rejects_unsafe_ipv4_literal_without_resolving(enforce_ip_safety, monkeypatch):
    # An unsafe IPv4 literal is rejected directly (DnsLookupUnsafeError, not DnsLookupError) and never resolved.
    def _fail(host):
        raise AssertionError(f"lookup_v4 must not be called for an IP literal: {host}")

    monkeypatch.setattr(safe_resolve, "lookup_v4", _fail)
    for addr in ["10.0.0.1", "192.168.1.1", "169.254.169.254"]:
        with pytest.raises(safe_resolve.DnsLookupUnsafeError):
            safe_resolve.safe_resolve(addr)
