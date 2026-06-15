import ipaddress
import socket
from sys import platform

from dns.exception import DNSException
from dns.resolver import resolve
from loguru import logger

from xngin.apiserver.flags import ALLOW_CONNECTING_TO_PRIVATE_IPS

DNS_TIMEOUT_SECS = 5

# Sentinel value that unit tests can use to ensure a host is treated as invalid.
UNSAFE_IP_FOR_TESTING = "127.0.0.9"

# When True, lookup_v4() never performs real DNS resolution and instead returns 127.0.0.1 for every host. An autouse
# fixture flips this on so unit tests never depend on a working resolver. Hosts not in INTERCEPT_DNS_ALLOWLIST are
# logged so we can spot tests that should be using a literal IP or a mock instead.
INTERCEPT_DNS_FOR_TESTING = False
INTERCEPT_DNS_ALLOWLIST = frozenset({"localhost", "example.com"})


class DnsLookupError(Exception):
    """Raised when the DNS lookup of a customer-specified address failed."""

    def __init__(self, host: str):
        self.host = host
        super().__init__(f"DNS issue with host: {host}")


class DnsLookupUnsafeError(DnsLookupError):
    """Raised when the DNS lookup of a customer-specified address succeeded but the result was deemed unsafe."""


def lookup_v4(host: str) -> list[str] | None:
    """Returns the IP addresses for a hostname, or None if there was some kind of failure."""
    if INTERCEPT_DNS_FOR_TESTING:
        if host not in INTERCEPT_DNS_ALLOWLIST:
            logger.warning(f"Intercepting unit test DNS lookup for unexpected host {host!r}; returning 127.0.0.1.")
        return ["127.0.0.1"]
    if platform == "darwin":
        # dnspython doesn't function properly on OSX machines so call socket.getaddrinfo directly.
        try:
            answer = socket.getaddrinfo(host, None, socket.AF_INET)
            return [str(a[4][0]) for a in answer]
        except socket.gaierror:
            return None
    try:
        dns_answer = resolve(host, "A", lifetime=DNS_TIMEOUT_SECS)
        return [r.to_text() for r in dns_answer]
    except DNSException:
        return None


def is_safe_ip(ip: str):
    """Returns true iff the ip is a safe unicast IPv4 address to try to connect to.

    Only unicast IPv4 is supported. IPv6 and multicast addresses are always rejected.

    If ALLOW_CONNECTING_TO_PRIVATE_IPS is enabled, any unicast IPv4 address is accepted without checking whether it is
    globally routable.
    """
    try:
        parsed = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if parsed.version != 4 or parsed.is_multicast:
        return False
    if ALLOW_CONNECTING_TO_PRIVATE_IPS:
        return True
    return parsed.is_global


def is_safe_ipset(ips: set[str]):
    return all(is_safe_ip(address) for address in ips)


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def safe_resolve(host: str | None):
    if not host:
        raise DnsLookupError("Missing hostname.")

    if host == UNSAFE_IP_FOR_TESTING:
        raise DnsLookupError("Detected sentinel value of invalid IP used for testing purposes.")

    # IP literals are decided directly and never resolved as a name.
    if is_safe_ip(host):
        return host

    # If host contains an IP literal, reject it outright so that we do not send an IP address to the resolver.
    if _is_ip_literal(host):
        raise DnsLookupUnsafeError(host)

    # Find an IPv4 A record for host.
    answers = lookup_v4(host)
    if not answers:
        raise DnsLookupError(host)
    safe = is_safe_ipset(set(answers))
    if not safe:
        raise DnsLookupUnsafeError(host)
    return answers.pop()


if __name__ == "__main__":
    print("Resolving:")
    print(safe_resolve("localhost"))
