import ipaddress
import socket
from sys import platform

import dns.resolver
from dns.resolver import resolve

from xngin.apiserver.flags import ALLOW_CONNECTING_TO_PRIVATE_IPS

DNS_TIMEOUT_SECS = 5

# Sentinel value that unit tests can use to ensure a host is treated as invalid.
UNSAFE_IP_FOR_TESTING = "127.0.0.9"


class DnsLookupError(Exception):
    """Raised when the DNS lookup of a customer-specified address failed."""


class DnsLookupUnsafeError(DnsLookupError):
    """Raised when the DNS lookup of a customer-specified address succeeded but the result was deemed unsafe."""


def lookup_v4(host: str) -> list[str] | None:
    """Returns the IP addresses for a hostname, or None if there was some kind of failure."""
    if platform == "darwin":
        # dnspython doesn't function properly on OSX machines so call socket.getaddrinfo directly.
        answer = socket.getaddrinfo(host, None, socket.AF_INET)
        return [str(a[4][0]) for a in answer]
    try:
        dns_answer = resolve(host, "A", lifetime=DNS_TIMEOUT_SECS)
        return [r.to_text() for r in dns_answer]
    except dns.exception.DNSException:
        return None


def is_safe_ip(ip):
    """Returns true iff the ip is safe to try to connect to.

    If ALLOW_CONNECTING_TO_PRIVATE_IPS is enabled, we will validate the IP address but not check whether it is
    globally routable.
    """
    try:
        parsed = ipaddress.ip_address(ip)
        if ALLOW_CONNECTING_TO_PRIVATE_IPS:
            return True
        return parsed.is_global and (
            (parsed.version == 4 and parsed.packed[0] != 192)
            or (parsed.version == 6 and parsed.exploded.split(":")[0] not in {"2001", "2620", "64"})
        )
    except ValueError:
        return False


def is_safe_ipset(ips: set[str]):
    return all(is_safe_ip(address) for address in ips)


def safe_resolve(host: str):
    if host == UNSAFE_IP_FOR_TESTING:
        raise DnsLookupError("Detected sentinel value of invalid IP used for testing purposes.")

    # If it is a safe IP address, return it immediately.
    if is_safe_ip(host):
        return host

    answers = lookup_v4(host)
    if not answers:
        raise DnsLookupError(host)
    safe = is_safe_ipset(set(answers))
    if not safe:
        raise DnsLookupUnsafeError(f"lookup({host}) => {answers}")
    return answers.pop()


if __name__ == "__main__":
    print("Resolving:")
    print(safe_resolve("localhost"))
