from xngin.xsecrets.provider import Provider, Registry

NAME = "noop"


def initialize(registry: Registry):
    instance = NoopProvider()
    registry.register(instance.name(), instance)


class NoopProvider(Provider):
    """Implements a Kms that doesn't perform any cryptographic operations."""

    def name(self) -> str:
        return NAME

    def encrypt(self, pt: bytes, aad: bytes) -> bytes:
        return pt

    def decrypt(self, ct: bytes, aad: bytes) -> bytes:
        return ct
