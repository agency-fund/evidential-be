from typing import Protocol


class ProviderNotRegisteredError(Exception):
    pass


class Provider(Protocol):
    """Kms implements a universal interface for encrypting and decrypting strings using envelope encryption."""

    def name(self) -> str:
        """Returns a short string representing this KMS."""

    def encrypt(self, pt: bytes, aad: bytes) -> bytes:
        """Encrypts a string with optional additional authenticated data.

        :arg pt: plaintext to encrypt
        :arg aad: optional additional authenticated data
        :raises ValueError: raises ValueError if plaintext exceeds size limits
        :returns: encrypted string
        """

    def decrypt(self, ct: bytes, aad: bytes) -> bytes:
        """Decrypts a string with optional additional authenticated data.

        :arg ct: ciphertext to decrypt
        :arg aad: optional additional authenticated data
        :raises ValueError: raises ValueError if ciphertext exceeds size limits
        :raises ValueError: raises ValueError if ciphertext cannot be decrypted
        :returns: plaintext
        """


class Registry:
    """Tracks registration of providers."""

    def __init__(self):
        self.registry: dict[str, Provider] = {}

    def get_providers(self):
        return list(self.registry.keys())

    def get(self, name: str):
        provider = self.registry.get(name)
        if not provider:
            raise ProviderNotRegisteredError(name)
        return provider

    def register(self, name: str, instance: Provider):
        self.registry[name] = instance
