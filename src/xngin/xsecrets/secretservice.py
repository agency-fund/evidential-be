import base64
import os
from typing import Optional

from loguru import logger

from xngin.xsecrets import (
    aws_provider,
    gcp_provider,
    local_provider,
    noop_provider,
)
from xngin.xsecrets.constants import (
    ENV_XNGIN_SECRETS_BACKEND,
    SERIALIZED_ENCRYPTED_VALUE_PREFIX,
)
from xngin.xsecrets.exceptions import InvalidSecretStoreConfigurationError
from xngin.xsecrets.provider import Provider, Registry

_SERVICE: Optional["SecretService"] = None


def setup():
    """Configures a secrets service according to environment variables."""
    global _SERVICE

    registry = Registry()

    aws_provider.initialize(registry)
    gcp_provider.initialize(registry)
    local_provider.initialize(registry)

    registered = registry.get_providers()

    backend_spec = os.environ.get(ENV_XNGIN_SECRETS_BACKEND, noop_provider.NAME)
    if backend_spec == noop_provider.NAME:
        logger.warning(
            f"Secrets: Encryption is disabled because {ENV_XNGIN_SECRETS_BACKEND} is unset "
            f"or set to {noop_provider.NAME}."
        )
        noop_provider.initialize(registry)
        backend_spec = noop_provider.NAME
    elif backend_spec not in registered:
        raise InvalidSecretStoreConfigurationError(
            f"Requested backend '{backend_spec}' is not registered (available: {', '.join(registered)})"
        )

    logger.info(
        f"Secrets: Using '{backend_spec}' for encryption (available: {', '.join(registered)})"
    )
    _SERVICE = SecretService(registry.get(backend_spec), registry)


def _serialize(backend: str, ciphertext: bytes):
    """Serializes the encrypted secret as a string."""
    return f"{SERIALIZED_ENCRYPTED_VALUE_PREFIX}{backend}:{base64.standard_b64encode(ciphertext).decode()}"


def _deserialize(serialized: str) -> (str, bytes):
    """Deserializes a string encoded with _serialize into a provider name and ciphertext."""
    prefix = SERIALIZED_ENCRYPTED_VALUE_PREFIX
    if len(serialized) < len(prefix) or not serialized.startswith(prefix):
        raise ValueError(f"String must start with '{prefix}' prefix")
    start_pos = len(prefix)
    separator_pos = serialized.find(":", start_pos)
    if separator_pos == -1:
        raise ValueError("Missing separator colon between kms and ciphertext")
    kms = serialized[start_pos:separator_pos]
    ciphertext = serialized[separator_pos + 1 :]
    return kms, base64.standard_b64decode(ciphertext)


class SecretService:
    """Implements a secret storage mechanism for string values."""

    def __init__(self, encryption_provider: Provider, registry: Registry):
        """Configures SecretService for cryptographic operations."""
        self.provider = encryption_provider
        self.registry = registry

    def encrypt(self, pt: str, aad: str) -> str:
        """Encrypts a value with the default encryption provider."""
        return _serialize(
            backend=self.provider.name(),
            ciphertext=self.provider.encrypt(pt.encode("utf-8"), aad.encode("utf-8")),
        )

    def decrypt(self, ct: str, aad: str) -> str:
        """Decrypts a value with any registered encryption provider.

        If the ciphertext does not have the expected prefix, we assume it is not actually encrypted, and return it
        as-is. This allows backwards compatibility with values persisted prior to the introduction of encryption.
        """
        if not ct.startswith(SERIALIZED_ENCRYPTED_VALUE_PREFIX):
            return ct
        kms, ciphertext = _deserialize(ct)
        provider = self.registry.get(kms)
        return (provider.decrypt(ciphertext, aad.encode("utf-8"))).decode("utf-8")


def get_symmetric() -> SecretService:
    if not _SERVICE:
        raise InvalidSecretStoreConfigurationError(
            "setup() must be called before encryption operations"
        )
    return _SERVICE
