import base64
import json
import os

import nacl.exceptions
import nacl.secret
from chafernet import Keyset

from xngin.xsecrets import constants
from xngin.xsecrets.exceptions import InvalidSecretStoreConfigurationError
from xngin.xsecrets.provider import Provider, Registry

NAME = "nacl"


def initialize(registry: Registry, *, keyset: Keyset | None = None):
    """Registers a NaclProvider with the registry if configuration information is available to do so."""
    if keyset is None and (key := os.environ.get(constants.ENV_XNGIN_SECRETS_NACL_KEYSET)):
        try:
            keyset = Keyset.from_base64(key)
        except ValueError as err:
            raise InvalidSecretStoreConfigurationError(
                f"{constants.ENV_XNGIN_SECRETS_NACL_KEYSET} is not valid base64 encoded keyset"
            ) from err
    if keyset is not None:
        instance = NaclProvider(keyset)
        registry.register(instance.name(), instance)


class NaclProvider(Provider):
    """Provides pynacl's Aead."""

    def __init__(self, keyset: Keyset):
        self.boxes = [nacl.secret.Aead(base64.standard_b64decode(key)) for key in json.loads(keyset.to_json())["keys"]]

    def name(self) -> str:
        return NAME

    def encrypt(self, pt: bytes, aad: bytes) -> bytes:
        """Encrypts using the current default (first) key."""
        return self.boxes[0].encrypt(pt, aad)

    def decrypt(self, ct: bytes, aad: bytes) -> bytes:
        """Decrypts ct with the available keys, in order.

        Raises:
            ValueError or TypeError on message format or configuration problems.
            CryptoError when the value is structurally correct but not decryptable with any of the keys.
        """
        head, last = self.boxes[:-1], self.boxes[-1]
        for box in head:
            try:
                return box.decrypt(ct, aad)
            except nacl.exceptions.CryptoError:
                pass
        return last.decrypt(ct, aad)
