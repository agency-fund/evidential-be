import base64
import binascii
import json
import os
from typing import Annotated

import nacl.exceptions
import nacl.secret
import nacl.utils
from annotated_types import Len
from pydantic import BaseModel, ValidationError

from xngin.xsecrets import constants
from xngin.xsecrets.exceptions import InvalidSecretStoreConfigurationError
from xngin.xsecrets.provider import Provider, Registry

NAME = "nacl"


class NaclProviderKeyset(BaseModel):
    """Describes the active nacl encryption keys."""

    # The encryption keys. The first element in the list is the default encryption key. Decrypt operations will be
    # tried against all the keys until one succeeds.
    keys: Annotated[list[str], Len(min_length=1)]

    def with_new_key(self):
        """Returns a new copy of the current instance with a new default key.

        Useful when rotating keys.
        """
        return self.model_copy(update={"keys": [self._create_key(), *self.keys]})

    def serialize_json(self) -> str:
        """Returns the keyset serialized to compact JSON."""
        return json.dumps(
            self.model_dump(),
            separators=(",", ":"),
            sort_keys=True,
        )

    def serialize_base64(self) -> str:
        """Returns the keyset serialized to base64."""
        return base64.standard_b64encode(self.serialize_json().encode()).decode()

    @classmethod
    def deserialize_base64(cls, base64_keyset: str) -> "NaclProviderKeyset":
        """Constructs a new keyset from the output of serialize_base64."""
        return NaclProviderKeyset.model_validate_json(base64.standard_b64decode(base64_keyset))

    @classmethod
    def _create_key(cls):
        """Creates a new base64 encoded key."""
        key_bytes = nacl.utils.random(nacl.secret.Aead.KEY_SIZE)
        return base64.standard_b64encode(key_bytes).decode()

    @classmethod
    def create(cls):
        """Creates a new instance of KeySet with a single key."""
        return NaclProviderKeyset(keys=[cls._create_key()])


def initialize(registry: Registry, *, keyset: NaclProviderKeyset | None = None):
    """Registers a NaclProvider with the registry if configuration information is available to do so."""
    if keyset is None and (key := os.environ.get(constants.ENV_XNGIN_SECRETS_NACL_KEYSET)):
        try:
            keyset = NaclProviderKeyset.deserialize_base64(key)
        except (binascii.Error, ValidationError) as err:
            raise InvalidSecretStoreConfigurationError(
                f"{constants.ENV_XNGIN_SECRETS_NACL_KEYSET} is not valid base64 encoded keyset"
            ) from err
    if keyset:
        instance = NaclProvider(keyset)
        registry.register(instance.name(), instance)


class NaclProvider(Provider):
    """Provides pynacl's Aead."""

    def __init__(self, keyset: NaclProviderKeyset):
        self.boxes = [nacl.secret.Aead(base64.standard_b64decode(key)) for key in keyset.keys]

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
