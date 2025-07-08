import base64
import os

import tink
from tink import aead, secret_key_access

from xngin.xsecrets import constants
from xngin.xsecrets.provider import Provider, Registry

NAME = "local"


def initialize(registry: Registry, *, static_key: str | None = None):
    key = os.environ.get(constants.ENV_XNGIN_SECRETS_TINK_KEYSET)
    if static_key:
        key = static_key
    if key:
        aead.register()
        instance = LocalProvider(key)
        registry.register(instance.name(), instance)


class LocalProvider(Provider):
    def __init__(self, key: str):
        key = base64.standard_b64decode(key).decode("utf-8")
        keyset_handle = tink.json_proto_keyset_format.parse(
            key, secret_key_access.TOKEN
        )
        self.primitive = keyset_handle.primitive(aead.Aead)  # type: ignore[type-abstract]

    def name(self) -> str:
        return NAME

    def encrypt(self, pt: bytes, aad: bytes) -> bytes:
        return self.primitive.encrypt(pt, aad)

    def decrypt(self, ct: bytes, aad: bytes) -> bytes:
        return self.primitive.decrypt(ct, aad)
