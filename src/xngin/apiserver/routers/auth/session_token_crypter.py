import json
from datetime import timedelta

from pydantic import ValidationError

from xngin.apiserver import flags
from xngin.apiserver.routers.auth.principal import Principal
from xngin.xsecrets.chafernet import Chafernet
from xngin.xsecrets.nacl_provider import NaclProvider, NaclProviderKeyset


class SessionTokenCrypter:
    """Convenience wrapper for Chafernet tokens for encoding a Principal."""

    def __init__(self):
        self._chafernet = None

    @property
    def _instance(self):
        if not self._chafernet:
            keys = flags.SESSION_TOKEN_KEYSET
            if not keys:
                raise RuntimeError(f"{flags.ENV_SESSION_TOKEN_KEYSET} is not set but is required.")
            try:
                keyset = NaclProviderKeyset.deserialize_base64(keys)
            except ValidationError as err:
                raise RuntimeError(f"{flags.ENV_SESSION_TOKEN_KEYSET} is invalid") from err
            self._chafernet = Chafernet(NaclProvider(keyset))
        return self._chafernet

    def encrypt(self, principal: Principal):
        return self._instance.encrypt(json.dumps(principal.model_dump(), separators=(",", ":")), b"")

    def decrypt(self, token: str) -> Principal:
        decrypted = self._instance.decrypt(token, b"", timedelta(hours=12).seconds)
        return Principal.model_validate_json(decrypted)
