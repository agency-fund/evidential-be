import json
import os

from pydantic import ValidationError

from xngin.apiserver import flags
from xngin.apiserver.routers.auth.principal import Principal
from xngin.xsecrets.chafernet import Chafernet, InvalidTokenError
from xngin.xsecrets.nacl_provider import NaclProvider, NaclProviderKeyset

SESSION_TOKEN_PREFIX = "xa_"


class SessionTokenCrypterMisconfiguredError(Exception):
    pass


class SessionTokenCrypter:
    """Convenience wrapper for Chafernet tokens for encoding a Principal."""

    def __init__(self, ttl: int):
        self._chafernet: Chafernet | None = None
        self._ttl = ttl

    @property
    def _instance(self):
        if not self._chafernet:
            keys = os.environ.get(flags.ENV_SESSION_TOKEN_KEYSET, "")
            if not keys:
                raise SessionTokenCrypterMisconfiguredError(
                    f"{flags.ENV_SESSION_TOKEN_KEYSET} is not set but is required."
                )
            try:
                keyset = NaclProviderKeyset.deserialize_base64(keys)
            except ValidationError as err:
                raise SessionTokenCrypterMisconfiguredError(f"{flags.ENV_SESSION_TOKEN_KEYSET} is invalid") from err
            self._chafernet = Chafernet(NaclProvider(keyset))
        return self._chafernet

    def encrypt(self, principal: Principal):
        return SESSION_TOKEN_PREFIX + self._instance.encrypt(
            json.dumps(principal.model_dump(), separators=(",", ":")).encode(), b""
        )

    def decrypt(self, token: str) -> Principal:
        if not token.startswith(SESSION_TOKEN_PREFIX):
            raise InvalidTokenError
        decrypted = self._instance.decrypt(token[len(SESSION_TOKEN_PREFIX) :], b"", self._ttl)
        return Principal.model_validate_json(decrypted)
