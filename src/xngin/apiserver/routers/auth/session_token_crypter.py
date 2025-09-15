import json
import os

from pydantic import ValidationError

from xngin.apiserver import flags
from xngin.apiserver.routers.auth.principal import Principal
from xngin.xsecrets.chafernet import Chafernet, InvalidTokenError
from xngin.xsecrets.nacl_provider import NaclProvider, NaclProviderKeyset

# File containing the session token key to read when XNGIN_SESSION_TOKEN_KEYSET is set to "local".
LOCAL_KEYSET_FILE = ".xngin_session_token_keyset"

# The session token value is prefixed with this string to visually distinguish it from other tokens.
SESSION_TOKEN_PREFIX = "xa_"


def _read_local_keyset(keys):
    """Development environments may use a key in the local filesystem."""
    try:
        with open(LOCAL_KEYSET_FILE) as f:
            keys = f.read()
    except OSError as err:
        raise SessionTokenCrypterMisconfiguredError(f"The {LOCAL_KEYSET_FILE} file cannot be read.") from err
    return keys


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
                if keys == "local":
                    keys = _read_local_keyset(keys)
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
