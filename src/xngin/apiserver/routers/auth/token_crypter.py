import os

from pydantic import ValidationError

from xngin.xsecrets.chafernet import Chafernet, InvalidTokenError
from xngin.xsecrets.nacl_provider import NaclProvider, NaclProviderKeyset
from xngin.xsecrets.noop_provider import NoopProvider

LOCAL_KEYSET_SENTINEL = "local"


def _read_local_keyset(local_keyset_filename: str) -> str:
    """Development environments may use a key in the local filesystem."""
    try:
        with open(local_keyset_filename) as f:
            return f.read()
    except OSError as err:
        raise TokenCrypterMisconfiguredError(f"The {local_keyset_filename} file cannot be read.") from err


class TokenCrypterMisconfiguredError(Exception):
    pass


class TokenCrypter:
    """Convenience wrapper for Chafernet tokens with configurable keyset source and token prefix."""

    def __init__(
        self,
        ttl: int,
        keyset_env_var: str,
        local_keyset_filename: str,
        prefix: str,
        allow_noop_fallback: bool = False,
    ):
        self._chafernet: Chafernet | None = None
        self._ttl = ttl
        self._keyset_env_var = keyset_env_var
        self._local_keyset_filename = local_keyset_filename
        self._prefix = prefix
        self._allow_noop_fallback = allow_noop_fallback

    @property
    def _instance(self) -> Chafernet:
        if not self._chafernet:
            try:
                provider = self._new_provider()
            except TokenCrypterMisconfiguredError:
                if not self._allow_noop_fallback:
                    raise
                provider = NoopProvider()
            self._chafernet = Chafernet(provider)
        return self._chafernet

    def _new_provider(self):
        keys = os.environ.get(self._keyset_env_var, "")
        if not keys:
            raise TokenCrypterMisconfiguredError(f"{self._keyset_env_var} is not set but is required.")
        if keys == LOCAL_KEYSET_SENTINEL:
            keys = _read_local_keyset(self._local_keyset_filename)
        try:
            keyset = NaclProviderKeyset.deserialize_base64(keys)
        except ValidationError as err:
            raise TokenCrypterMisconfiguredError(f"{self._keyset_env_var} is invalid") from err
        return NaclProvider(keyset)

    def encrypt(self, plaintext: bytes | str) -> str:
        return self._prefix + self._instance.encrypt(plaintext, b"")

    def decrypt(self, token: str) -> bytes:
        if not token.startswith(self._prefix):
            raise InvalidTokenError
        return self._instance.decrypt(token[len(self._prefix) :], b"", self._ttl)
