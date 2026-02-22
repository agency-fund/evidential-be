import os

from pydantic import ValidationError

from xngin.xsecrets.chafernet import Chafernet, InvalidTokenError
from xngin.xsecrets.nacl_provider import NaclProvider, NaclProviderKeyset

LOCAL_KEYSET_SENTINEL = "local"


def _read_local_keyset(local_keyset_filename: str) -> str:
    """Development environments may use a key in the local filesystem."""
    try:
        with open(local_keyset_filename) as f:
            return f.read()
    except OSError as err:
        raise TokenCryptorMisconfiguredError(f"The {local_keyset_filename} file cannot be read.") from err


class TokenCryptorMisconfiguredError(Exception):
    pass


class TokenCryptor:
    """TokenCryptor configures a Chafernet with NaclProvider using environment variables as the primary source of keys.

    These tokens allow passing opaque blobs of authenticated and encrypted data to the client. They can only be
    decoded by the server.
    """

    def __init__(self, ttl: int, keyset_env_var: str, local_keyset_filename: str, prefix: str):
        self._chafernet: Chafernet | None = None
        self._ttl = ttl
        self._keyset_env_var = keyset_env_var
        self._local_keyset_filename = local_keyset_filename
        self._prefix = prefix

    @property
    def _instance(self) -> Chafernet:
        if not self._chafernet:
            provider = self._new_provider()
            self._chafernet = Chafernet(provider)
        return self._chafernet

    def _new_provider(self):
        keys = os.environ.get(self._keyset_env_var, "")
        if not keys:
            raise TokenCryptorMisconfiguredError(f"{self._keyset_env_var} is not set but is required.")
        if keys == LOCAL_KEYSET_SENTINEL:
            keys = _read_local_keyset(self._local_keyset_filename)
        try:
            keyset = NaclProviderKeyset.deserialize_base64(keys)
        except ValidationError as err:
            raise TokenCryptorMisconfiguredError(f"{self._keyset_env_var} is invalid") from err
        return NaclProvider(keyset)

    def encrypt(self, plaintext: bytes | str) -> str:
        return self._prefix + self._instance.encrypt(plaintext, b"")

    def decrypt(self, token: str) -> bytes:
        if not token.startswith(self._prefix):
            raise InvalidTokenError
        return self._instance.decrypt(token[len(self._prefix) :], b"", self._ttl)
