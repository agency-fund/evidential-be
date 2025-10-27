import base64
import binascii
import time

import nacl.exceptions

from xngin.xsecrets.nacl_provider import NaclProvider

# Number of seconds in the past that a token is allowed to be issued at.
_MAX_CLOCK_SKEW = 5

# Version number of the serialized payload.
_VERSION = 1


class InvalidTokenError(Exception):
    pass


_urlsafe_decode_translation = bytes.maketrans(b"-_", b"+/")


def _safe_encode(encrypted: bytes) -> str:
    return base64.urlsafe_b64encode(encrypted).decode()


def _safe_decode(encoded: str) -> bytes:
    """Equivalent to base64.urlsafe_b64decode, except passes validate=True to b64decode()."""
    translated = encoded.encode().translate(_urlsafe_decode_translation)
    return base64.b64decode(translated, validate=True)


def _to_bytes(s: str | bytes):
    if isinstance(s, str):
        return s.encode()
    return s


class Chafernet:
    """Chafernet encrypts and decrypts messages, with authentication and timestamps.

    This differs from Fernet [1] in a few ways. Chafernet uses xchacha20 AEAD instead of AES128-CBC+HMAC. The timestamp
    is encrypted under the AEAD cipher rather than left plaintext in the token. We get key rotation from
    via NaclProvider. We also allow AAD to be provided.

    The encrypted value is: BASE64URL ( ENCRYPT ( VERSION (1 byte) || TIMESTAMP (8 bytes) || PLAINTEXT ) )

    The canonical representation of the encrypted value is a base64url string.

    [1] https://cryptography.io/en/latest/fernet/
    """

    def __init__(self, nacl_provider: NaclProvider) -> None:
        self.nacl_provider = nacl_provider

    def encrypt(self, plaintext: str | bytes, aad: bytes) -> str:
        """Encrypts plaintext at the current time and returns base64url encoded ciphertext."""
        return self.encrypt_at_time(plaintext, aad, int(time.time()))

    def encrypt_at_time(self, plaintext: str | bytes, aad: bytes, current_time: int) -> str:
        """Encrypts plaintext at the specified time and returns base64url encoded ciphertext."""
        return self._encrypt_from_parts(plaintext, aad=aad, current_time=current_time)

    def _encrypt_from_parts(self, plaintext: str | bytes, *, aad: bytes, current_time: int) -> str:
        plaintext_bytes = _to_bytes(plaintext)
        serialized = (
            _VERSION.to_bytes(length=1, byteorder="big")
            + current_time.to_bytes(length=8, byteorder="big")
            + plaintext_bytes
        )
        encrypted = self.nacl_provider.encrypt(serialized, aad)
        return _safe_encode(encrypted)

    def decrypt(self, ciphertext: str, aad: bytes, ttl: int) -> bytes:
        """Decrypts the Chafernet token, checks timestamp against TTL using current time, and returns plaintext."""
        return self._decrypt(ciphertext, aad=aad, ttl=ttl, current_time=int(time.time()))

    def decrypt_at_time(self, ciphertext: str, *, aad: bytes, ttl: int, current_time: int) -> bytes:
        """Decrypts the Chafernet token, checks timestamp against TTL using specified time, and returns plaintext."""
        return self._decrypt(ciphertext, aad=aad, ttl=ttl, current_time=current_time)

    def _decrypt(self, ciphertext: str, *, aad: bytes, ttl: int, current_time: int) -> bytes:
        try:
            decoded = _safe_decode(ciphertext)
        except binascii.Error:
            raise InvalidTokenError from None
        try:
            plaintext = self.nacl_provider.decrypt(decoded, aad)
        except nacl.exceptions.CryptoError:
            raise InvalidTokenError from None
        if plaintext[0] != _VERSION:
            raise InvalidTokenError
        timestamp = int.from_bytes(plaintext[1:9], byteorder="big")
        if timestamp + ttl < current_time:
            raise InvalidTokenError
        if current_time + _MAX_CLOCK_SKEW < timestamp:
            raise InvalidTokenError
        return plaintext[9:]
