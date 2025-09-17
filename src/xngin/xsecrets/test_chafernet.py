import base64
import random
from operator import itemgetter

import pytest

from xngin.xsecrets.chafernet import Chafernet, InvalidTokenError
from xngin.xsecrets.nacl_provider import NaclProvider, NaclProviderKeyset

TEST_PLAINTEXTS = ["string", b"", b"\0abc", b"\xff", b"\xc0", b"emoji\xe2\x82\xac\x00\x01", b"long" + b"abcd" * 1000]


plaintexts = pytest.mark.parametrize("plaintext", TEST_PLAINTEXTS, ids=itemgetter(slice(4)))


@pytest.fixture
def chaf():
    return Chafernet(NaclProvider(NaclProviderKeyset.create()))


@plaintexts
def test_chafernet(plaintext, chaf):
    ttl = 30
    encrypted = chaf.encrypt(plaintext, b"")
    decrypted = chaf.decrypt(encrypted, b"", ttl)
    assert decrypted == plaintext if isinstance(plaintext, bytes) else plaintext.encode()

    # decrypt fails with a different AAD
    with pytest.raises(InvalidTokenError):
        chaf.decrypt(encrypted, b"x", ttl)

    # decrypt fails when a random byte is modified
    encrypted_mut = bytearray(base64.urlsafe_b64decode(encrypted))
    encrypted_mut[random.randrange(0, len(encrypted_mut))] ^= 0xFF
    modified = base64.urlsafe_b64encode(encrypted_mut).decode()
    with pytest.raises(InvalidTokenError):
        chaf.decrypt(modified, b"", ttl)


def test_chafernet_expiration(chaf):
    now = 100
    ttl = 30
    encrypted = chaf.encrypt_at_time("plaintext", b"", now)
    last_valid_moment = now + ttl
    chaf.decrypt_at_time(encrypted, aad=b"", ttl=ttl, current_time=last_valid_moment)
    with pytest.raises(InvalidTokenError):
        chaf.decrypt_at_time(encrypted, aad=b"", ttl=ttl, current_time=last_valid_moment + 1)


@plaintexts
def test_chafernet_clock_skew(plaintext, chaf):
    now = 100
    encrypted = chaf.encrypt_at_time(plaintext, b"", now)
    # Small skew is tolerated
    decrypted = chaf.decrypt_at_time(encrypted, aad=b"", ttl=1, current_time=now - 3)
    assert decrypted == plaintext if isinstance(plaintext, bytes) else plaintext.encode()

    # Great skew is not
    with pytest.raises(InvalidTokenError):
        chaf.decrypt_at_time(encrypted, aad=b"", ttl=1, current_time=now - 6)
