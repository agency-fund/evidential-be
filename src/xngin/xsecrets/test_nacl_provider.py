import base64
import contextlib
import os

import nacl.exceptions
import pytest

from xngin.xsecrets import nacl_provider
from xngin.xsecrets.constants import ENV_XNGIN_SECRETS_NACL_KEYSET
from xngin.xsecrets.exceptions import InvalidSecretStoreConfigurationError
from xngin.xsecrets.nacl_provider import NaclProvider, NaclProviderKeyset
from xngin.xsecrets.provider import Registry


@contextlib.contextmanager
def temporary_env_var(name: str, value: str):
    """Temporarily set environment variable for the duration of the context."""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is not None:
            os.environ[name] = previous
        else:
            os.environ.pop(name, None)


def test_initialize_when_provider_is_not_configured():
    registry = Registry()
    nacl_provider.initialize(registry)
    assert len(registry.get_providers()) == 0


def test_initialize_valid_env_var():
    registry = Registry()
    with temporary_env_var(
        ENV_XNGIN_SECRETS_NACL_KEYSET, NaclProviderKeyset.create().serialize_base64()
    ):
        nacl_provider.initialize(registry)
    registry.get(nacl_provider.NAME)
    assert len(registry.get_providers()) == 1


def test_initialize_env_var_not_base64():
    registry = Registry()
    with (
        temporary_env_var(ENV_XNGIN_SECRETS_NACL_KEYSET, "invalid"),
        pytest.raises(InvalidSecretStoreConfigurationError),
    ):
        nacl_provider.initialize(registry)


def test_initialize_env_var_not_a_keyset():
    registry = Registry()
    with (
        temporary_env_var(
            ENV_XNGIN_SECRETS_NACL_KEYSET,
            base64.standard_b64encode(b"not a keyset").decode(),
        ),
        pytest.raises(InvalidSecretStoreConfigurationError),
    ):
        nacl_provider.initialize(registry)


def test_keyset_construction():
    first_keyset = NaclProviderKeyset.create()
    assert len(first_keyset.keys) == 1

    second_keyset = first_keyset.with_new_key()
    assert len(second_keyset.keys) == 2
    assert second_keyset.keys[1] == first_keyset.keys[0]

    third_keyset = second_keyset.with_new_key().with_new_key()
    assert len(third_keyset.keys) == 4
    assert set(third_keyset.keys[:2]) & set(second_keyset.keys) == set()
    assert third_keyset.keys[2:] == second_keyset.keys


def test_keyset_keys_are_base64_keys_of_appropriate_size():
    keyset = NaclProviderKeyset.create().with_new_key().with_new_key()
    assert len(set(keyset.keys)) == 3
    for key in keyset.keys:
        decoded = base64.b64decode(key)
        assert len(decoded) == 32


def test_keyset_serde():
    keyset = NaclProviderKeyset.create().with_new_key().with_new_key()
    assert NaclProviderKeyset.deserialize_base64(keyset.serialize_base64()) == keyset
    assert NaclProviderKeyset.model_validate_json(keyset.serialize_json()) == keyset


def test_key_rotation():
    first_keyset = NaclProviderKeyset.create()
    first_provider = NaclProvider(first_keyset)
    encrypted = first_provider.encrypt(b"pt", b"aad")

    second_keyset = first_keyset.with_new_key()
    second_provider = NaclProvider(second_keyset)

    assert second_provider.decrypt(encrypted, b"aad") == b"pt", (
        "provider with two keys failed to decrypt ciphertext encrypted with first provider"
    )

    encrypted_with_new_key = second_provider.encrypt(b"pt2", b"aad")

    with pytest.raises(nacl.exceptions.CryptoError):
        first_provider.decrypt(encrypted_with_new_key, b"aad")
