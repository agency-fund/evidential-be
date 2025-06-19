import base64
import json

import pytest
from tink import TinkError

from xngin.xsecrets import local_provider
from xngin.xsecrets.constants import SERIALIZED_ENCRYPTED_VALUE_PREFIX
from xngin.xsecrets.provider import Registry
from xngin.xsecrets.secretservice import (
    SecretService,
    _deserialize,  # noqa: PLC2701
    _serialize,  # noqa: PLC2701
)


@pytest.mark.parametrize(
    "backend,ciphertext",
    [
        ("test_backend", b"encrypted_data"),
        (
            "gcp_kms",
            b"\xe2\x82\xac\xf0\x9f\x98\x80\x00\xff\xfe\xfd\xfc\xfb\xfa",
        ),  # non-ASCII bytes
        ("aws_kms", b""),  # Empty ciphertext
    ],
)
def test_serialize_deserialize_roundtrip(backend, ciphertext):
    """Test serialization and deserialization of encrypted data with various inputs."""
    # Test serialization
    serialized = _serialize(backend, ciphertext)

    # Check prefix
    assert serialized.startswith(SERIALIZED_ENCRYPTED_VALUE_PREFIX)

    # Test deserialization (roundtrip)
    result_backend, result_ciphertext = _deserialize(serialized)

    assert result_backend == backend
    assert result_ciphertext == ciphertext


def test_deserialize_invalid_prefix():
    """Test deserialization with invalid prefix."""
    with pytest.raises(ValueError, match="String must start with"):
        _deserialize("invalid_prefix{}")
    with pytest.raises(ValueError, match="String must start with"):
        _deserialize("")


def test_deserialize_invalid_json():
    """Test deserialization with invalid JSON."""
    invalid_serialized = f"{SERIALIZED_ENCRYPTED_VALUE_PREFIX}{{invalid json"

    with pytest.raises(ValueError, match="does not match expected format"):
        _deserialize(invalid_serialized)


def test_deserialize_invalid_structure():
    """Test deserialization with valid JSON but invalid structure."""
    # Test with a string instead of a list
    invalid_serialized = f"{SERIALIZED_ENCRYPTED_VALUE_PREFIX}" + json.dumps(
        "not_a_list"
    )
    with pytest.raises(ValueError, match="does not match expected format"):
        _deserialize(invalid_serialized)

    # Test with a list that doesn't contain a nested list
    invalid_serialized = f"{SERIALIZED_ENCRYPTED_VALUE_PREFIX}" + json.dumps([
        "not_a_nested_list"
    ])
    with pytest.raises(ValueError, match="does not match expected format"):
        _deserialize(invalid_serialized)

    # Test with a nested list that's too short
    invalid_serialized = f"{SERIALIZED_ENCRYPTED_VALUE_PREFIX}" + json.dumps([
        ["backend"]
    ])
    with pytest.raises(ValueError, match="does not match expected format"):
        _deserialize(invalid_serialized)


SAMPLE_TINK_KEY = base64.standard_b64encode(
    json.dumps({
        "primaryKeyId": 791033902,
        "key": [
            {
                "keyData": {
                    "typeUrl": "type.googleapis.com/google.crypto.tink.AesGcmKey",
                    "value": "GhBkcDg+TK2y1NiO/jPT6P96",
                    "keyMaterialType": "SYMMETRIC",
                },
                "status": "ENABLED",
                "keyId": 791033902,
                "outputPrefixType": "TINK",
            }
        ],
    }).encode("utf-8")
).decode("utf-8")


@pytest.fixture
def secretservice():
    registry = Registry()
    local_provider.initialize(registry, static_key=SAMPLE_TINK_KEY)
    return SecretService(registry.get("local"), registry)


@pytest.mark.parametrize(
    "plaintext,aad",
    [
        ("This is a secret value", "test_context"),
        ("Unicode text with emojis 😀🔑🌍 and special chars €£¥", "unicode_test"),
        ("", "empty_plaintext_test"),  # Test with empty plaintext
        ("non-empty plaintext", ""),  # Test with empty AAD
        ("", ""),  # Test with both empty
    ],
)
def test_secretservice_encrypt_decrypt(secretservice, plaintext, aad):
    """Test that SecretService can encrypt and decrypt values with various inputs."""
    # Encrypt the value
    encrypted = secretservice.encrypt(plaintext, aad)

    # Verify it has the expected format
    assert encrypted.startswith(SERIALIZED_ENCRYPTED_VALUE_PREFIX)

    # Decrypt the value
    decrypted = secretservice.decrypt(encrypted, aad)

    # Verify we got the original plaintext back
    assert decrypted == plaintext


@pytest.mark.parametrize(
    "plaintext,aad",
    [
        ("This is not an encrypted value", "any_context"),
        ("", "empty_plaintext_test"),  # Test with empty plaintext
        ("non-empty plaintext", ""),  # Test with empty AAD
    ],
)
def test_secretservice_decrypt_unencrypted_value(secretservice, plaintext, aad):
    """Test that SecretService.decrypt returns unencrypted values as-is."""
    # Since the value doesn't have the prefix, it should be returned as-is
    result = secretservice.decrypt(plaintext, aad)

    assert result == plaintext


@pytest.mark.parametrize(
    "plaintext,correct_aad,wrong_aad",
    [
        ("Secret message", "correct_context", "wrong_context"),
        ("", "correct_context", "wrong_context"),  # Empty plaintext
        ("Secret message", "", "non-empty"),  # Empty correct AAD
    ],
)
def test_secretservice_decrypt_with_wrong_aad(
    secretservice, plaintext, correct_aad, wrong_aad
):
    """Test that decryption fails when using the wrong AAD."""
    encrypted = secretservice.encrypt(plaintext, correct_aad)

    # Attempting to decrypt with the wrong AAD should raise an exception
    with pytest.raises(TinkError):
        secretservice.decrypt(encrypted, wrong_aad)


def test_secretservice_provider_selection(secretservice):
    """Test that SecretService uses the correct provider based on the serialized data."""
    # The fixture uses the LocalProvider
    plaintext = "Test provider selection"
    aad = "provider_test"

    encrypted = secretservice.encrypt(plaintext, aad)

    # Verify the provider name in the serialized data
    _, serialized_json = encrypted.split(SERIALIZED_ENCRYPTED_VALUE_PREFIX, 1)
    data = json.loads(serialized_json)

    # The provider name should be "local"
    assert data[0][0] == "local"

    # Decryption should work
    assert secretservice.decrypt(encrypted, aad) == plaintext
