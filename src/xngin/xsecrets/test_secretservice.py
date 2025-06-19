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


def test_serialize_basic():
    """Test basic serialization of encrypted data."""
    backend = "test_backend"
    ciphertext = b"encrypted_data"

    result = _serialize(backend, ciphertext)

    # Check prefix
    assert result.startswith(SERIALIZED_ENCRYPTED_VALUE_PREFIX)

    # Parse the JSON part
    json_str = result[len(SERIALIZED_ENCRYPTED_VALUE_PREFIX) :]
    parsed = json.loads(json_str)

    # Verify structure
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert len(parsed[0]) == 2
    assert parsed[0][0] == backend
    assert base64.standard_b64decode(parsed[0][1]) == ciphertext


def test_serialize_with_non_ascii_bytes():
    """Test serialization with non-ASCII bytes in ciphertext."""
    backend = "test_backend"
    # Include various non-ASCII bytes and special characters
    ciphertext = b"\xe2\x82\xac\xf0\x9f\x98\x80\x00\xff\xfe\xfd\xfc\xfb\xfa"

    result = _serialize(backend, ciphertext)

    # Check prefix
    assert result.startswith(SERIALIZED_ENCRYPTED_VALUE_PREFIX)

    # Parse the JSON part
    json_str = result[len(SERIALIZED_ENCRYPTED_VALUE_PREFIX) :]
    parsed = json.loads(json_str)

    # Verify structure and content
    assert parsed[0][0] == backend
    assert base64.standard_b64decode(parsed[0][1]) == ciphertext


def test_deserialize_basic():
    """Test basic deserialization of encrypted data."""
    backend = "test_backend"
    ciphertext = b"encrypted_data"
    serialized = _serialize(backend, ciphertext)

    result_backend, result_ciphertext = _deserialize(serialized)

    assert result_backend == backend
    assert result_ciphertext == ciphertext


def test_deserialize_with_non_ascii_bytes():
    """Test deserialization with non-ASCII bytes in ciphertext."""
    backend = "gcp_kms"
    # Include various non-ASCII bytes and special characters
    ciphertext = b"\xe2\x82\xac\xf0\x9f\x98\x80\x00\xff\xfe\xfd\xfc\xfb\xfa"
    serialized = _serialize(backend, ciphertext)

    result_backend, result_ciphertext = _deserialize(serialized)

    assert result_backend == backend
    assert result_ciphertext == ciphertext


def test_serialize_deserialize_roundtrip():
    """Test that serialization followed by deserialization returns the original values."""
    backend = "local"
    ciphertext = (
        b"Some \xf0\x9f\x94\x92 encrypted data with non-ASCII \xe2\x98\xa2 bytes"
    )

    serialized = _serialize(backend, ciphertext)
    result_backend, result_ciphertext = _deserialize(serialized)

    assert result_backend == backend
    assert result_ciphertext == ciphertext


def test_deserialize_invalid_prefix():
    """Test deserialization with invalid prefix."""
    with pytest.raises(ValueError, match="String must start with"):
        _deserialize("invalid_prefix{}")


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


def test_secretservice_encrypt_decrypt(secretservice):
    """Test that SecretService can encrypt and decrypt values."""
    plaintext = "This is a secret value"
    aad = "test_context"

    # Encrypt the value
    encrypted = secretservice.encrypt(plaintext, aad)

    # Verify it has the expected format
    assert encrypted.startswith(SERIALIZED_ENCRYPTED_VALUE_PREFIX)

    # Decrypt the value
    decrypted = secretservice.decrypt(encrypted, aad)

    # Verify we got the original plaintext back
    assert decrypted == plaintext


def test_secretservice_encrypt_decrypt_with_unicode(secretservice):
    """Test that SecretService can handle Unicode characters."""
    plaintext = "Unicode text with emojis 😀🔑🌍 and special chars €£¥"
    aad = "unicode_test"

    encrypted = secretservice.encrypt(plaintext, aad)
    decrypted = secretservice.decrypt(encrypted, aad)

    assert decrypted == plaintext


def test_secretservice_decrypt_unencrypted_value(secretservice):
    """Test that SecretService.decrypt returns unencrypted values as-is."""
    plaintext = "This is not an encrypted value"
    aad = "any_context"

    # Since the value doesn't have the prefix, it should be returned as-is
    result = secretservice.decrypt(plaintext, aad)

    assert result == plaintext


def test_secretservice_decrypt_with_wrong_aad(secretservice):
    """Test that decryption fails when using the wrong AAD."""
    plaintext = "Secret message"
    correct_aad = "correct_context"
    wrong_aad = "wrong_context"

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
