import base64
import json

import pytest

from xngin.xsecrets.constants import SERIALIZED_ENCRYPTED_VALUE_PREFIX
from xngin.xsecrets.secretservice import _deserialize, _serialize  # noqa: PLC2701


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
