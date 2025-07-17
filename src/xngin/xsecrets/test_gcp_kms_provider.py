"""Unit and integration tests for the gcp_kms_provider.

To run the integration tests:

    export XNGIN_SECRETS_GCP_CREDENTIALS=...
    export XNGIN_SECRETS_GCP_KMS_KEY_URL=...
    uv run pytest -vv -rA -m integration src/xngin/xsecrets
"""

import itertools
import os

import google.api_core.exceptions
import nacl.exceptions
import pytest

from xngin.xsecrets import constants
from xngin.xsecrets.gcp_kms_provider import (
    GcpKmsProvider,
    _pack_envelope_ciphertext,  # noqa: PLC2701
    _read_gcp_env,  # noqa: PLC2701
    _unpack_envelope_ciphertext,  # noqa: PLC2701
)

# Skips an integration test if the environment variables aren't available.
skip_unless_gcp_available = pytest.mark.skipif(
    not (
        os.environ.get(constants.ENV_XNGIN_SECRETS_GCP_CREDENTIALS)
        and os.environ.get(constants.ENV_XNGIN_SECRETS_GCP_KMS_KEY_URI)
    ),
    reason="GCP credentials and KMS key URI required for integration tests",
)


# Provides a GcpKmsProvider configured from environment variables. This should only be used when the
# variables are defined.
@pytest.fixture(name="gcp_kms_provider")
def fixture_gcp_kms_provider():
    """Fixture that provides a GcpKmsProvider instance for integration tests."""
    config = _read_gcp_env()
    assert config is not None, "GCP configuration should be available"
    return GcpKmsProvider(key_name=config.key_name, credentials=config.credentials)


def test_pack_envelope_ciphertext():
    """Test packing encrypted DEK and data into combined ciphertext."""
    encrypted_dek = b"fake_encrypted_dek_12345"
    encrypted_data = b"fake_encrypted_data_contents"

    result = _pack_envelope_ciphertext(encrypted_dek, encrypted_data)

    # First 4 bytes should be the length of encrypted_dek as little-endian uint32
    expected_length = len(encrypted_dek)
    assert result[:4] == expected_length.to_bytes(4, "little")

    # Next bytes should be the encrypted DEK
    assert result[4 : 4 + expected_length] == encrypted_dek

    # Remaining bytes should be the encrypted data
    assert result[4 + expected_length :] == encrypted_data


def test_unpack_envelope_ciphertext():
    """Test unpacking combined ciphertext into encrypted DEK and data."""
    encrypted_dek = b"fake_encrypted_dek_67890"
    encrypted_data = b"fake_encrypted_data_payload"

    # Pack the data first
    packed = _pack_envelope_ciphertext(encrypted_dek, encrypted_data)

    # Unpack and verify
    unpacked_dek, unpacked_data = _unpack_envelope_ciphertext(packed)
    assert unpacked_dek == encrypted_dek
    assert unpacked_data == encrypted_data


def test_unpack_envelope_ciphertext_empty_data():
    """Test unpacking when encrypted data is empty."""
    encrypted_dek = b"some_dek"
    encrypted_data = b""

    packed = _pack_envelope_ciphertext(encrypted_dek, encrypted_data)
    unpacked_dek, unpacked_data = _unpack_envelope_ciphertext(packed)

    assert unpacked_dek == encrypted_dek
    assert unpacked_data == encrypted_data


def test_unpack_envelope_ciphertext_empty_dek():
    """Test unpacking when encrypted DEK is empty."""
    encrypted_dek = b""
    encrypted_data = b"some_data"

    packed = _pack_envelope_ciphertext(encrypted_dek, encrypted_data)
    unpacked_dek, unpacked_data = _unpack_envelope_ciphertext(packed)

    assert unpacked_dek == encrypted_dek
    assert unpacked_data == encrypted_data


def test_unpack_envelope_ciphertext_too_short():
    """Test unpacking fails when ciphertext is too short."""
    with pytest.raises(ValueError, match="Ciphertext too short"):
        _unpack_envelope_ciphertext(b"abc")  # Less than 4 bytes


def test_unpack_envelope_ciphertext_invalid_format():
    """Test unpacking fails when ciphertext format is invalid."""
    # Create a ciphertext that claims DEK length is longer than available data
    fake_length = (1000).to_bytes(4, "little")  # Claims 1000 bytes for DEK
    short_data = b"only_short_data"
    invalid_ciphertext = fake_length + short_data

    with pytest.raises(ValueError, match="Ciphertext format invalid"):
        _unpack_envelope_ciphertext(invalid_ciphertext)


@pytest.mark.parametrize(
    "encrypted_dek,encrypted_data",
    [
        (b"", b""),
        (b"short_dek", b"short_data"),
        (b"longer_encrypted_dek_content", b"longer_encrypted_data_content"),
        (b"dek_with_nulls\x00\x00", b"data_with_nulls\x00\x00"),
        (b"dek_with_nulls\x00\x00trailer", b"data_with_nulls\x00\x00trailer"),
        (b"\xff" * 100, b"\x00" * 200),  # Binary data
    ],
    ids=itertools.count(),
)
def test_pack_unpack_roundtrip(encrypted_dek, encrypted_data):
    """Test that packing and unpacking are inverse operations."""
    packed = _pack_envelope_ciphertext(encrypted_dek, encrypted_data)
    unpacked_dek, unpacked_data = _unpack_envelope_ciphertext(packed)

    assert unpacked_dek == encrypted_dek
    assert unpacked_data == encrypted_data


def test_pack_envelope_ciphertext_large_dek():
    """Test packing with a large encrypted DEK."""
    large_dek = b"x" * 10000
    small_data = b"small"

    result = _pack_envelope_ciphertext(large_dek, small_data)

    # Verify the length is correctly encoded
    assert result[:4] == (10000).to_bytes(4, "little")
    assert result[4:10004] == large_dek
    assert result[10004:] == small_data


# Integration tests
@pytest.mark.integration
@pytest.mark.parametrize(
    "plaintext,aad",
    [
        (b"", b""),
        (b"", b"some_aad"),
        (b"short", b""),
        (b"short", b"some_aad"),
        (b"x" * 1000, b""),
        (b"x" * 1000, b"y" * 1000),
        (b"binary\x00\xff\x01data", b"binary\x00aad"),
    ],
    ids=itertools.count(),
)
@skip_unless_gcp_available
def test_encrypt_decrypt_roundtrip(gcp_kms_provider, plaintext, aad):
    """Test encrypt/decrypt roundtrip with various data sizes and AAD."""
    ciphertext = gcp_kms_provider.encrypt(plaintext, aad)
    assert len(ciphertext) > 4 + 32  # at least EDEK size plus key

    if plaintext:
        assert ciphertext != plaintext

    decrypted = gcp_kms_provider.decrypt(ciphertext, aad)
    assert decrypted == plaintext


@pytest.mark.integration
@skip_unless_gcp_available
def test_mismatched_aad_raises_nacl_error(gcp_kms_provider):
    """Test that decryption fails with mismatched AAD."""
    plaintext = b"sensitive_data"
    correct_aad = b"correct_aad"
    wrong_aad = b"wrong_aad"

    ciphertext = gcp_kms_provider.encrypt(plaintext, correct_aad)

    with pytest.raises(nacl.exceptions.CryptoError, match=r"Decryption failed[.]"):
        gcp_kms_provider.decrypt(ciphertext, wrong_aad)


@pytest.mark.integration
@skip_unless_gcp_available
def test_dek_decrypt_failure(gcp_kms_provider):
    ciphertext = gcp_kms_provider.encrypt(b"message", b"aad")
    ciphertext = (
        ciphertext[:4] + ciphertext[5:9:-1] + ciphertext[9:]
    )  # swap a few bytes of encrypted DEK

    with pytest.raises(
        google.api_core.exceptions.InvalidArgument, match="400 Decryption failed"
    ):
        gcp_kms_provider.decrypt(ciphertext, b"aad")


@pytest.mark.integration
@skip_unless_gcp_available
def test_provider_name(gcp_kms_provider):
    """Test that provider returns correct name."""
    assert gcp_kms_provider.name() == "gcpkms"


@pytest.mark.integration
@skip_unless_gcp_available
def test_multiple_encryptions_produce_different_ciphertexts(gcp_kms_provider):
    """Test that encrypting the same data twice produces different ciphertexts."""
    plaintext = b"test_data"
    aad = b"test_aad"

    ciphertext1 = gcp_kms_provider.encrypt(plaintext, aad)
    ciphertext2 = gcp_kms_provider.encrypt(plaintext, aad)

    # Should be different due to random DEK generation
    assert ciphertext1 != ciphertext2

    # But both should decrypt to the same plaintext
    assert gcp_kms_provider.decrypt(ciphertext1, aad) == plaintext
    assert gcp_kms_provider.decrypt(ciphertext2, aad) == plaintext
