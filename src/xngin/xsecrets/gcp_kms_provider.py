import base64
import dataclasses
import json
import os
import struct

import nacl.secret
import nacl.utils
from google.auth.credentials import Credentials
from google.cloud import kms
from google.oauth2 import service_account
from loguru import logger

from xngin.xsecrets import constants
from xngin.xsecrets.exceptions import InvalidSecretStoreConfigurationError
from xngin.xsecrets.provider import Provider, Registry

NAME = "gcpkms"


def _pack_envelope_ciphertext(encrypted_dek: bytes, encrypted_data: bytes) -> bytes:
    """Packs encrypted DEK and data into combined ciphertext.

    Format: [4 bytes: encrypted_dek_length][encrypted_dek][encrypted_data]
    """
    encrypted_dek_length = len(encrypted_dek)
    return struct.pack("<I", encrypted_dek_length) + encrypted_dek + encrypted_data


def _unpack_envelope_ciphertext(ct: bytes) -> tuple[bytes, bytes]:
    """Unpacks combined ciphertext into encrypted DEK and encrypted data.

    Returns tuple of (encrypted_dek, encrypted_data)
    """
    if len(ct) < 4:
        raise ValueError("Ciphertext too short")

    # Extract encrypted DEK length
    encrypted_dek_length = struct.unpack("<I", ct[:4])[0]

    if len(ct) < 4 + encrypted_dek_length:
        raise ValueError("Ciphertext format invalid")

    # Extract encrypted DEK and encrypted data
    encrypted_dek = ct[4 : 4 + encrypted_dek_length]
    encrypted_data = ct[4 + encrypted_dek_length :]

    return encrypted_dek, encrypted_data


@dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
class GcpKmsConfiguration:
    key_name: str
    credentials: Credentials


def _read_gcp_env() -> GcpKmsConfiguration | None:
    credentials = os.environ.get(constants.ENV_XNGIN_SECRETS_GCP_CREDENTIALS)
    key_uri = os.environ.get(constants.ENV_XNGIN_SECRETS_GCP_KMS_KEY_URI)
    if credentials and key_uri:
        if not key_uri.startswith("gcp-kms://"):
            raise InvalidSecretStoreConfigurationError(
                f"{constants.ENV_XNGIN_SECRETS_GCP_KMS_KEY_URI} must be prefixed with 'gcp-kms://'."
            )
        if "cryptoKeyVersions" in key_uri:
            raise InvalidSecretStoreConfigurationError(
                f"/cryptoKeyVersions/ suffix in {constants.ENV_XNGIN_SECRETS_GCP_KMS_KEY_URI} must be removed."
            )

        key_name = key_uri[len("gcp-kms://") :]

        logger.info(
            f"Secrets: GCP credentials and key URI configured from environment: {key_uri}"
        )
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(base64.standard_b64decode(credentials))
        )
        return GcpKmsConfiguration(credentials=credentials, key_name=key_name)
    return None


def initialize(registry: Registry):
    params = _read_gcp_env()
    if params:
        instance = GcpKmsProvider(
            key_name=params.key_name, credentials=params.credentials
        )
        registry.register(instance.name(), instance)


class GcpKmsProvider(Provider):
    """Implements envelope encryption using Google Cloud KMS and pynacl AEAD.

    This provider generates a random data encryption key (DEK), encrypts the plaintext
    with the DEK using nacl.secret.Aead, then encrypts the DEK using GCP KMS.
    The final ciphertext contains both the encrypted DEK and the encrypted data.
    """

    def __init__(self, *, key_name: str, credentials: Credentials):
        """Constructs a GcpKmsProvider.

        :param key_name: GCP KMS key name (e.g. projects/.../locations/.../keyRings/.../cryptoKeys/...)
        :param credentials: GCP service account credentials
        """
        self.key_name = key_name
        self.kms_client = kms.KeyManagementServiceClient(credentials=credentials)

    def name(self) -> str:
        return NAME

    def encrypt(self, pt: bytes, aad: bytes) -> bytes:
        """Encrypts plaintext using envelope encryption."""
        dek = nacl.utils.random(nacl.secret.Aead.KEY_SIZE)
        aead = nacl.secret.Aead(dek)
        encrypted_data = aead.encrypt(pt, aad)

        request = kms.EncryptRequest(name=self.key_name, plaintext=dek)
        response = self.kms_client.encrypt(request)
        encrypted_dek = response.ciphertext

        return _pack_envelope_ciphertext(encrypted_dek, encrypted_data)

    def decrypt(self, ct: bytes, aad: bytes) -> bytes:
        """Decrypts ciphertext using envelope encryption."""
        encrypted_dek, encrypted_data = _unpack_envelope_ciphertext(ct)

        request = kms.DecryptRequest(name=self.key_name, ciphertext=encrypted_dek)
        response = self.kms_client.decrypt(request)
        dek = response.plaintext

        aead = nacl.secret.Aead(dek)
        return aead.decrypt(encrypted_data, aad)
