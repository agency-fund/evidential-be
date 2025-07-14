import base64
import dataclasses
import json
import os

from google.auth.credentials import Credentials
from google.oauth2 import service_account
from loguru import logger
from tink import aead
from tink.integration import gcpkms

from xngin.xsecrets import constants
from xngin.xsecrets.exceptions import InvalidSecretStoreConfigurationError
from xngin.xsecrets.kms_provider import KmsProvider
from xngin.xsecrets.provider import Registry

NAME = "gcpkms"


@dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
class GcpKmsConfiguration:
    key_uri: str
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
                f"/cryptoKeyVersions/ suffix in {constants.ENV_XNGIN_SECRETS_GCP_KMS_KEY_URI} should be removed because"
                f"of a bug in Tink (https://github.com/tink-crypto/tink-py/issues/31)."
            )

        logger.info(
            f"Secrets: GCP credentials and key URI configured from environment: {key_uri}"
        )
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(base64.standard_b64decode(credentials))
        )
        return GcpKmsConfiguration(credentials=credentials, key_uri=key_uri)
    return None


def initialize(registry: Registry):
    params = _read_gcp_env()
    if params:
        aead.register()
        client = gcpkms.GcpKmsClient(key_uri=None, credentials=params.credentials)
        remote_aead = client.get_aead(params.key_uri)
        instance = KmsProvider(variant=NAME, remote_aead=remote_aead)
        registry.register(instance.name(), instance)
