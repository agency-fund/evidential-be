import configparser
import dataclasses
import os
import tempfile

from tink import aead
from tink.integration import awskms

from xngin.xsecrets import constants
from xngin.xsecrets.kms_provider import KmsProvider
from xngin.xsecrets.provider import Registry

NAME = "awskms"


@dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
class AwsKmsConfiguration:
    key_uri: str
    credentials_path: str


def _read_aws_env():
    key_uri = os.environ.get(constants.ENV_XNGIN_SECRETS_AWS_KEY_URI)
    access_key_id = os.environ.get(constants.ENV_XNGIN_SECRETS_AWS_ACCESS_KEY_ID)
    secret_access_key = os.environ.get(
        constants.ENV_XNGIN_SECRETS_AWS_SECRET_ACCESS_KEY
    )
    if not (key_uri and access_key_id and secret_access_key):
        return None

    # Write out the config file that Tink wants its credentials in.
    config = configparser.ConfigParser()
    config["default"] = {
        "aws_access_key_id": access_key_id,
        "aws_secret_access_key": secret_access_key,
    }
    temp_file = tempfile.NamedTemporaryFile(delete=False)  # noqa: SIM115
    credentials_path = temp_file.name
    with open(credentials_path, "w") as credentials_file:
        config.write(credentials_file)
    return AwsKmsConfiguration(key_uri=key_uri, credentials_path=credentials_path)


def initialize(registry: Registry):
    params = _read_aws_env()
    if params:
        aead.register()
        client = awskms.AwsKmsClient(
            key_uri=None, credentials_path=params.credentials_path
        )
        remote_aead = client.get_aead(params.key_uri)
        instance = KmsProvider(variant=NAME, remote_aead=remote_aead)
        registry.register(instance.name(), instance)
