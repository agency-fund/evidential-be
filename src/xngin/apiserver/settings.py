import json
import os
from functools import lru_cache
from typing import Literal, List

from pydantic import BaseModel, PositiveInt, SecretStr

from xngin.apiserver.utils import merge_dicts

DEFAULT_SECRETS_DIRECTORY = "secrets"
DEFAULT_SETTINGS_FILE = "xngin.settings.json"


@lru_cache
def get_settings_for_server():
    """Constructs an XnginSettings for use by the API server."""
    with open(os.environ.get("XNGIN_SETTINGS", DEFAULT_SETTINGS_FILE)) as f:
        settings_raw = json.load(f)

    # Also load supplemental values from the secrets/ directory.
    for root, _, files in os.walk(
        os.environ.get("XNGIN_SECRETS", DEFAULT_SECRETS_DIRECTORY)
    ):
        for key in files:
            if key not in XnginSettings.model_fields:
                continue
            with open(os.path.join(root, key), "r") as f:
                value = json.load(f)
            if isinstance(settings_raw.get(key), dict):
                settings_raw[key] = merge_dicts(settings_raw.get(key), value)
            else:
                settings_raw[key] = value
    return XnginSettings.model_validate(settings_raw)


class PostgresDsn(BaseModel):
    user: str
    port: PositiveInt = 5432
    host: str
    password: SecretStr
    dbname: str
    sslmode: Literal[
        "disable", "allow", "prefer", "require", "verify-ca", "verify-full"
    ]


class RocketLearningSettings(BaseModel):
    dwh: PostgresDsn
    api_host: str
    api_token: SecretStr


class XnginSettings(BaseModel):
    customer: RocketLearningSettings
    trusted_ips: List[str] = list()
    db_connect_timeout_secs: int = 3


class SettingsForTesting(XnginSettings):
    pass
