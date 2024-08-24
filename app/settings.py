from functools import lru_cache
from typing import Literal, List

from pydantic import BaseModel, PositiveInt, SecretStr
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    JsonConfigSettingsSource,
)


@lru_cache
def get_settings_for_server() -> "XnginSettings":
    """Constructs an XnginSettings for use by the API server."""
    return SettingsForServer(_secrets_dir="secrets")


class PostgresDsn(BaseModel):
    user: str
    port: PositiveInt
    host: str
    password: SecretStr
    dbname: str
    sslmode: Literal[
        "disable", "allow", "prefer", "require", "verify-ca", "verify-full"
    ]


class RocketLearningSettings(BaseModel):
    dwh: PostgresDsn


class XnginSettings(BaseSettings):
    customer: RocketLearningSettings
    trusted_ips: List[str] = list()


class SettingsForServer(XnginSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[XnginSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            JsonConfigSettingsSource(settings_cls, json_file="xngin.settings.json"),
            init_settings,
            file_secret_settings,
        )


class SettingsForTesting(XnginSettings):
    pass
