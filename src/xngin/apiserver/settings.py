import json
import os
from functools import lru_cache
from typing import Literal, List, Union

import sqlalchemy
from pydantic import BaseModel, PositiveInt, SecretStr, Field
from sqlalchemy.exc import NoSuchTableError

DEFAULT_SECRETS_DIRECTORY = "secrets"
DEFAULT_SETTINGS_FILE = "xngin.settings.json"


@lru_cache
def get_settings_for_server():
    """Constructs an XnginSettings for use by the API server."""
    with open(os.environ.get("XNGIN_SETTINGS", DEFAULT_SETTINGS_FILE)) as f:
        settings_raw = json.load(f)
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


class SqlalchemyAndTable(BaseModel):
    sqlalchemy_url: str
    table_name: str


class SheetRef(BaseModel):
    url: str
    worksheet: str


class RocketLearningConfig(BaseModel):
    type: Literal["customer"]
    table_name: str
    sheet: SheetRef
    dwh: PostgresDsn
    api_host: str
    api_token: SecretStr

    def to_sqlalchemy_url_and_table(self) -> SqlalchemyAndTable:
        return SqlalchemyAndTable(
            sqlalchemy_url=str(
                sqlalchemy.URL.create(
                    drivername="postgresql+psycopg2",
                    username=self.dwh.user,
                    password=self.dwh.password.get_secret_value(),
                    host=self.dwh.host,
                    port=self.dwh.port,
                    database=self.dwh.dbname,
                    query={"sslmode": self.dwh.sslmode},
                )
            ),
            table_name=self.table_name,
        )


class SqliteLocalConfig(BaseModel):
    type: Literal["sqlite_local"]
    table_name: str
    sheet: SheetRef
    sqlite_filename: str

    def to_sqlalchemy_url_and_table(self) -> SqlalchemyAndTable:
        """Returns a tuple of SQLAlchemy URL and a table name."""
        return SqlalchemyAndTable(
            sqlalchemy_url=str(
                sqlalchemy.URL.create(
                    drivername="sqlite",
                    database=self.sqlite_filename,
                    query={"mode": "ro"},
                )
            ),
            table_name=self.table_name,
        )


class ClientConfig(BaseModel):
    id: str
    config: Union[RocketLearningConfig, SqliteLocalConfig] = Field(
        ..., discriminator="type"
    )


class XnginSettings(BaseModel):
    trusted_ips: List[str] = list()
    db_connect_timeout_secs: int = 3
    client_configs: List[ClientConfig]

    def get_client_config(self, config_id):
        """Finds the config for a specific ID if it exists, or returns None."""
        for config in self.client_configs:
            if config.id == config_id:
                return config
        return None


class SettingsForTesting(XnginSettings):
    pass


class CannotFindTheTableException(Exception):
    def __init__(self, table_name, existing_tables):
        self.table_name = table_name
        self.alternatives = existing_tables
        if existing_tables:
            self.message = f"The {table_name} table does not exist. Known tables: {", ".join(sorted(existing_tables))}"
        else:
            self.message = "The specified database does not contain any tables. Check the DSN and try again."

    def __str__(self):
        return self.message


def get_sqlalchemy_table(sqlat: SqlalchemyAndTable):
    """Connects to a SQLAlchemy DSN and creates a sqlalchemy.Table for introspection."""
    connect_args = {}
    if sqlat.sqlalchemy_url.startswith("postgres"):
        connect_args["connect_timeout"] = 5
    elif sqlat.sqlalchemy_url.startswith("sqlite"):
        connect_args["timeout"] = 5
    engine = sqlalchemy.create_engine(sqlat.sqlalchemy_url, connect_args=connect_args)
    metadata = sqlalchemy.MetaData()
    try:
        return sqlalchemy.Table(sqlat.table_name, metadata, autoload_with=engine)
    except NoSuchTableError as nste:
        metadata.reflect(engine)
        existing_tables = metadata.tables.keys()
        raise CannotFindTheTableException(sqlat.table_name, existing_tables) from nste
