import json
import os
from functools import lru_cache
from typing import Literal, List, Union

import sqlalchemy
from pydantic import BaseModel, PositiveInt, SecretStr, Field
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.orm import Session

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
    # worksheet is the name of the worksheet. This is usually the name of the database warehouse table.
    worksheet: str


class Unit(BaseModel):
    """Units are a logical representation of a table in the data warehouse.

    Units are defined by a table_name and a configuration worksheet.
    """

    table_name: str
    sheet: SheetRef


class UnitsMixin(BaseModel):
    units: List[Unit]

    def find_unit(self, unit_type: str):
        found = next(
            (u for u in self.units if u.table_name.lower() == unit_type.lower()), None
        )
        if found is None:
            raise CannotFindUnitException(unit_type)
        return found


class RocketLearningConfig(UnitsMixin, BaseModel):
    """

    TODO: implement dbsession(self, unit_type)
    """

    type: Literal["customer"]
    dwh: PostgresDsn
    api_host: str
    api_token: SecretStr

    def to_sqlalchemy_url_and_table(self, unit_type: str) -> SqlalchemyAndTable:
        unit = self.find_unit(unit_type)
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
            table_name=unit.table_name,
        )


class SqliteLocalConfig(UnitsMixin, BaseModel):
    type: Literal["sqlite_local"]
    sqlite_filename: str

    def to_sqlalchemy_url_and_table(self, unit_type: str) -> SqlalchemyAndTable:
        """Returns a tuple of SQLAlchemy URL and a table name."""
        unit = self.find_unit(unit_type)
        return SqlalchemyAndTable(
            sqlalchemy_url=str(
                sqlalchemy.URL.create(
                    drivername="sqlite",
                    database=self.sqlite_filename,
                    query={"mode": "ro"},
                )
            ),
            table_name=unit.table_name,
        )

    def dbsession(self, unit_type: str):
        """Returns a Session to be used to send queries to the customer database.

        Use this in a `with` block to ensure correct transaction handling. If you need the
        sqlalchemy Engine, call .get_bind().
        """
        return Session(
            sqlite_connect(self.to_sqlalchemy_url_and_table(unit_type).sqlalchemy_url)
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


class CannotFindTableException(Exception):
    def __init__(self, table_name, existing_tables):
        self.table_name = table_name
        self.alternatives = existing_tables
        if existing_tables:
            self.message = f"The table '{table_name}' does not exist. Known tables: {", ".join(sorted(existing_tables))}"
        else:
            self.message = f"The table '{table_name}' does not exist; the database does not contain any tables."

    def __str__(self):
        return self.message


class CannotFindUnitException(Exception):
    def __init__(self, unit_name):
        self.unit_name = unit_name
        self.message = (
            f"The unit {unit_name} does not exist. Check the configuration files."
        )

    def __str__(self):
        return self.message


def get_sqlalchemy_table_from_engine(engine: sqlalchemy.engine.Engine, table_name: str):
    """Constructs a Table via reflection.

    Raises CannotFindTheTableException containing helpful error message if the table doesn't exist.
    """
    metadata = sqlalchemy.MetaData()
    try:
        return sqlalchemy.Table(table_name, metadata, autoload_with=engine)
    except NoSuchTableError as nste:
        metadata.reflect(engine)
        existing_tables = metadata.tables.keys()
        raise CannotFindTableException(table_name, existing_tables) from nste


def sqlite_connect(sqlalchemy_url):
    connect_args = {}
    if sqlalchemy_url.startswith("postgres"):
        connect_args["connect_timeout"] = 5
    elif sqlalchemy_url.startswith("sqlite"):
        connect_args["timeout"] = 5
    return sqlalchemy.create_engine(sqlalchemy_url, connect_args=connect_args)
