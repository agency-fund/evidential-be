import json
import os
from functools import lru_cache
from typing import Literal

import sqlalchemy
from pydantic import BaseModel, PositiveInt, SecretStr, Field, field_validator
from sqlalchemy import event
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.orm import Session

from xngin.sqlite_extensions import NumpyStddev

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


class Participant(BaseModel):
    """Participants are a logical representation of a table in the data warehouse.

    Participants are defined by a table_name and a configuration worksheet.
    """

    table_name: str
    sheet: SheetRef


class ParticipantsMixin(BaseModel):
    """ParticipantsMixin can be added to a config type to add standardized participant definitions."""

    participants: list[Participant]

    def find_participants(self, participant_type: str):
        found = next(
            (
                u
                for u in self.participants
                if u.table_name.lower() == participant_type.lower()
            ),
            None,
        )
        if found is None:
            raise CannotFindParticipantsException(participant_type)
        return found


class WebhookUrl(BaseModel):
    """Represents a url and HTTP method to use with it."""

    method: Literal["get", "post", "put", "patch"]
    url: str
    # headers: dict[str, str]

    @field_validator("method", mode="before")
    @classmethod
    def to_lower(cls, value):
        """Force the http 'method' to be lowercase before validation."""

        return str(value).lower().strip()


class WebhookActions(BaseModel):
    """The set of supported actions that trigger a user callback."""

    # No action is required, so a user can leave it out completely.
    commit: WebhookUrl | None = None
    assignment_file: WebhookUrl | None = None
    update_timestamps: WebhookUrl | None = None
    update_description: WebhookUrl | None = None


class WebhookCommonHeaders(BaseModel):
    """Enumerates supported headers to attach to all webhook requests."""

    authorization: str | None


class WebhookConfig(BaseModel):
    """Top-level configuration object for user-defined webhooks."""

    actions: WebhookActions
    common_headers: WebhookCommonHeaders


class WebhookMixin(BaseModel):
    """Add this to a config type to support using webhooks to persist changes.

    Example:
    "webhook_config": {
      "actions": {
        "commit": {
          "method": "POST",
          "url": "http://localhost:4001/dev/api/v1/experiment-commit/save-experiment-commit"
        },
        "assignment_file": {
          "method": "POST",
          "url": "http://localhost:4001/dev/api/v1/experiment-commit/get-file-name-by-experiment-id/{experimentId}?qs={experimentCode}",
        }
      },
      "common_headers": {
        "authorization": "abc"
      }
    }
    """

    webhook_config: WebhookConfig


class RocketLearningConfig(ParticipantsMixin, WebhookMixin, BaseModel):
    """

    TODO: implement dbsession(self, participant_type)
    """

    type: Literal["customer"]

    dwh: PostgresDsn
    api_host: str
    api_token: SecretStr

    def to_sqlalchemy_url_and_table(self, participant_type: str) -> SqlalchemyAndTable:
        participants = self.find_participants(participant_type)
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
            table_name=participants.table_name,
        )


class SqliteLocalConfig(ParticipantsMixin, BaseModel):
    type: Literal["sqlite_local"]
    sqlite_filename: str

    def to_sqlalchemy_url_and_table(self, participant_type: str) -> SqlalchemyAndTable:
        """Returns a tuple of SQLAlchemy URL and a table name."""
        participants = self.find_participants(participant_type)
        return SqlalchemyAndTable(
            sqlalchemy_url=str(
                sqlalchemy.URL.create(
                    drivername="sqlite",
                    database=self.sqlite_filename,
                    query={"mode": "ro"},
                )
            ),
            table_name=participants.table_name,
        )

    def dbsession(self, participant_type: str):
        """Returns a Session to be used to send queries to the customer database.

        Use this in a `with` block to ensure correct transaction handling. If you need the
        sqlalchemy Engine, call .get_bind().
        """
        url = self.to_sqlalchemy_url_and_table(participant_type).sqlalchemy_url
        engine = sqlalchemy_connect(url)
        if url.startswith("sqlite://"):

            @event.listens_for(engine, "connect")
            def register_sqlite_functions(dbapi_connection, _):
                NumpyStddev.register(dbapi_connection)

        return Session(engine)


type ClientConfigType = RocketLearningConfig | SqliteLocalConfig


class ClientConfig(BaseModel):
    id: str
    config: ClientConfigType = Field(..., discriminator="type")


class XnginSettings(BaseModel):
    trusted_ips: list[str] = Field(default_factory=list)
    db_connect_timeout_secs: int = 3
    client_configs: list[ClientConfig]

    def get_client_config(self, config_id):
        """Finds the config for a specific ID if it exists, or returns None."""
        for config in self.client_configs:
            if config.id == config_id:
                return config
        return None


class SettingsForTesting(XnginSettings):
    pass


class CannotFindTableException(Exception):
    """Raised when we cannot find a table in the database."""

    def __init__(self, table_name, existing_tables):
        self.table_name = table_name
        self.alternatives = existing_tables
        if existing_tables:
            self.message = f"The table '{table_name}' does not exist. Known tables: {', '.join(sorted(existing_tables))}"
        else:
            self.message = f"The table '{table_name}' does not exist; the database does not contain any tables."

    def __str__(self):
        return self.message


class CannotFindParticipantsException(Exception):
    """Raised when we cannot find a participant in the configuration."""

    def __init__(self, participant_type):
        self.participant_type = participant_type
        self.message = f"The configuration for participant type {participant_type} does not exist. Check the configuration files."

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


def sqlalchemy_connect(sqlalchemy_url):
    """Connect to a database, given a SQLAlchemy-compatible URL.

    This is intended to be used to connect to customer databases.
    """
    connect_args = {}
    if sqlalchemy_url.startswith("postgres"):
        connect_args["connect_timeout"] = 5
    elif sqlalchemy_url.startswith("sqlite"):
        connect_args["timeout"] = 5
    return sqlalchemy.create_engine(sqlalchemy_url, connect_args=connect_args)
