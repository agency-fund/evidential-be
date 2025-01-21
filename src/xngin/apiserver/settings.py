import json
import logging
import os
from collections import Counter
from functools import lru_cache
from typing import Literal, Annotated, Protocol
from urllib.parse import urlparse

import httpx
import sqlalchemy
from httpx import codes
from pydantic import (
    BaseModel,
    PositiveInt,
    SecretStr,
    Field,
    field_validator,
    ConfigDict,
    model_validator,
)
from sqlalchemy import Engine, event, text
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    wait_random,
    retry_if_not_exception_type,
    stop_after_delay,
)

from xngin.apiserver.settings_secrets import replace_secrets
from xngin.db_extensions import NumpyStddev

DEFAULT_SETTINGS_FILE = "xngin.settings.json"

logger = logging.getLogger(__name__)


class UnclassifiedRemoteSettingsError(Exception):
    """Raised when we fail to fetch remote settings for an unclassified reason."""

    pass


class RemoteSettingsClientError(Exception):
    """Raised when we fail to fetch remote settings due to our misconfiguration."""

    pass


@lru_cache
def get_settings_for_server():
    """Constructs an XnginSettings for use by the API server."""
    settings_path = os.environ.get("XNGIN_SETTINGS", DEFAULT_SETTINGS_FILE)

    if settings_path.startswith("https://"):
        settings_raw = get_remote_settings(settings_path)
    else:
        logger.info("Loading XNGIN_SETTINGS from disk: %s", settings_path)
        with open(settings_path) as f:
            settings_raw = json.load(f)
    settings_raw = replace_secrets(settings_raw)
    return XnginSettings.model_validate(settings_raw)


@retry(
    reraise=True,
    retry=retry_if_not_exception_type(RemoteSettingsClientError),
    stop=stop_after_delay(15),
    wait=wait_random(1, 3),
)
def get_remote_settings(url):
    """Fetches the settings from a remote URL.

    Retries: Requests that take more than 5 seconds, or that respond with a server error, will be retried. We do not
    retry errors that look like misconfigurations on our side (e.g. 404s).
    """
    parsed = urlparse(url)
    headers: dict = {}
    if auth := os.environ.get("XNGIN_SETTINGS_AUTHORIZATION"):
        headers["Authorization"] = auth.strip()
    if parsed.hostname == "api.github.com" and parsed.path.startswith("/repos"):
        headers["Accept"] = "application/vnd.github.v3.raw"
    logger.info("Loading XNGIN_SETTINGS from URL: %s", url)
    retrying_transport = httpx.HTTPTransport(retries=2)
    with httpx.Client(
        transport=retrying_transport, headers=headers, timeout=5
    ) as client:
        response = client.get(url)
        status = response.status_code
        if status == codes.OK:
            return response.json()
        if codes.is_client_error(status):
            raise RemoteSettingsClientError(f"{status}: {url}")
        raise UnclassifiedRemoteSettingsError(f"{status} {response.text}")


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SheetRef(ConfigBaseModel):
    url: str
    # worksheet is the name of the worksheet. This is usually the name of the database warehouse table.
    worksheet: str


class ParticipantsRef(ConfigBaseModel):
    """Participants are a logical representation of a table in the data warehouse.

    Participants are defined by a table_name and a configuration worksheet.
    """

    participant_type: str
    table_name: str
    sheet: SheetRef


class ParticipantsMixin(ConfigBaseModel):
    """ParticipantsMixin can be added to a config type to add standardized participant definitions."""

    participants: list[ParticipantsRef]

    def find_participants(self, participant_type: str):
        """Returns the participant matching participant_type or raises CannotFindParticipantsException."""
        found = next(
            (
                u
                for u in self.participants
                if u.participant_type.lower() == participant_type.lower()
            ),
            None,
        )
        if found is None:
            raise CannotFindParticipantsError(participant_type)
        return found

    @model_validator(mode="after")
    def check_unique_participant_types(self):
        counted = Counter([
            participant.participant_type for participant in self.participants
        ])
        duplicates = [item for item, count in counted.items() if count > 1]
        if duplicates:
            raise ValueError(
                f"Participants with conflicting identifiers found: {', '.join(duplicates)}."
            )
        return self


class WebhookUrl(ConfigBaseModel):
    """Represents a url and HTTP method to use with it."""

    method: Literal["get", "post", "put", "patch"]
    url: str

    # headers: dict[str, str]

    @field_validator("method", mode="before")
    @classmethod
    def to_lower(cls, value):
        """Force the http 'method' to be lowercase before validation."""

        return str(value).lower().strip()


class WebhookActions(ConfigBaseModel):
    """The set of supported actions that trigger a user callback."""

    # No action is required, so a user can leave it out completely.
    commit: WebhookUrl | None = None
    assignment_file: WebhookUrl | None = None
    update_timestamps: WebhookUrl | None = None
    update_description: WebhookUrl | None = None


class WebhookCommonHeaders(ConfigBaseModel):
    """Enumerates supported headers to attach to all webhook requests."""

    authorization: SecretStr | None


class WebhookConfig(ConfigBaseModel):
    """Top-level configuration object for user-defined webhooks."""

    actions: WebhookActions
    common_headers: WebhookCommonHeaders


class ToSqlalchemyUrl(Protocol):
    def to_sqlalchemy_url(self) -> sqlalchemy.URL:
        """Creates a sqlalchemy.URL from this Dsn."""
        ...


class BaseDsn:
    def is_redshift(self):
        """Return true iff the hostname indicates that this is connecting to Redshift."""
        return False

    def supports_table_reflection(self):
        return not self.is_redshift()


class GcpServiceAccountInfo(ConfigBaseModel):
    """Describes a Google Cloud Service Account credential."""

    type: Literal["serviceaccountinfo"]
    content_base64: Annotated[
        str,
        Field(
            ...,
            description="The base64-encoded service account info in the canonical JSON form.",
        ),
    ]


class GcpServiceAccountFile(ConfigBaseModel):
    """Describes a file path to a Google Cloud Service Account credential file."""

    type: Literal["serviceaccountfile"]
    path: Annotated[
        str,
        Field(
            ...,
            description="The path to the service account credentials file containing the credentials "
            "in canonical JSON form.",
        ),
    ]


class BqDsn(ConfigBaseModel, BaseDsn):
    """Describes a BigQuery connection."""

    driver: Literal["bigquery"]
    project_id: Annotated[
        str,
        Field(..., description="The Google Cloud Project ID containing the dataset."),
    ]
    dataset_id: Annotated[str, Field(..., description="The dataset name.")]

    # These two authentication modes are documented here:
    # https://googleapis.dev/python/google-api-core/latest/auth.html#service-accounts
    credentials: Annotated[
        GcpServiceAccountInfo | GcpServiceAccountFile,
        Field(
            ...,
            discriminator="type",
            description="The Google Cloud Service Account credentials.",
        ),
    ]

    def to_sqlalchemy_url(self) -> sqlalchemy.URL:
        qopts = {}
        if self.credentials.type == "serviceaccountinfo":
            qopts["credentials_base64"] = self.credentials.content_base64
        elif self.credentials.type == "serviceaccountfile":
            qopts["credentials_path"] = self.credentials.path
        return sqlalchemy.URL.create(
            drivername="bigquery",
            host=self.project_id,
            database=self.dataset_id,
            query=qopts,
        )


class Dsn(ConfigBaseModel, BaseDsn):
    """Describes a set of parameters suitable for connecting to most types of remote databases."""

    driver: Literal[
        "postgresql+psycopg",  # Preferred for most Postgres-compatible databases.
        "postgresql+psycopg2",  # Use with: Redshift
    ]
    host: str
    port: PositiveInt = 5432
    user: str
    password: SecretStr
    dbname: str
    sslmode: (
        Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
        | None
    ) = None
    # Specify the order in which schemas are searched if your dwh supports it.
    search_path: str | None = None

    def is_redshift(self):
        return self.host.endswith("redshift.amazonaws.com")

    def to_sqlalchemy_url(self):
        url = sqlalchemy.URL.create(
            drivername=self.driver,
            username=self.user,
            password=self.password.get_secret_value(),
            host=self.host,
            port=self.port,
            database=self.dbname,
        )
        if self.driver.startswith("postgresql"):
            query = dict(url.query)
            query.update({
                "sslmode": self.sslmode if self.sslmode else "verify-full",
                # re: redshift issue https://github.com/psycopg/psycopg/issues/122#issuecomment-985742751
                "client_encoding": "utf-8",
            })
            url = url.set(query=query)
        return url


class DbapiArg(ConfigBaseModel):
    """Describes a DBAPI connect() argument.

    These can be arbitrary kv pairs and are database-driver specific."""

    arg: str
    value: str


class RemoteDatabaseConfig(ParticipantsMixin, ConfigBaseModel):
    """RemoteDatabaseConfig defines a configuration for a remote data warehouse."""

    webhook_config: WebhookConfig | None = None

    type: Literal["remote"]

    dwh: Annotated[Dsn | BqDsn, Field(discriminator="driver")]

    def supports_reflection(self):
        return self.dwh.supports_table_reflection()

    def dbsession(self):
        """Returns a Session to be used to send queries to the customer database.

        Use this in a `with` block to ensure correct transaction handling. If you need the
        sqlalchemy Engine, call .get_bind().
        """
        url = self.dwh.to_sqlalchemy_url()
        connect_args: dict = {}
        if url.get_backend_name() == "postgres":
            connect_args["connect_timeout"] = 5
        engine = sqlalchemy.create_engine(
            url,
            connect_args=connect_args,
            echo=os.environ.get("ECHO_SQL", "").lower() in ("true", "1"),
        )
        self._extra_engine_setup(engine)
        return Session(engine)

    def _extra_engine_setup(self, engine: Engine):
        """Do any extra configuration if needed before a connection is made."""

        if isinstance(self.dwh, Dsn) and self.dwh.search_path:
            # Avoid explicitly setting schema whenever we build a Table.
            #   https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#setting-alternate-search-paths-on-connect
            # If possible, have the client also consider setting that as a default on the role, e.g.:
            #   ALTER USER username SET search_path = schema1, schema2, public;

            @event.listens_for(engine, "connect", insert=True)
            def set_search_path(dbapi_connection, _connection_record):
                existing_autocommit = dbapi_connection.autocommit
                dbapi_connection.autocommit = True
                cursor = dbapi_connection.cursor()
                cursor.execute(f"SET SESSION search_path={self.dwh.search_path}")  # type: ignore[union-attr]
                cursor.close()
                dbapi_connection.autocommit = existing_autocommit

        # Partially address any Redshift incompatibilities
        # re: https://github.com/sqlalchemy-redshift/sqlalchemy-redshift/issues/264#issuecomment-2181124071
        if self.dwh.is_redshift() and hasattr(engine.dialect, "_set_backslash_escapes"):
            engine.dialect._set_backslash_escapes = lambda _: None


class SqliteLocalConfig(ParticipantsMixin, ConfigBaseModel):
    type: Literal["sqlite_local"]
    sqlite_filename: str

    def supports_reflection(self):
        return True

    def dbsession(self):
        """Returns a Session to be used to send queries to a SQLite database.

        Use this in a `with` block to ensure correct transaction handling. If you need the
        sqlalchemy Engine, call .get_bind().
        """
        url = sqlalchemy.URL.create(
            drivername="sqlite",
            database=self.sqlite_filename,
            query={"mode": "ro"},
        )
        engine = sqlalchemy.create_engine(
            url,
            connect_args={"timeout": 5},
            echo=os.environ.get("ECHO_SQL", "").lower() in ("true", "1"),
        )

        @event.listens_for(engine, "connect")
        def register_sqlite_functions(dbapi_connection, _):
            NumpyStddev.register(dbapi_connection)

        return Session(engine)


type DatasourceConfig = RemoteDatabaseConfig | SqliteLocalConfig


class Datasource(ConfigBaseModel):
    """Datasource describes data warehouse configuration and policy."""

    id: str
    config: Annotated[DatasourceConfig, Field(discriminator="type")]
    require_api_key: Annotated[bool | None, Field(...)] = None


class XnginSettings(ConfigBaseModel):
    trusted_ips: Annotated[list[str], Field(default_factory=list)]
    db_connect_timeout_secs: int = 3
    datasources: list[Datasource]

    def get_datasource(self, datasource_id):
        """Finds the datasource for a specific ID if it exists, or returns None."""
        for datasource in self.datasources:
            if datasource.id == datasource_id:
                return datasource
        return None


class SettingsForTesting(XnginSettings):
    pass


class CannotFindTableError(Exception):
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


class CannotFindParticipantsError(Exception):
    """Raised when we cannot find a participant in the configuration."""

    def __init__(self, participant_type):
        self.participant_type = participant_type
        self.message = f"""The participant type '{participant_type}' does not exist.
            (Possible typo in request, or server settings for your dwh may be
            misconfigured.)"""

    def __str__(self):
        return self.message


def infer_table_from_cursor(
    engine: sqlalchemy.engine.Engine, table_name: str
) -> sqlalchemy.Table:
    """Creates a SQLAlchemy Table instance from cursor description metadata."""

    columns = []
    metadata = sqlalchemy.MetaData()
    try:
        with engine.connect() as connection:
            safe_table = sqlalchemy.quoted_name(table_name, quote=True)
            # Create a select statement - this is safe from SQL injection
            query = sqlalchemy.select(text("*")).select_from(text(safe_table)).limit(0)
            result = connection.execute(query)
            description = result.cursor.description
            # print("CURSOR DESC: ", result.cursor.description)
            for col in description:
                # Unpack cursor.description tuple
                (
                    name,
                    type_code,
                    _,  # display_size,
                    internal_size,
                    precision,
                    scale,
                    null_ok,
                ) = col

                # Map Redshift type codes to SQLAlchemy types. Not comprehensive.
                # https://docs.sqlalchemy.org/en/20/core/types.html
                # Comment shows both pg_type.typename / information_schema.data_type
                sa_type: type[sqlalchemy.TypeEngine] | sqlalchemy.TypeEngine
                match type_code:
                    case 16:  # BOOL / boolean
                        sa_type = sqlalchemy.Boolean
                    case 20:  # INT8 / bigint
                        sa_type = sqlalchemy.BigInteger
                    case 23:  # INT4 / integer
                        sa_type = sqlalchemy.Integer
                    case 701:  # FLOAT8 / double precision
                        sa_type = sqlalchemy.Double
                    case 1043:  # VARCHAR / character varying
                        sa_type = sqlalchemy.String(internal_size)
                    case 1082:  # DATE / date
                        sa_type = sqlalchemy.Date
                    case 1114:  # TIMESTAMP / timestamp without time zone
                        sa_type = sqlalchemy.DateTime
                    case 1700:  # NUMERIC / numeric
                        sa_type = sqlalchemy.Numeric(precision, scale)
                    case _:  # type_code == 25
                        # Default to Text for unknown types
                        sa_type = sqlalchemy.Text

                columns.append(
                    sqlalchemy.Column(
                        name, sa_type, nullable=null_ok if null_ok is not None else True
                    )
                )
            return sqlalchemy.Table(table_name, metadata, *columns, quote=False)
    except NoSuchTableError as nste:
        metadata.reflect(engine)
        existing_tables = metadata.tables.keys()
        raise CannotFindTableError(table_name, existing_tables) from nste


def infer_table(engine: sqlalchemy.engine.Engine, table_name: str, use_reflection=True):
    """Constructs a Table via reflection.

    Raises CannotFindTheTableException containing helpful error message if the table doesn't exist.
    """
    metadata = sqlalchemy.MetaData()
    try:
        if use_reflection:
            return sqlalchemy.Table(
                table_name, metadata, autoload_with=engine, quote=False
            )
        # This method of introspection should only be used if the db dialect doesn't support Sqlalchemy2 reflection.
        return infer_table_from_cursor(engine, table_name)
    except sqlalchemy.exc.ProgrammingError:
        logger.exception("Failed to create a Table! use_reflection: %s", use_reflection)
        raise
    except NoSuchTableError as nste:
        metadata.reflect(engine)
        existing_tables = metadata.tables.keys()
        raise CannotFindTableError(table_name, existing_tables) from nste
