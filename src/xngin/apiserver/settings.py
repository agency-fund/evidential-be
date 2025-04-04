import base64
import binascii
import json
import logging
import os
from collections import Counter
from functools import lru_cache
from typing import Annotated, Literal, Protocol
from urllib.parse import urlparse

import httpx
import sqlalchemy
from httpx import codes
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_serializer,
    field_validator,
    model_validator,
)
from sqlalchemy import Engine, event, text
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_delay,
    wait_random,
)
from xngin.apiserver import flags
from xngin.apiserver.certs.certs import PATH_TO_AMAZON_TRUST_CA_BUNDLE
from xngin.apiserver.dns.safe_resolve import safe_resolve
from xngin.apiserver.settings_secrets import replace_secrets
from xngin.db_extensions import NumpyStddev
from xngin.schema.schema_types import ParticipantsSchema

DEFAULT_SETTINGS_FILE = "xngin.settings.json"

logger = logging.getLogger(__name__)


class UnclassifiedRemoteSettingsError(Exception):
    """Raised when we fail to fetch remote settings for an unclassified reason."""


class RemoteSettingsClientError(Exception):
    """Raised when we fail to fetch remote settings due to our misconfiguration."""


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


class BaseParticipantsRef(ConfigBaseModel):
    """Participants are a logical representation of a table in the data warehouse.

    Participants are defined by a participant_type, table_name and a schema.
    """

    participant_type: str


class SheetParticipantsRef(BaseParticipantsRef):
    type: Annotated[
        Literal["sheet"],
        Field(
            description="Indicates that the schema is determined by a remote Google Sheet."
        ),
    ]
    table_name: str
    sheet: SheetRef


class ParticipantsDef(BaseParticipantsRef, ParticipantsSchema):
    type: Annotated[
        Literal["schema"],
        Field(
            description="Indicates that the schema is determined by an inline schema."
        ),
    ]


type ParticipantsConfig = Annotated[
    SheetParticipantsRef | ParticipantsDef, Field(discriminator="type")
]


class ParticipantsMixin(ConfigBaseModel):
    """ParticipantsMixin can be added to a config type to add standardized participant definitions."""

    participants: Annotated[
        list[ParticipantsConfig],
        Field(),
    ]

    def find_participants(self, participant_type: str):
        """Returns the ParticipantsConfig matching participant_type or raises CannotFindParticipantsException."""
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


type HttpMethodTypes = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]


class WebhookUrl(ConfigBaseModel):
    """Represents a url and HTTP method to use with it."""

    method: HttpMethodTypes
    url: str


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
            min_length=4,
            max_length=8000,
        ),
    ]

    @field_validator("content_base64")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        """Validates that content_base64 contains valid base64 data."""
        try:
            # Decode and validate the JSON structure matches Google Cloud Service Account format.
            decoded = base64.b64decode(value, validate=True)
            try:
                creds = json.loads(decoded)
                required_fields = {
                    "type",
                    "project_id",
                    "private_key_id",
                    "private_key",
                    "client_email",
                }
                if not all(field in creds for field in required_fields):
                    raise ValueError("Missing required fields in service account JSON")
                if creds["type"] != "service_account":
                    raise ValueError(
                        'Service account JSON must have type="service_account"'
                    )
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON in service account credentials") from e
        except binascii.Error as e:
            raise ValueError("Invalid base64 content") from e
        return value


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


type GcpCredentials = Annotated[
    GcpServiceAccountInfo | GcpServiceAccountFile,
    Field(
        ...,
        discriminator="type",
        description="The Google Cloud Service Account credentials.",
    ),
]


class BqDsn(ConfigBaseModel, BaseDsn):
    """Describes a BigQuery connection."""

    driver: Literal["bigquery"]
    project_id: Annotated[
        str,
        Field(
            ...,
            description="The Google Cloud Project ID containing the dataset.",
            min_length=6,
            max_length=30,
            pattern=r"^[a-z0-9-]+$",
        ),
    ]
    dataset_id: Annotated[
        str,
        Field(
            ...,
            description="The dataset name.",
            min_length=1,
            max_length=1024,
            pattern=r"^[a-zA-Z0-9_]+$",
        ),
    ]

    # These two authentication modes are documented here:
    # https://googleapis.dev/python/google-api-core/latest/auth.html#service-accounts
    credentials: GcpCredentials

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
    port: Annotated[int, Field(ge=1024, le=65535)] = 5432
    user: str
    password: SecretStr
    dbname: str
    sslmode: (
        Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
        | None
    ) = None
    # Specify the order in which schemas are searched if your dwh supports it.
    search_path: str | None = None

    @field_serializer("password", when_used="json")
    def reveal_password(self, v):
        return v.get_secret_value()

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
                "sslmode": self.sslmode or "verify-full",
                # re: redshift issue https://github.com/psycopg/psycopg/issues/122#issuecomment-985742751
                "client_encoding": "utf-8",
            })
            if self.is_redshift():
                query.update({
                    "sslmode": "verify-full",
                    "sslrootcert": PATH_TO_AMAZON_TRUST_CA_BUNDLE,
                })
            url = url.set(query=query)
        return url

    @model_validator(mode="after")
    def check_redshift_safe(self):
        if self.is_redshift():
            if self.driver != "postgresql+psycopg2":
                raise ValueError(
                    "Redshift connections must use postgresql+psycopg2 driver"
                )
            if self.sslmode != "verify-full":
                raise ValueError("Redshift connections must use sslmode=verify_full")
        return self


class DbapiArg(ConfigBaseModel):
    """Describes a DBAPI connect() argument.

    These can be arbitrary kv pairs and are database-driver specific."""

    arg: str
    value: str


type Dwh = Annotated[Dsn | BqDsn, Field(discriminator="driver")]


class RemoteDatabaseConfig(ParticipantsMixin, ConfigBaseModel):
    """RemoteDatabaseConfig defines a configuration for a remote data warehouse."""

    webhook_config: WebhookConfig | None = None

    type: Literal["remote"]

    dwh: Dwh

    def supports_reflection(self):
        return self.dwh.supports_table_reflection()

    def dbsession(self):
        """Returns a Session to be used to send queries to the customer database.

        Use this in a `with` block to ensure correct transaction handling. If you need the
        sqlalchemy Engine, call .get_bind().
        """
        engine = self.dbengine()
        return Session(engine)

    def dbengine(self):
        """Returns a SQLAlchemy Engine for the customer database.

        Use this when reflecting. If you're doing any queries on the tables, prefer dbsession().
        """
        url = self.dwh.to_sqlalchemy_url()
        connect_args: dict = {}
        if url.get_backend_name() == "postgres":
            connect_args["connect_timeout"] = 5
            # Replace the Postgres' client default DNS lookup with one that applies security checks first; this prevents
            # us from connecting to addresses like 127.0.0.1 or addresses that are on our hosting provider's internal
            # network.
            # https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-PARAMKEYWORDS
            connect_args["hostaddr"] = safe_resolve(url.host)

        engine = sqlalchemy.create_engine(
            url,
            connect_args=connect_args,
            echo=flags.ECHO_SQL,
            poolclass=sqlalchemy.pool.NullPool,
        )
        self._extra_engine_setup(engine)
        return engine

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

    @field_validator("sqlite_filename")
    @classmethod
    def validate_sqlite_filename(cls, value):
        if value.startswith("sqlite://"):
            raise ValueError("sqlite_filename should not start with sqlite://")
        return value

    def supports_reflection(self):
        return True

    def dbsession(self):
        """Returns a Session to be used to send queries to a SQLite database.

        Use this in a `with` block to ensure correct transaction handling. If you need the
        sqlalchemy Engine, call .get_bind().
        """
        engine = self.dbengine()
        return Session(engine)

    def dbengine(self):
        """Returns a SQLAlchemy Engine for the customer database.

        Use this when reflecting. If you're doing any queries on the tables, prefer dbsession().
        """
        url = sqlalchemy.URL.create(
            drivername="sqlite",
            database=self.sqlite_filename,
            query={"mode": "ro"},
        )
        engine = sqlalchemy.create_engine(
            url,
            connect_args={"timeout": 5},
            echo=flags.ECHO_SQL,
        )

        @event.listens_for(engine, "connect")
        def register_sqlite_functions(dbapi_connection, _):
            NumpyStddev.register(dbapi_connection)

        return engine


type DatasourceConfig = Annotated[
    RemoteDatabaseConfig | SqliteLocalConfig, Field(discriminator="type")
]


class Datasource(ConfigBaseModel):
    """Datasource describes data warehouse configuration and policy."""

    id: str
    config: DatasourceConfig
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
        self.message = (
            f"The participant type '{participant_type}' does not exist."
            "(Possible typo in request, or server settings for your dwh may be"
            "misconfigured.)"
        )

    def __str__(self):
        return self.message


def infer_table_from_cursor(
    engine: sqlalchemy.engine.Engine, table_name: str
) -> sqlalchemy.Table:
    """Creates a SQLAlchemy Table instance from cursor description metadata."""

    columns = []
    metadata = sqlalchemy.MetaData()
    try:
        with engine.begin() as connection:
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
