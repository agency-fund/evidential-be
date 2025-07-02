"""Settings describe customer configuration data loaded at startup from static JSON files (obsolete)
or read from the database (see tables.Datasource.get_config()).

The Pydantic classes herein also provide some methods for connecting to the customer databases.
"""

import base64
import binascii
import json
import os
from collections import Counter
from functools import lru_cache
from typing import Annotated, Literal, Protocol
from urllib.parse import urlparse

import httpx
import sqlalchemy
from httpx import codes
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_serializer,
    field_validator,
    model_validator,
)
from sqlalchemy import make_url
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_delay,
    wait_random,
)

from xngin.apiserver.certs import get_amazon_trust_ca_bundle_path
from xngin.apiserver.dwh.inspection_types import ParticipantsSchema
from xngin.apiserver.settings_secrets import replace_secrets

DEFAULT_SETTINGS_FILE = "xngin.settings.json"
SA_LOGGER_NAME_FOR_DWH = "xngin_dwh"
TIMEOUT_SECS_FOR_CUSTOMER_POSTGRES = 10


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
        logger.info("Loading XNGIN_SETTINGS from disk: {}", settings_path)
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
    logger.info("Loading XNGIN_SETTINGS from URL: {}", url)
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

    participant_type: Annotated[
        str,
        Field(
            description="The name of the set of participants defined by the filters. This name must be unique "
            "within a datasource."
        ),
    ]


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

    def find_participants(self, participant_type: str) -> ParticipantsConfig:
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
                f"Participant types with conflicting names found: {', '.join(duplicates)}."
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

    def supports_sa_autoload(self):
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
    sslmode: Literal["disable", "require", "verify-ca", "verify-full"]
    # Specify the order in which schemas are searched if your dwh supports it.
    search_path: str | None = None

    @staticmethod
    def from_url(url: str):
        """Constructs a Dsn from a SQLAlchemy-compatible URL (Postgres or BigQuery only).

        Use only in trusted code paths. If url is BigQuery, credentials are assumed to be in a file referenced by
        the GOOGLE_APPLICATION_CREDENTIALS environment variable.
        """

        if url.startswith("bigquery"):
            # TODO: support URL-encoded bigquery credentials from the query string
            url = make_url(url)
            credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None)
            if credentials is None:
                raise ValueError(
                    "GOOGLE_APPLICATION_CREDENTIALS must be set when using Dsn.from_url."
                )
            return BqDsn(
                driver="bigquery",
                project_id=url.host,
                dataset_id=url.database,
                credentials=GcpServiceAccountFile(
                    type="serviceaccountfile", path=credentials
                ),
            )

        if url.startswith("postgres"):
            url = make_url(url)
            return Dsn(
                driver=f"postgresql+{url.get_driver_name()}",
                host=url.host,
                port=url.port,
                user=url.username,
                password=url.password,
                dbname=url.database,
                sslmode=url.query.get("sslmode", "verify-ca"),
                search_path=url.query.get("search_path", None),
            )
        raise NotImplementedError("Dsn.from_url() only supports postgres databases.")

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
                "sslmode": self.sslmode,
                # re: redshift issue https://github.com/psycopg/psycopg/issues/122#issuecomment-985742751
                "client_encoding": "utf-8",
            })
            if self.is_redshift():
                query.update({
                    "sslmode": "verify-full",
                    "sslrootcert": get_amazon_trust_ca_bundle_path(),
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
                raise ValueError("Redshift connections must use sslmode=verify-full")
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

    def to_sqlalchemy_url(self):
        return self.dwh.to_sqlalchemy_url()

    def supports_sa_autoload(self):
        return self.dwh.supports_sa_autoload()


# TODO: use a Field(discriminator="type") when we support more than just "remote" databases.
type DatasourceConfig = RemoteDatabaseConfig


class Datasource(ConfigBaseModel):
    """Datasource describes data warehouse configuration and policy."""

    id: str
    config: DatasourceConfig
    require_api_key: Annotated[bool | None, Field(...)] = None


class XnginSettings(ConfigBaseModel):
    trusted_ips: Annotated[list[str], Field(default_factory=list, deprecated=True)]
    db_connect_timeout_secs: Annotated[int, Field(deprecated=True)] = 3
    datasources: list[Datasource]

    def get_datasource(self, datasource_id):
        """Finds the datasource for a specific ID if it exists, or returns None."""
        for datasource in self.datasources:
            if datasource.id == datasource_id:
                return datasource
        return None


class SettingsForTesting(XnginSettings):
    pass


class CannotFindParticipantsError(Exception):
    """Raised when we cannot find a participant in the configuration."""

    def __init__(self, participant_type):
        self.participant_type = participant_type
        self.message = (
            f"The participant type '{participant_type}' does not exist."
            "(Possible typo in request, or server settings for your dwh may be "
            "misconfigured.)"
        )

    def __str__(self):
        return self.message
