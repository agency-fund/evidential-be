"""Settings describe customer configuration data loaded at startup from static JSON files (obsolete)
or read from the database (see tables.Datasource.get_config()).

The Pydantic classes herein also provide some methods for connecting to the customer databases.
"""

import base64
import binascii
import os
from collections import Counter
from typing import Annotated, Literal, Protocol

import sqlalchemy
from cachetools.func import ttl_cache
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from sqlalchemy import make_url

from xngin.apiserver.certs import get_amazon_trust_ca_bundle_path
from xngin.apiserver.dwh.inspection_types import ParticipantsSchema
from xngin.apiserver.routers.common_api_types import (
    validate_gcp_service_account_info_json,
)
from xngin.xsecrets import secretservice

SA_LOGGER_NAME_FOR_DWH = "xngin_dwh"
TIMEOUT_SECS_FOR_CUSTOMER_POSTGRES = 10


@ttl_cache(maxsize=128, ttl=600)
def _decrypt_string(ciphertext: str, aad: str) -> str:
    """Decrypts a serialized ciphertext string.

    This method is cached because it can avoid a remote API call to the key management service.
    """
    return secretservice.get_symmetric().decrypt(ciphertext, aad)


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParticipantsDef(ParticipantsSchema):
    """Participants are a logical representation of a table in the data warehouse.

    Participants are defined by a participant_type, table_name and a schema.
    """

    type: Annotated[
        Literal["schema"],
        Field(description="Indicates that the schema is determined by an inline schema."),
    ]
    participant_type: Annotated[
        str,
        Field(
            description="The name of the set of participants defined by the filters. This name must be unique "
            "within a datasource."
        ),
    ]
    hidden: Annotated[
        bool,
        Field(
            default=False,
            description="If true, this participant type is hidden from list_participant_types. "
            "Used for auto-generated participant types.",
        ),
    ] = False


class ParticipantsMixin(ConfigBaseModel):
    """ParticipantsMixin can be added to a config type to add standardized participant definitions."""

    participants: Annotated[
        list[ParticipantsDef],
        Field(),
    ]

    def find_participants(self, participant_type: str) -> ParticipantsDef:
        """Returns the ParticipantsDef matching participant_type or raises CannotFindParticipantsException."""
        found = self.find_participants_or_none(participant_type)
        if found is None:
            raise CannotFindParticipantsError(participant_type)
        return found

    def find_participants_or_none(self, participant_type) -> ParticipantsDef | None:
        return next(
            (u for u in self.participants if u.participant_type.lower() == participant_type.lower()),
            None,
        )

    @model_validator(mode="after")
    def check_unique_participant_types(self):
        counted = Counter([participant.participant_type for participant in self.participants])
        duplicates = [item for item, count in counted.items() if count > 1]
        if duplicates:
            raise ValueError(f"Participant types with conflicting names found: {', '.join(duplicates)}.")
        return self


type HttpMethodTypes = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]


class ToSqlalchemyUrl(Protocol):
    def to_sqlalchemy_url(self) -> sqlalchemy.URL:
        """Creates a sqlalchemy.URL from this Dsn."""


class EncryptedDsn:
    """Marker interface for Dsn types that have encrypted fields."""

    def encrypt(self, datasource_id: str):
        """Returns a copy of the Dsn with some fields encrypted.

        :arg datasource_id: The ID of the datasource. This is used as AAD to protect the encrypted payload.
        """
        raise NotImplementedError

    def decrypt(self, datasource_id: str):
        """Returns a copy of the Dsn with some fields decrypted.

        :arg datasource_id: The ID of the datasource. This is used as AAD to protect the encrypted payload.
        """
        raise NotImplementedError


class BaseDsn:
    def is_redshift(self):
        """Return true iff the hostname indicates that this is connecting to Redshift."""
        return False

    def supports_sa_autoload(self):
        return not self.is_redshift()


class GcpServiceAccountInfo(ConfigBaseModel):
    """Describes a Google Cloud Service Account credential."""

    type: Literal["serviceaccountinfo"] = "serviceaccountinfo"
    # Note: this field may be stored in an encrypted form.
    content_base64: Annotated[
        str,
        Field(
            ...,
            description="The base64-encoded service account info in the canonical JSON form.",
            min_length=4,
            max_length=8000,
        ),
    ]

    def encrypt(self, datasource_id):
        return self.model_copy(
            update={
                "content_base64": secretservice.get_symmetric().encrypt(self.content_base64, f"dsn.{datasource_id}")
            }
        )

    def decrypt(self, datasource_id):
        return self.model_copy(
            update={"content_base64": _decrypt_string(self.content_base64, aad=f"dsn.{datasource_id}")}
        )

    @field_validator("content_base64")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        """Validates that content_base64 contains valid base64 data."""
        # Hack: do not validate encrypted values
        if secretservice.is_encrypted(value):
            return value

        try:
            # Decode and validate the JSON structure matches Google Cloud Service Account format.
            decoded = base64.b64decode(value, validate=True)
            validate_gcp_service_account_info_json(decoded)
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


class BqDsn(ConfigBaseModel, BaseDsn, EncryptedDsn):
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

    def encrypt(self, datasource_id):
        if self.credentials.type == "serviceaccountinfo":
            return self.model_copy(update={"credentials": self.credentials.encrypt(datasource_id)})
        return self

    def decrypt(self, datasource_id):
        if self.credentials.type == "serviceaccountinfo":
            return self.model_copy(update={"credentials": self.credentials.decrypt(datasource_id)})
        return self

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


class Dsn(ConfigBaseModel, BaseDsn, EncryptedDsn):
    """Describes a set of parameters suitable for connecting to most types of remote databases."""

    driver: Literal[
        "postgresql+psycopg",  # Preferred for most Postgres-compatible databases.
        "postgresql+psycopg2",  # Use with: Redshift
    ]
    host: str
    port: Annotated[int, Field(ge=1024, le=65535)] = 5432
    user: str
    # Note: this field may be stored in an encrypted form.
    password: str
    dbname: str
    sslmode: Literal["disable", "require", "verify-ca", "verify-full"]
    # Specify the order in which schemas are searched if your dwh supports it.
    search_path: str | None = None

    def encrypt(self, datasource_id):
        return self.model_copy(
            update={"password": secretservice.get_symmetric().encrypt(self.password, f"dsn.{datasource_id}")}
        )

    def decrypt(self, datasource_id):
        return self.model_copy(
            update={
                "password": _decrypt_string(
                    self.password,
                    aad=f"dsn.{datasource_id}",
                )
            }
        )

    @staticmethod
    def from_url(url: str):
        """Constructs a Dsn from a SQLAlchemy-compatible URL (Postgres or BigQuery only).

        Use only in trusted code paths. If url is BigQuery, credentials are assumed to be in a file referenced by
        the GOOGLE_APPLICATION_CREDENTIALS environment variable.
        """

        if url.startswith("bigquery"):
            # This ignores URL-encoded bigquery credentials from the query string
            parsed_url = make_url(url)
            credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None)
            if credentials is None:
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS must be set when using Dsn.from_url.")
            return BqDsn(
                driver="bigquery",
                project_id=parsed_url.host,
                dataset_id=parsed_url.database,
                credentials=GcpServiceAccountFile(type="serviceaccountfile", path=credentials),
            )

        if url.startswith("postgres"):
            parsed_url = make_url(url)
            return Dsn(
                driver=f"postgresql+{parsed_url.get_driver_name()}",
                host=parsed_url.host,
                port=parsed_url.port,
                user=parsed_url.username,
                password=parsed_url.password,
                dbname=parsed_url.database,
                sslmode=parsed_url.query.get("sslmode", "verify-ca"),
                search_path=parsed_url.query.get("search_path", None),
            )
        raise NotImplementedError(f"Dsn.from_url() only supports postgres databases: {url}")

    def is_redshift(self):
        return self.host.endswith("redshift.amazonaws.com")

    def to_sqlalchemy_url(self):
        url = sqlalchemy.URL.create(
            drivername=self.driver,
            username=self.user,
            password=self.password,
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
                raise ValueError("Redshift connections must use postgresql+psycopg2 driver")
            if self.sslmode != "verify-full":
                raise ValueError("Redshift connections must use sslmode=verify-full")
        return self


class NoDwh(ConfigBaseModel):
    """NoDwh is used to indicate that no data warehouse is configured."""

    driver: Literal["none"] = "none"

    def to_sqlalchemy_url(self):
        raise NotImplementedError("NoDwh does not support to_sqlalchemy_url()")

    def supports_sa_autoload(self):
        return False


type Dwh = Annotated[Dsn | BqDsn | NoDwh, Field(discriminator="driver")]


class RemoteDatabaseConfig(ParticipantsMixin, ConfigBaseModel):
    """RemoteDatabaseConfig defines a configuration for a remote data warehouse."""

    type: Literal["remote"]

    dwh: Dwh

    def to_sqlalchemy_url(self):
        return self.dwh.to_sqlalchemy_url()

    def supports_sa_autoload(self):
        return self.dwh.supports_sa_autoload()


type DatasourceConfig = RemoteDatabaseConfig


class Datasource(ConfigBaseModel):
    """Datasource describes data warehouse configuration and policy."""

    id: str
    config: DatasourceConfig


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
