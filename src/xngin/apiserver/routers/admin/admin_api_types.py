import enum
from datetime import datetime
from typing import Annotated, Literal
from urllib.parse import urlparse

from annotated_types import Ge, Le
from pydantic import BaseModel, ConfigDict, Field, field_validator

from xngin.apiserver.common_field_types import FieldName
from xngin.apiserver.dwh.inspection_types import FieldDescriptor, ParticipantsSchema
from xngin.apiserver.limits import (
    MAX_LENGTH_OF_DESCRIPTION_VALUE,
    MAX_LENGTH_OF_EMAIL_VALUE,
    MAX_LENGTH_OF_ID_VALUE,
    MAX_LENGTH_OF_NAME_VALUE,
    MAX_LENGTH_OF_URL_VALUE,
    MAX_NUMBER_OF_FIELDS,
)
from xngin.apiserver.routers.common_api_types import (
    ApiBaseModel,
    ConstrainedUrl,
    DataType,
    ExperimentAnalysisResponse,
    GcpServiceAccountBlob,
    GetFiltersResponseElement,
    GetMetricsResponseElement,
    GetStrataResponseElement,
    Impact,
)
from xngin.apiserver.settings import ParticipantsConfig


def validate_webhook_url(url: str) -> str:
    """Validates that a URL is a properly formatted HTTP or HTTPS URL."""
    parsed = urlparse(url)
    if not parsed.scheme or parsed.scheme not in {"http", "https"}:
        raise ValueError("URL must use http or https scheme")
    if not parsed.netloc:
        raise ValueError("URL must include a valid domain")
    return url


class AdminApiBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SnapshotStatus(enum.StrEnum):
    """Describes the status of a snapshot."""

    SUCCESS = "success"
    RUNNING = "running"
    FAILED = "failed"


class Snapshot(AdminApiBaseModel):
    experiment_id: Annotated[str, Field(description="The experiment that this snapshot was captured for.")]
    id: Annotated[str, Field(description="The unique ID of the snapshot.")]

    status: Annotated[
        SnapshotStatus,
        Field(description="The status of the snapshot. When not `success`, data will be null."),
    ]
    details: Annotated[dict | None, Field(description="Additional data about this snapshot.")]
    created_at: Annotated[datetime, Field(description="The time the snapshot was requested.")]
    updated_at: Annotated[datetime, Field(description="The time the snapshot was acquired.")]
    data: Annotated[ExperimentAnalysisResponse | None, Field(description="Analysis results as of the updated_at time.")]


class GetSnapshotResponse(AdminApiBaseModel):
    """Describes the status and content of a snapshot."""

    snapshot: Annotated[Snapshot | None, Field(description="The completed snapshot.")]


class ListSnapshotsResponse(AdminApiBaseModel):
    items: list[Snapshot]


class CreateSnapshotResponse(AdminApiBaseModel):
    id: str


class CreateOrganizationRequest(AdminApiBaseModel):
    name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]


class CreateOrganizationResponse(AdminApiBaseModel):
    id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]


class UpdateOrganizationRequest(AdminApiBaseModel):
    name: Annotated[str | None, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)] = None


class OrganizationSummary(AdminApiBaseModel):
    id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]
    name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]


class DatasourceSummary(AdminApiBaseModel):
    id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]
    name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]
    driver: str
    type: str
    organization_id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]
    organization_name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]


class UserSummary(AdminApiBaseModel):
    id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]
    email: Annotated[str, Field(max_length=MAX_LENGTH_OF_EMAIL_VALUE)]


class ListOrganizationsResponse(AdminApiBaseModel):
    items: list[OrganizationSummary]


class EventSummary(AdminApiBaseModel):
    """Describes an event."""

    id: Annotated[str, Field(description="The ID of the event.")]
    created_at: Annotated[datetime, Field(description="The time the event was created.")]
    type: Annotated[str, Field(description="The type of event.")]
    summary: Annotated[str, Field(description="Human-readable summary of the event.")]
    link: Annotated[str | None, Field(description="A navigable link to related information.")] = None
    details: Annotated[dict | None, Field(description="Details")]


class ListOrganizationEventsResponse(AdminApiBaseModel):
    items: list[EventSummary]


class GetOrganizationResponse(AdminApiBaseModel):
    id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]
    name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]
    users: list[UserSummary]
    datasources: list[DatasourceSummary]


class AddMemberToOrganizationRequest(AdminApiBaseModel):
    email: Annotated[str, Field(...)]


class ListDatasourcesResponse(AdminApiBaseModel):
    items: list[DatasourceSummary]


class AddWebhookToOrganizationRequest(AdminApiBaseModel):
    type: Literal["experiment.created"]
    name: Annotated[
        str,
        Field(
            max_length=MAX_LENGTH_OF_NAME_VALUE,
            description=(
                "User-friendly name for the webhook. This name is displayed in the UI and helps "
                "identify the webhook's purpose."
            ),
        ),
    ]
    url: Annotated[
        str,
        Field(
            max_length=MAX_LENGTH_OF_URL_VALUE,
            description="The HTTP or HTTPS URL that will receive webhook notifications when events occur.",
        ),
    ]

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        return validate_webhook_url(v)


class AddWebhookToOrganizationResponse(AdminApiBaseModel):
    """Information on the successfully created webhook."""

    id: Annotated[str, Field(description="The ID of the newly created webhook.")]
    type: Annotated[str, Field(description="The type of webhook; e.g. experiment.created")]
    name: Annotated[
        str,
        Field(description="User-friendly name for the webhook."),
    ]
    url: Annotated[str, Field(description="The URL to notify.")]
    auth_token: Annotated[
        str | None,
        Field(
            description=(
                "The value of the Webhook-Token: header that will be sent with the request to the configured URL."
            )
        ),
    ]


class WebhookSummary(AdminApiBaseModel):
    """Summarizes a Webhook configuration for an organization."""

    id: Annotated[str, Field(description="The ID of the webhook.")]
    type: Annotated[str, Field(description="The type of webhook; e.g. experiment.created")]
    name: Annotated[
        str,
        Field(description="User-friendly name for the webhook."),
    ]
    url: Annotated[str, Field(description="The URL to notify.")]
    auth_token: Annotated[
        str | None,
        Field(
            description=(
                "The value of the Webhook-Token: header that will be sent with the request to the configured URL."
            )
        ),
    ]


class UpdateOrganizationWebhookRequest(AdminApiBaseModel):
    """Request to update a webhook's name and URL."""

    name: Annotated[
        str,
        Field(
            max_length=MAX_LENGTH_OF_NAME_VALUE,
            description=(
                "User-friendly name for the webhook. This name is displayed in the UI and helps "
                "identify the webhook's purpose."
            ),
        ),
    ]
    url: Annotated[
        str,
        Field(
            max_length=MAX_LENGTH_OF_URL_VALUE,
            description="The HTTP or HTTPS URL that will receive webhook notifications when events occur.",
        ),
    ]

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        return validate_webhook_url(v)


class ListWebhooksResponse(AdminApiBaseModel):
    items: list[WebhookSummary]


class Hidden(AdminApiBaseModel):
    """Hidden represents a credential that is intentionally omitted."""

    type: Literal["hidden"] = "hidden"


class RevealedStr(AdminApiBaseModel):
    """RevealedStr contains a credential."""

    type: Literal["revealed"] = "revealed"
    value: str


class GcpServiceAccount(AdminApiBaseModel):
    """Describes a Google Cloud Platform service account."""

    type: Literal["serviceaccountinfo"] = "serviceaccountinfo"
    content: GcpServiceAccountBlob


class PostgresDsn(AdminApiBaseModel):
    """PostgresDsn describes a connection to a Postgres-compatible database."""

    type: Literal["postgres"] = "postgres"

    host: str
    port: Annotated[int, Ge(1024), Le(65535)]
    user: str
    password: Annotated[
        RevealedStr | Hidden,
        Field(
            discriminator="type",
            description=(
                "This value must be a RevealedStr when creating the datasource or when updating a "
                "datasource's credentials. It may be a Hidden when updating a datasource. When hidden, "
                "the existing credentials are retained."
            ),
        ),
    ]
    dbname: str
    sslmode: Literal["disable", "require", "verify-ca", "verify-full"]
    search_path: str | None


class RedshiftDsn(AdminApiBaseModel):
    """RedshiftDsn describes a connection to a Redshift database."""

    type: Literal["redshift"] = "redshift"

    host: str
    port: Annotated[int, Ge(1024), Le(65535)]
    user: str
    password: Annotated[
        RevealedStr | Hidden,
        Field(
            discriminator="type",
            description=(
                "This value must be a RevealedStr when creating the datasource or when updating a "
                "datasource's credentials. It may be a Hidden when updating a datasource. When hidden, "
                "the existing credentials are retained."
            ),
        ),
    ]
    dbname: str
    search_path: str | None


class BqDsn(AdminApiBaseModel):
    """BqDsn describes a connection to a BigQuery database."""

    type: Literal["bigquery"] = "bigquery"

    project_id: Annotated[
        str,
        Field(
            description="The Google Cloud Project ID containing the dataset.",
            min_length=6,
            max_length=30,
            pattern=r"^[a-z0-9-]+$",
        ),
    ]
    dataset_id: Annotated[
        str,
        Field(
            description="The dataset name.",
            min_length=1,
            max_length=1024,
            pattern=r"^[a-zA-Z0-9_]+$",
        ),
    ]

    credentials: Annotated[
        GcpServiceAccount | Hidden,
        Field(
            discriminator="type",
            description=(
                "This value must be a GcpServiceAccount when creating the datasource or when updating a "
                "datasource's credentials. It may be a Hidden when updating a datasource. When hidden, "
                "the existing credentials are retained."
            ),
        ),
    ]


class ApiOnlyDsn(AdminApiBaseModel):
    """ApiOnlyDsn describes a datasource where data is included in Evidential API requests."""

    type: Literal["api_only"] = "api_only"


type Dsn = Annotated[ApiOnlyDsn | PostgresDsn | BqDsn | RedshiftDsn, Field(discriminator="type")]


class CreateDatasourceRequest(AdminApiBaseModel):
    organization_id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]
    name: Annotated[str, Field(...)]
    dsn: Dsn


class UpdateDatasourceRequest(AdminApiBaseModel):
    name: Annotated[str | None, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)] = None
    dsn: Dsn | None = None


class CreateDatasourceResponse(AdminApiBaseModel):
    id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]


class GetDatasourceResponse(AdminApiBaseModel):
    id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]
    name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]
    dsn: Dsn
    organization_id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]
    organization_name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]


class InspectDatasourceResponse(ApiBaseModel):
    tables: list[str]


class FieldMetadata(ApiBaseModel):
    """Concise summary of fields in the table."""

    field_name: FieldName
    data_type: DataType
    description: Annotated[str, Field(max_length=MAX_LENGTH_OF_DESCRIPTION_VALUE)]


class InspectDatasourceTableResponse(ApiBaseModel):
    """Describes a table in the datasource."""

    detected_unique_id_fields: Annotated[
        list[str],
        Field(description="Fields that are possibly candidates for unique IDs."),
    ]
    fields: Annotated[list[FieldMetadata], Field(description="Fields in the table.")]


class InspectParticipantTypesResponse(ApiBaseModel):
    """Describes a participant type's strata, metrics, and filters (including exemplar values)."""

    filters: Annotated[list[GetFiltersResponseElement], Field()]
    metrics: Annotated[list[GetMetricsResponseElement], Field()]
    strata: Annotated[list[GetStrataResponseElement], Field()]


class ListParticipantsTypeResponse(ApiBaseModel):
    items: list[ParticipantsConfig]


class CreateParticipantsTypeRequest(ApiBaseModel):
    participant_type: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]
    schema_def: Annotated[ParticipantsSchema, Field()]


class CreateParticipantsTypeResponse(ApiBaseModel):
    participant_type: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]
    schema_def: Annotated[ParticipantsSchema, Field()]


class UpdateParticipantsTypeRequest(ApiBaseModel):
    participant_type: Annotated[str | None, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)] = None
    table_name: Annotated[FieldName | None, Field()] = None
    fields: Annotated[list[FieldDescriptor] | None, Field(max_length=MAX_NUMBER_OF_FIELDS)] = None


class GetParticipantsTypeResponse(ApiBaseModel):
    participants_config: ParticipantsConfig


class UpdateParticipantsTypeResponse(ApiBaseModel):
    participant_type: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]
    table_name: Annotated[FieldName | None, Field()] = None
    fields: Annotated[list[FieldDescriptor] | None, Field(max_length=MAX_NUMBER_OF_FIELDS)] = None


class ApiKeySummary(AdminApiBaseModel):
    id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]
    datasource_id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]
    organization_id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]
    organization_name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]


class ListApiKeysResponse(AdminApiBaseModel):
    items: list[ApiKeySummary]


class CreateApiKeyResponse(AdminApiBaseModel):
    id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]
    datasource_id: Annotated[str, Field(...)]
    key: Annotated[str, Field(...)]


class CreateUserRequest(AdminApiBaseModel):
    email: Annotated[str, Field(max_length=MAX_LENGTH_OF_EMAIL_VALUE)]
    organization_id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]


class UpdateExperimentRequest(AdminApiBaseModel):
    """Defines the subset of fields that can be updated for an experiment after creation."""

    name: Annotated[str | None, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)] = None
    description: Annotated[str | None, Field(max_length=MAX_LENGTH_OF_DESCRIPTION_VALUE)] = None
    design_url: Annotated[str | None, Field(max_length=MAX_LENGTH_OF_URL_VALUE)] = None
    start_date: Annotated[datetime | None, Field()] = None
    end_date: Annotated[datetime | None, Field()] = None

    impact: Annotated[Impact | None, Field()] = None
    decision: Annotated[str | None, Field()] = None

    @field_validator("design_url")
    @classmethod
    def validate_design_url(cls, design_url: str | None) -> str | None:
        if design_url is None:
            return design_url
        if design_url == "":
            return design_url
        return str(ConstrainedUrl(design_url))


class UpdateArmRequest(AdminApiBaseModel):
    """Defines the subset of fields that can be updated for an Arm after creation."""

    name: Annotated[str | None, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)] = None
    description: Annotated[str | None, Field(max_length=MAX_LENGTH_OF_DESCRIPTION_VALUE)] = None
