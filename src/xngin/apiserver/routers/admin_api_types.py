from datetime import datetime
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from xngin.apiserver.common_field_types import FieldName
from xngin.apiserver.limits import (
    MAX_LENGTH_OF_DESCRIPTION_VALUE,
    MAX_LENGTH_OF_EMAIL_VALUE,
    MAX_LENGTH_OF_ID_VALUE,
    MAX_LENGTH_OF_NAME_VALUE,
    MAX_LENGTH_OF_WEBHOOK_URL_VALUE,
    MAX_NUMBER_OF_FIELDS,
)
from xngin.apiserver.routers.stateless_api_types import (
    ApiBaseModel,
    DataType,
    GetFiltersResponseElement,
    GetMetricsResponseElement,
    GetStrataResponseElement,
)
from xngin.apiserver.settings import DatasourceConfig, Dwh, ParticipantsConfig
from xngin.schema.schema_types import FieldDescriptor, ParticipantsSchema


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
    created_at: Annotated[
        datetime, Field(description="The time the event was created.")
    ]
    type: Annotated[str, Field(description="The type of event.")]
    summary: Annotated[str, Field(description="Human-readable summary of the event.")]
    link: Annotated[
        str | None, Field(description="A navigable link to related information.")
    ] = None
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
            description="User-friendly name for the webhook. This name is displayed in the UI and helps identify the webhook's purpose.",
        ),
    ]
    url: Annotated[
        str,
        Field(
            max_length=MAX_LENGTH_OF_WEBHOOK_URL_VALUE,
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
    type: Annotated[
        str, Field(description="The type of webhook; e.g. experiment.created")
    ]
    name: Annotated[
        str,
        Field(description="User-friendly name for the webhook."),
    ]
    url: Annotated[str, Field(description="The URL to notify.")]
    auth_token: Annotated[
        str | None,
        Field(
            description="The value of the Authorization: header that will be sent with the request to the configured URL."
        ),
    ]


class WebhookSummary(AdminApiBaseModel):
    """Summarizes a Webhook configuration for an organization."""

    id: Annotated[str, Field(description="The ID of the webhook.")]
    type: Annotated[
        str, Field(description="The type of webhook; e.g. experiment.created")
    ]
    name: Annotated[
        str,
        Field(description="User-friendly name for the webhook."),
    ]
    url: Annotated[str, Field(description="The URL to notify.")]
    auth_token: Annotated[
        str | None,
        Field(
            description="The value of the Authorization: header that will be sent with the request to the configured URL."
        ),
    ]


class UpdateOrganizationWebhookRequest(AdminApiBaseModel):
    """Request to update a webhook's URL."""

    url: Annotated[str, Field(max_length=MAX_LENGTH_OF_WEBHOOK_URL_VALUE)]

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        return validate_webhook_url(v)


class ListWebhooksResponse(AdminApiBaseModel):
    items: list[WebhookSummary]


class CreateDatasourceRequest(AdminApiBaseModel):
    organization_id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]
    name: Annotated[str, Field(...)]
    dwh: Dwh


class UpdateDatasourceRequest(AdminApiBaseModel):
    name: Annotated[str | None, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)] = None
    dwh: Annotated[Dwh | None, Field()] = None


class CreateDatasourceResponse(AdminApiBaseModel):
    id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]


class GetDatasourceResponse(AdminApiBaseModel):
    id: Annotated[str, Field(max_length=MAX_LENGTH_OF_ID_VALUE)]
    name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]
    config: DatasourceConfig  # TODO: map this to a public type
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
    participant_type: Annotated[
        str | None, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)
    ] = None
    table_name: Annotated[FieldName | None, Field()] = None
    fields: Annotated[
        list[FieldDescriptor] | None, Field(max_length=MAX_NUMBER_OF_FIELDS)
    ] = None


class UpdateParticipantsTypeResponse(ApiBaseModel):
    participant_type: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]
    table_name: Annotated[FieldName | None, Field()] = None
    fields: Annotated[
        list[FieldDescriptor] | None, Field(max_length=MAX_NUMBER_OF_FIELDS)
    ] = None


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
