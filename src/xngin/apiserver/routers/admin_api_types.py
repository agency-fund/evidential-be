from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from xngin.apiserver.api_types import (
    ApiBaseModel,
    DataType,
    GetFiltersResponseElement,
    GetMetricsResponseElement,
    GetStrataResponseElement,
)
from xngin.apiserver.settings import Dwh, DatasourceConfig, ParticipantsConfig
from xngin.schema.schema_types import ParticipantsSchema, FieldDescriptor


class AdminApiBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateOrganizationRequest(AdminApiBaseModel):
    name: Annotated[str, Field(...)]


class CreateOrganizationResponse(AdminApiBaseModel):
    id: Annotated[str, Field(...)]


class UpdateOrganizationRequest(AdminApiBaseModel):
    name: Annotated[str | None, Field()] = None


class OrganizationSummary(AdminApiBaseModel):
    id: Annotated[str, Field(...)]
    name: Annotated[str, Field(...)]


class DatasourceSummary(AdminApiBaseModel):
    id: str
    name: str
    driver: str
    type: str
    organization_id: str
    organization_name: str


class UserSummary(AdminApiBaseModel):
    id: Annotated[str, Field(...)]
    email: Annotated[str, Field(...)]


class ListOrganizationsResponse(AdminApiBaseModel):
    items: list[OrganizationSummary]


class GetOrganizationResponse(AdminApiBaseModel):
    id: Annotated[str, Field(...)]
    name: Annotated[str, Field(...)]
    users: list[UserSummary]
    datasources: list[DatasourceSummary]


class AddMemberToOrganizationRequest(AdminApiBaseModel):
    email: Annotated[str, Field(...)]


class ListDatasourcesResponse(AdminApiBaseModel):
    items: list[DatasourceSummary]


class CreateDatasourceRequest(AdminApiBaseModel):
    organization_id: Annotated[str, Field(...)]
    name: Annotated[str, Field(...)]
    dwh: Dwh


class UpdateDatasourceRequest(AdminApiBaseModel):
    name: Annotated[str | None, Field()] = None
    dwh: Annotated[Dwh | None, Field()] = None


class CreateDatasourceResponse(AdminApiBaseModel):
    id: Annotated[str, Field(...)]


class GetDatasourceResponse(AdminApiBaseModel):
    id: Annotated[str, Field(...)]
    name: str
    config: DatasourceConfig  # TODO: map this to a public type
    organization_id: str
    organization_name: str


class InspectDatasourceResponse(ApiBaseModel):
    tables: list[str]


class FieldMetadata(ApiBaseModel):
    """Concise summary of fields in the table."""

    field_name: str
    data_type: DataType
    description: str


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
    participant_type: str
    schema_def: Annotated[ParticipantsSchema, Field()]


class CreateParticipantsTypeResponse(ApiBaseModel):
    participant_type: str
    schema_def: Annotated[ParticipantsSchema, Field()]


class UpdateParticipantsTypeRequest(ApiBaseModel):
    participant_type: str | None = None
    table_name: str | None = None
    fields: list[FieldDescriptor] | None = None


class UpdateParticipantsTypeResponse(ApiBaseModel):
    participant_type: str
    table_name: str
    fields: list[FieldDescriptor]


class ApiKeySummary(AdminApiBaseModel):
    id: Annotated[str, Field(...)]
    datasource_id: Annotated[str, Field(...)]
    organization_id: Annotated[str, Field(...)]
    organization_name: Annotated[str, Field(...)]


class ListApiKeysResponse(AdminApiBaseModel):
    items: list[ApiKeySummary]


class CreateApiKeyRequest(AdminApiBaseModel):
    datasource_id: Annotated[str, Field(...)]


class CreateApiKeyResponse(AdminApiBaseModel):
    id: Annotated[str, Field(...)]
    datasource_id: Annotated[str, Field(...)]
    key: Annotated[str, Field(...)]


class CreateUserRequest(AdminApiBaseModel):
    email: Annotated[str, Field(...)]
    organization_id: Annotated[str, Field(...)]
