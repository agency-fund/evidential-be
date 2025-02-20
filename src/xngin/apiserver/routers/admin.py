"""Implements a basic Admin API."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

import google.api_core.exceptions
import sqlalchemy
from fastapi import APIRouter, FastAPI, Depends, Path, Body, HTTPException
from fastapi import Response
from fastapi import status
import sqlalchemy.orm
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from xngin.apiserver import flags, settings
from xngin.apiserver.api_types import ApiBaseModel, DataType
from xngin.apiserver.apikeys import make_key, hash_key
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.models.tables import (
    ApiKey,
    User,
    Organization,
    Datasource,
    UserOrganization,
)
from xngin.apiserver.routers.oidc_dependencies import require_oidc_token, TokenInfo
from xngin.apiserver.settings import (
    RemoteDatabaseConfig,
    SqliteLocalConfig,
    Dwh,
    ParticipantsDef,
    ParticipantsConfig,
    DatasourceConfig,
)
from xngin.schema.schema_types import ParticipantsSchema, FieldDescriptor

logger = logging.getLogger(__name__)

GENERIC_SUCCESS = Response(status_code=status.HTTP_204_NO_CONTENT)


def is_enabled():
    """Feature flag: Returns true iff OIDC is enabled."""
    return flags.ENABLE_ADMIN


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


router = APIRouter(
    lifespan=lifespan,
    prefix="/m",
    dependencies=[
        Depends(require_oidc_token)
    ],  # All routes in this router require authentication.
)


def user_from_token(
    session: Annotated[Session, Depends(xngin_db_session)],
    token_info: Annotated[TokenInfo, Depends(require_oidc_token)],
) -> User:
    """Dependency for fetching the User record matching the authenticated user's email."""
    user = session.query(User).filter(User.email == token_info.email).first()
    if not user:
        # Privileged users will be created on the fly.
        if token_info.is_privileged():
            user = User(email=token_info.email)
            session.add(user)
            session.commit()
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"No user found with email: {token_info.email}",
            )
    return user


def get_organization_or_raise(session: Session, user: User, organization_id: str):
    """Reads the requested organization from the database. Raises if disallowed or not found."""
    stmt = (
        select(Organization)
        .join(UserOrganization)
        .where(Organization.id == organization_id)
        .where(UserOrganization.user_id == user.id)
    )
    org = session.execute(stmt).scalar_one_or_none()
    if org is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Datasource not found."
        )
    return org


def get_datasource_or_raise(session: Session, user: User, datasource_id: str):
    """Reads the requested datasource from the database. Raises if disallowed or not found."""
    stmt = (
        select(Datasource)
        .join(Organization)
        .join(UserOrganization)
        .where(UserOrganization.user_id == user.id, Datasource.id == datasource_id)
    )
    ds = session.execute(stmt).scalar_one_or_none()
    if ds is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Datasource not found."
        )
    return ds


class AdminApiBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateOrganizationRequest(AdminApiBaseModel):
    name: Annotated[str, Field(...)]


class CreateOrganizationResponse(AdminApiBaseModel):
    id: Annotated[str, Field(...)]


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


@router.get("/caller-identity")
def caller_identity(
    token_info: Annotated[TokenInfo, Depends(require_oidc_token)],
) -> TokenInfo:
    """Returns basic metadata about the authenticated caller of this method."""
    return token_info


@router.get("/organizations")
def list_organizations(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
) -> ListOrganizationsResponse:
    """Returns a list of organizations that the authenticated user is a member of."""
    stmt = select(Organization).join(Organization.users).where(User.id == user.id)
    result = session.execute(stmt)
    organizations = result.scalars().all()

    return ListOrganizationsResponse(
        items=[
            OrganizationSummary(
                id=org.id,
                name=org.name,
            )
            for org in sorted(organizations, key=lambda o: o.name)
        ]
    )


@router.post("/organizations")
def create_organizations(
    session: Annotated[Session, Depends(xngin_db_session)],
    token_info: Annotated[TokenInfo, Depends(require_oidc_token)],
    user: Annotated[User, Depends(user_from_token)],
    body: Annotated[CreateOrganizationRequest, Body(...)],
) -> CreateOrganizationResponse:
    """Creates a new organization.

    Only users with an @agency.fund email address can create organizations.
    """
    if not token_info.is_privileged():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only privileged users can create organizations",
        )

    organization = Organization(name=body.name)
    session.add(organization)
    organization.users.append(user)  # Add the creating user to the organization
    session.commit()

    return CreateOrganizationResponse(id=organization.id)


@router.post(
    "/organizations/{organization_id}/members", status_code=status.HTTP_204_NO_CONTENT
)
def add_member_to_organization(
    organization_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    token_info: Annotated[TokenInfo, Depends(require_oidc_token)],
    user: Annotated[User, Depends(user_from_token)],
    body: Annotated[AddMemberToOrganizationRequest, Body(...)],
):
    """Adds a new member to an organization.

    The authenticated user must be part of the organization to add members.
    """
    # Check if the organization exists
    org = session.get(Organization, organization_id)
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )

    if not token_info.is_privileged():
        # Verify user belongs to the organization
        stmt = (
            select(UserOrganization)
            .where(UserOrganization.user_id == user.id)
            .where(UserOrganization.organization_id == organization_id)
        )
        is_member = session.execute(stmt).scalar_one_or_none()
        if not is_member:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to add members to this organization",
            )

    # Add the new member
    new_user = session.query(User).filter(User.email == body.email).first()
    if not new_user:
        new_user = User(email=body.email)
        session.add(new_user)

    org.users.append(new_user)
    session.commit()
    return GENERIC_SUCCESS


@router.delete(
    "/organizations/{organization_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def remove_member_from_organization(
    organization_id: str,
    user_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
):
    """Removes a member from an organization.

    The authenticated user must be part of the organization to remove members.
    """
    get_organization_or_raise(session, user, organization_id)
    stmt = delete(UserOrganization).where(
        UserOrganization.organization_id == organization_id,
        UserOrganization.user_id == user_id,
    )
    result = session.execute(stmt)
    if result.rowcount == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User is not a member of this organization",
        )

    session.commit()
    return GENERIC_SUCCESS


@router.get("/organizations/{organization_id}")
def get_organization(
    organization_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
) -> GetOrganizationResponse:
    """Returns detailed information about a specific organization.

    The authenticated user must be a member of the organization.
    """
    # First get the organization and verify user has access
    org = get_organization_or_raise(session, user, organization_id)
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )

    # Get users and datasources separately
    users = (
        session.query(User)
        .join(UserOrganization)
        .filter(UserOrganization.organization_id == organization_id)
        .all()
    )
    datasources = (
        session.query(Datasource)
        .filter(Datasource.organization_id == organization_id)
        .all()
    )

    return GetOrganizationResponse(
        id=org.id,
        name=org.name,
        users=[
            UserSummary(id=u.id, email=u.email)
            for u in sorted(users, key=lambda x: x.email)
        ],
        datasources=[
            DatasourceSummary(
                id=ds.id,
                name=ds.name,
                driver="sqlite"
                if isinstance(ds.get_config(), SqliteLocalConfig)
                else ds.get_config().dwh.driver,
                type=ds.get_config().type,
                # Nit: Redundant in this response
                organization_id=ds.organization_id,
                organization_name=org.name,
            )
            for ds in sorted(datasources, key=lambda x: x.name)
        ],
    )


@router.get("/datasources")
def list_datasources(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
) -> ListDatasourcesResponse:
    """Returns a list of datasources accessible to the authenticated user."""
    stmt = (
        select(Datasource)
        .join(Organization)
        .join(Organization.users)
        .where(User.id == user.id)
    )
    result = session.execute(stmt)
    datasources = result.scalars().all()

    def convert_ds_to_summary(ds: Datasource) -> DatasourceSummary:
        config = ds.get_config()
        return DatasourceSummary(
            id=ds.id,
            name=ds.name,
            driver="sqlite"
            if isinstance(config, SqliteLocalConfig)
            else config.dwh.driver,
            type=config.type,
            organization_id=ds.organization_id,
            organization_name=ds.organization.name,
        )

    return ListDatasourcesResponse(
        items=[
            convert_ds_to_summary(ds)
            for ds in sorted(datasources, key=lambda d: d.name)
        ]
    )


@router.post("/datasources")
def create_datasource(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
    body: Annotated[CreateDatasourceRequest, Body(...)],
) -> CreateDatasourceResponse:
    """Creates a new datasource for the specified organization."""
    org = session.get(Organization, body.organization_id)
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )

    stmt = (
        select(UserOrganization)
        .where(UserOrganization.user_id == user.id)
        .where(UserOrganization.organization_id == org.id)
    )
    allowed = session.execute(stmt).scalar_one_or_none()
    if allowed is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a member of this organization",
        )

    if (
        body.dwh.driver == "bigquery"
        and body.dwh.credentials.type != "serviceaccountinfo"
    ):
        raise HTTPException(
            status_code=400,
            detail="BigQuery credentials must be specified using type=serviceaccountinfo",
        )

    config = RemoteDatabaseConfig(participants=[], type="remote", dwh=body.dwh)

    datasource = Datasource(name=body.name, organization_id=org.id)
    datasource.set_config(config)
    session.add(datasource)
    session.commit()

    return CreateDatasourceResponse(id=datasource.id)


@router.patch("/datasources/{datasource_id}")
def update_datasource(
    datasource_id: str,
    body: UpdateDatasourceRequest,
    user: Annotated[User, Depends(user_from_token)],
    session: Annotated[Session, Depends(xngin_db_session)],
):
    ds = get_datasource_or_raise(session, user, datasource_id)
    if body.name is not None:
        ds.name = body.name
    if body.dwh is not None:
        cfg = ds.get_config()
        cfg.dwh = body.dwh
        ds.set_config(cfg)
    session.commit()
    return GENERIC_SUCCESS


@router.get("/datasources/{datasource_id}")
def get_datasource(
    datasource_id: str,
    user: Annotated[User, Depends(user_from_token)],
    session: Annotated[Session, Depends(xngin_db_session)],
) -> GetDatasourceResponse:
    """Returns detailed information about a specific datasource."""
    ds = get_datasource_or_raise(session, user, datasource_id)
    config = ds.get_config()
    return GetDatasourceResponse(
        id=ds.id,
        name=ds.name,
        config=config,
        organization_id=ds.organization_id,
        organization_name=ds.organization.name,
    )


@router.get("/datasources/{datasource_id}/inspect")
def inspect_datasource(
    datasource_id: str,
    user: Annotated[User, Depends(user_from_token)],
    session: Annotated[Session, Depends(xngin_db_session)],
) -> InspectDatasourceResponse:
    """Verifies connectivity to a datasource and returns a list of readable tables."""
    ds = get_datasource_or_raise(session, user, datasource_id)
    try:
        config = ds.get_config()
        metadata = sqlalchemy.MetaData()
        metadata.reflect(config.dbengine())
        tables = list(sorted(metadata.tables.keys()))
        return InspectDatasourceResponse(tables=tables)
    except google.api_core.exceptions.NotFound as exc:
        # Google returns a 404 when authentication succeeds but when the specified datasource does not exist.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


def create_inspect_table_response_from_table(table: sqlalchemy.Table):
    """Creates an InspectDatasourceTableResponse from a sqlalchemy.Table.

    This is similar to config_sheet.create_schema_from_table but tailored to use in the API.
    """
    possible_id_columns = {
        c.name for c in table.columns.values() if c.name.endswith("id")
    }
    primary_key_columns = {c.name for c in table.columns.values() if c.primary_key}
    if len(primary_key_columns) > 0:
        # If there is more than one PK, it probably isn't usable for experiments.
        primary_key_columns = set()
    possible_id_columns |= primary_key_columns

    collected = []
    for column in table.columns.values():
        type_hint = column.type
        collected.append(
            FieldMetadata(
                field_name=column.name,
                data_type=DataType.match(type_hint),
                description=column.comment if column.comment else "",
            )
        )

    return InspectDatasourceTableResponse(
        detected_unique_id_fields=list(sorted(possible_id_columns)),
        fields=list(sorted(collected, key=lambda f: f.field_name)),
    )


@router.get("/datasources/{datasource_id}/inspect/{table_name}")
def inspect_table_in_datasource(
    datasource_id: str,
    table_name: str,
    user: Annotated[User, Depends(user_from_token)],
    session: Annotated[Session, Depends(xngin_db_session)],
) -> InspectDatasourceTableResponse:
    """Inspects a single table in a datasource and returns a summary of its fields."""
    ds = get_datasource_or_raise(session, user, datasource_id)
    config = ds.get_config()
    engine = config.dbengine()
    # CannotFindTableError will be handled by exceptionhandlers.py.
    try:
        table = settings.infer_table(engine, table_name, use_reflection=False)
    except sqlalchemy.exc.ProgrammingError:
        table = settings.infer_table(engine, table_name, use_reflection=True)
    return create_inspect_table_response_from_table(table)


@router.delete("/datasources/{datasource_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_datasource(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
    datasource_id: Annotated[str, Path(...)],
):
    """Deletes a datasource.

    The user must be a member of the organization that owns the datasource.
    """
    # Delete the datasource, but only if the user has access to it
    stmt = (
        delete(Datasource)
        .where(Datasource.id == datasource_id)
        .where(
            Datasource.id.in_(
                select(Datasource.id)
                .join(Organization)
                .join(Organization.users)
                .where(User.id == user.id)
            )
        )
    )
    session.execute(stmt)
    session.commit()

    return GENERIC_SUCCESS


@router.get("/datasources/{datasource_id}/participants")
def list_participant_types(
    datasource_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
) -> ListParticipantsTypeResponse:
    ds = get_datasource_or_raise(session, user, datasource_id)
    return ListParticipantsTypeResponse(
        items=list(
            sorted(ds.get_config().participants, key=lambda p: p.participant_type)
        )
    )


@router.post("/datasources/{datasource_id}/participants")
def create_participant_type(
    datasource_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
    body: CreateParticipantsTypeRequest,
) -> CreateParticipantsTypeResponse:
    ds = get_datasource_or_raise(session, user, datasource_id)
    participants_def = ParticipantsDef(
        type="schema",
        participant_type=body.participant_type,
        table_name=body.schema_def.table_name,
        fields=body.schema_def.fields,
    )
    config = ds.get_config()
    config.participants.append(participants_def)
    ds.set_config(config)
    session.commit()
    return CreateParticipantsTypeResponse(
        participant_type=participants_def.participant_type,
        schema_def=body.schema_def,
    )


@router.get("/datasources/{datasource_id}/participants/{participant_id}")
def get_participant_types(
    datasource_id: str,
    participant_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
) -> ParticipantsConfig:
    ds = get_datasource_or_raise(session, user, datasource_id)
    # CannotFindParticipantsError will be handled by exceptionhandlers.
    return ds.get_config().find_participants(participant_id)


@router.patch(
    "/datasources/{datasource_id}/participants/{participant_id}",
    response_model=UpdateParticipantsTypeResponse,
)
def update_participant_type(
    datasource_id: str,
    participant_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
    body: UpdateParticipantsTypeRequest,
):
    ds = get_datasource_or_raise(session, user, datasource_id)
    config = ds.get_config()
    participant = config.find_participants(participant_id)
    config.participants.remove(participant)
    if not isinstance(participant, ParticipantsDef):
        return Response(
            status_code=405, content="Only schema participants can be updated"
        )
    if body.participant_type is not None:
        participant.participant_type = body.participant_type
    if body.table_name is not None:
        participant.table_name = body.table_name
    if body.fields is not None:
        participant.fields = body.fields
    config.participants.append(participant)
    ds.set_config(config)
    session.commit()
    return UpdateParticipantsTypeResponse(
        participant_type=participant.participant_type,
        table_name=participant.table_name,
        fields=participant.fields,
    )


@router.delete(
    "/datasources/{datasource_id}/participants/{participant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_participant(
    datasource_id: str,
    participant_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
):
    ds = get_datasource_or_raise(session, user, datasource_id)
    config = ds.get_config()
    participant = config.find_participants(participant_id)
    config.participants.remove(participant)
    ds.set_config(config)
    session.commit()
    return GENERIC_SUCCESS


@router.get("/apikeys")
def list_api_keys(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
) -> ListApiKeysResponse:
    """Returns API keys that the caller has access to via their organization memberships.

    An API key is visible if the user belongs to the organization that owns any of the
    datasources that the API key can access.
    """
    # Get API keys that have access to datasources owned by organizations the user belongs to
    stmt = (
        select(ApiKey)
        .distinct()
        .join(ApiKey.datasource)
        .join(Organization)
        .join(Organization.users)
        .where(User.id == user.id)
    )
    result = session.execute(stmt)
    api_keys = result.scalars().all()
    return ListApiKeysResponse(
        items=[
            ApiKeySummary(
                id=api_key.id,
                datasource_id=api_key.datasource_id,
                organization_id=api_key.datasource.organization_id,
                organization_name=api_key.datasource.organization.name,
            )
            for api_key in sorted(api_keys, key=lambda a: a.id)
        ]
    )


@router.post("/apikeys")
def create_api_key(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
    body: Annotated[CreateApiKeyRequest, Body(...)],
) -> CreateApiKeyResponse:
    """Creates an API key for the specified datasource.

    The user must belong to the organization that owns the requested datasource.
    """
    ds = get_datasource_or_raise(session, user, body.datasource_id)
    label, key = make_key()
    key_hash = hash_key(key)
    api_key = ApiKey(id=label, key=key_hash, datasource_id=ds.id)
    session.add(api_key)
    session.commit()
    return CreateApiKeyResponse(id=label, datasource_id=ds.id, key=key)


@router.delete("/apikeys/{api_key_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_api_key(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
    api_key_id: Annotated[str, Path(...)],
):
    """Deletes the specified API key."""
    stmt = (
        delete(ApiKey)
        .where(ApiKey.id == api_key_id)
        .where(
            ApiKey.id.in_(
                select(ApiKey.id)
                .join(ApiKey.datasource)
                .join(Organization)
                .join(Organization.users)
                .where(User.id == user.id)
            )
        )
    )
    session.execute(stmt)
    session.commit()
    return GENERIC_SUCCESS
