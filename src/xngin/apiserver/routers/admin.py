"""Implements a basic Admin API."""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, FastAPI, Depends, Path, Body, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from fastapi import Response
from fastapi import status

from xngin.apiserver.apikeys import make_key, hash_key
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.models.tables import (
    ApiKeyTable,
    User,
    Organization,
    Datasource,
)
from xngin.apiserver.routers.oidc_dependencies import require_oidc_token, TokenInfo
from xngin.apiserver.settings import RemoteDatabaseConfig, SqliteLocalConfig

logger = logging.getLogger(__name__)

GENERIC_SUCCESS = Response(status_code=status.HTTP_204_NO_CONTENT)


def is_enabled():
    """Feature flag: Returns true iff OIDC is enabled."""
    return os.environ.get("ENABLE_ADMIN", "").lower() in ("true", "1")


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


class AdminApiBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateOrganizationRequest(AdminApiBaseModel):
    name: Annotated[str, Field(...)]


class CreateOrganizationResponse(AdminApiBaseModel):
    id: Annotated[str, Field(...)]


class OrganizationSummary(AdminApiBaseModel):
    id: Annotated[str, Field(...)]
    name: Annotated[str, Field(...)]


class ListOrganizationsResponse(AdminApiBaseModel):
    items: list[OrganizationSummary]


class DatasourceSummary(AdminApiBaseModel):
    id: str
    name: str
    driver: str
    type: str
    organization_id: str
    organization_name: str


class ListDatasourcesResponse(AdminApiBaseModel):
    items: list[DatasourceSummary]


class CreateDatasourceRequest(AdminApiBaseModel):
    organization_id: Annotated[str, Field(...)]
    name: Annotated[str, Field(...)]
    config: Annotated[
        RemoteDatabaseConfig | SqliteLocalConfig, Field(discriminator="type")
    ]


class CreateDatasourceResponse(AdminApiBaseModel):
    id: Annotated[str, Field(...)]


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


async def get_user_from_token(
    session: Annotated[Session, Depends(xngin_db_session)],
    token_info: Annotated[TokenInfo, Depends(require_oidc_token)],
) -> User:
    """Returns the User record matching the authenticated user's email.

    Raises:
        HTTPException: If no user is found with the email from the token.
    """
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


@router.get("/caller-identity")
def caller_identity(
    token_info: Annotated[TokenInfo, Depends(require_oidc_token)],
) -> TokenInfo:
    """Returns basic metadata about the authenticated caller of this method."""
    return token_info


@router.get("/organizations")
async def organizations_list(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(get_user_from_token)],
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
            for org in organizations
        ]
    )


@router.post("/organizations")
async def organizations_create(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(get_user_from_token)],
    body: Annotated[CreateOrganizationRequest, Body(...)],
) -> CreateOrganizationResponse:
    """Creates a new organization.

    Only users with an agency.fund email address can create organizations.

    Raises:
        HTTPException: If the caller's email is not from agency.fund domain.
    """
    if not user.email.endswith("@agency.fund"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only agency.fund users can create organizations",
        )

    organization = Organization(name=body.name)
    session.add(organization)
    organization.users.append(user)  # Add the creating user to the organization
    session.commit()

    return CreateOrganizationResponse(id=organization.id)


@router.get("/datasources")
async def datasources_list(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(get_user_from_token)],
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
        config = RemoteDatabaseConfig.model_validate(ds.config)
        return DatasourceSummary(
            id=ds.id,
            name=ds.name,
            driver=config.dwh.driver,
            type=config.type,
            organization_id=ds.organization_id,
            organization_name=ds.organization.name,
        )

    return ListDatasourcesResponse(
        items=[convert_ds_to_summary(ds) for ds in datasources]
    )


@router.post("/datasources")
async def datasources_create(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(get_user_from_token)],
    body: Annotated[CreateDatasourceRequest, Body(...)],
) -> CreateDatasourceResponse:
    """Creates a new datasource for the specified organization."""
    # Verify organization exists
    org = session.get(Organization, body.organization_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    # Verify user is a member of this organization
    stmt = (
        select(User)
        .join(User.organizations)
        .where(User.id == user.id, Organization.id == body.organization_id)
    )
    result = session.execute(stmt)
    if not result.first():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a member of this organization",
        )

    if body.config.webhook_config:
        raise HTTPException(
            status_code=400, detail="Configuring webhooks is disallowed."
        )

    if body.config.type != "remote":
        raise HTTPException(400, detail='config.type must be "remote"')

    if (
        body.config.dwh.driver == "bigquery"
        and body.config.dwh.credentials.type != "serviceaccountinfo"
    ):
        raise HTTPException(
            status_code=400,
            detail="BigQuery credentials must be specified using type=serviceaccountinfo",
        )

    datasource = Datasource(
        name=body.name,
        organization_id=org.id,
        # TODO: for now, round-trip through model_dump_json() to persist SecretStr fields.
        config=json.loads(body.config.model_dump_json()),
    )
    session.add(datasource)
    session.commit()

    return CreateDatasourceResponse(id=datasource.id)


@router.delete("/datasources/{datasource_id}")
async def datasources_delete(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(get_user_from_token)],
    datasource_id: Annotated[str, Path(...)],
):
    """Deletes a datasource.

    The user must be a member of the organization that owns the datasource.

    Raises:
        HTTPException: If the datasource doesn't exist or the user doesn't have access to it.
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


@router.get("/apikeys")
async def apikeys_list(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(get_user_from_token)],
) -> ListApiKeysResponse:
    """Returns API keys that the caller has access to via their organization memberships.

    An API key is visible if the user belongs to the organization that owns any of the
    datasources that the API key can access.
    """
    # Get API keys that have access to datasources owned by organizations the user belongs to
    stmt = (
        select(ApiKeyTable)
        .distinct()
        .join(ApiKeyTable.datasource)
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
            for api_key in api_keys
        ]
    )


@router.post("/apikeys")
async def apikeys_create(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(get_user_from_token)],
    body: Annotated[CreateApiKeyRequest, Body(...)],
) -> CreateApiKeyResponse:
    """Creates an API key for the specified datasource.

    The user must belong to the organization that owns the requested datasource.

    Raises:
        HTTPException: If the user doesn't have access to the requested datasource.
    """
    # Verify user has access to the requested datasource
    stmt = (
        select(Datasource.id)
        .join(Organization)
        .join(Organization.users)
        .where(Datasource.id == body.datasource_id, User.id == user.id)
    )
    result = session.execute(stmt)
    if not result.first():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You don't have access to datasource: {body.datasource_id}",
        )

    # Create the API key
    label, key = make_key()
    key_hash = hash_key(key)
    api_key = ApiKeyTable(id=label, key=key_hash, datasource_id=body.datasource_id)
    session.add(api_key)
    session.commit()
    return CreateApiKeyResponse(id=label, datasource_id=body.datasource_id, key=key)


@router.delete("/apikeys/{api_key_id}")
async def apikeys_delete(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(get_user_from_token)],
    api_key_id: Annotated[str, Path(...)],
):
    """Deletes the specified API key."""
    stmt = (
        delete(ApiKeyTable)
        .where(ApiKeyTable.id == api_key_id)
        .where(
            ApiKeyTable.id.in_(
                select(ApiKeyTable.id)
                .join(ApiKeyTable.datasource)
                .join(Organization)
                .join(Organization.users)
                .where(User.id == user.id)
            )
        )
    )
    session.execute(stmt)
    session.commit()
    return GENERIC_SUCCESS


@router.post("/users/invites")
async def user_invite_create(
    session: Annotated[Session, Depends(xngin_db_session)],
    body: Annotated[CreateUserRequest, Body(...)],
    token_info: Annotated[TokenInfo, Depends(require_oidc_token)],
):
    """Creates a new user in the system.

    Only privileged callers can invoke this method.
    """
    if not token_info.is_privileged():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only privileged users can create new users",
        )

    new_user = User(email=body.email)
    session.add(new_user)
    try:
        session.commit()
    except IntegrityError as ierr:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="User already exists."
        ) from ierr
    return GENERIC_SUCCESS
