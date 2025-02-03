"""Implements a basic Admin API."""
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated
import uuid

from fastapi import APIRouter, FastAPI, Depends, Path, Body, HTTPException
from starlette import status
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from xngin.apiserver.apikeys import make_key, hash_key
from xngin.apiserver.models.tables import (
    ApiKeyTable,
    ApiKeyDatasourceTable,
    User,
    Organization,
    Datasource,
)
from xngin.apiserver.settings import DatasourceConfig, RemoteDatabaseConfig
from xngin.apiserver.dependencies import settings_dependency, xngin_db_session
from xngin.apiserver.routers.oidc_dependencies import require_oidc_token, TokenInfo
from xngin.apiserver.settings import XnginSettings

logger = logging.getLogger(__name__)


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


class DatasourceSummary(AdminApiBaseModel):
    """Summary information about a datasource."""

    id: str
    name: str
    driver: str
    type: str


class ListDatasourcesResponse(AdminApiBaseModel):
    """Response model for the /datasources endpoint."""

    items: list[DatasourceSummary]


class ApiKey(AdminApiBaseModel):
    id: Annotated[str, Field(...)]
    datasource_ids: Annotated[list[str], Field(...)]
    key: Annotated[str | None, Field(...)] = None


class UpdateApiKeyRequest(AdminApiBaseModel):
    datasource_ids: Annotated[list[str], Field(..., min_length=1)]


class CreateUserRequest(AdminApiBaseModel):
    email: Annotated[str, Field(...)]


class CreateDatasourceRequest(AdminApiBaseModel):
    organization_id: Annotated[str, Field(...)]
    name: Annotated[str, Field(...)]
    config: Annotated[RemoteDatabaseConfig, Field(...)]


async def get_user_by_token(
    session: Annotated[Session, Depends(xngin_db_session)],
    token_info: Annotated[TokenInfo, Depends(require_oidc_token)],
) -> User:
    """Returns the User record matching the authenticated user's email.

    Raises:
        HTTPException: If no user is found with the email from the token.
    """
    user = session.query(User).filter(User.email == token_info.email).first()
    if not user:
        # TODO: use hd instead
        if token_info.email.endswith("@agency.fund"):
            user = User(email=token_info.email)
            session.add(user)
            session.commit()
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"No user found with email: {token_info.email}",
            )
    return user


@router.get("/caller-identity", response_model=TokenInfo)
def caller_identity(
    token_info: Annotated[TokenInfo, Depends(require_oidc_token)],
) -> TokenInfo:
    """Returns basic metadata about the authenticated caller of this method."""
    return token_info


@router.get("/datasources", response_model=ListDatasourcesResponse)
async def datasources(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(get_user_by_token)],
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
        # TODO: also return Org ID and Org Name
        return DatasourceSummary(
            id=ds.id, name=ds.name, driver=config.dwh.driver, type=config.type
        )

    return ListDatasourcesResponse(
        items=[convert_ds_to_summary(ds) for ds in datasources]
    )


# TODO: use new ListApiKeysResponse type; items= and include org name and org id
@router.get("/apikeys")
async def apikeys_list(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(get_user_by_token)],
) -> list[ApiKey]:
    """Returns API keys that the caller has access to via their organization memberships.

    An API key is visible if the user belongs to the organization that owns any of the
    datasources that the API key can access.
    """
    # Get API keys that have access to datasources owned by organizations the user belongs to
    stmt = (
        select(ApiKeyTable)
        .distinct()
        .join(ApiKeyTable.datasources)
        .join(ApiKeyDatasourceTable.datasource)
        .join(Organization)
        .join(Organization.users)
        .where(User.id == user.id)
        .options(selectinload(ApiKeyTable.datasources))
    )
    result = session.execute(stmt)
    api_keys = result.scalars().all()
    return [
        # TODO: link API Keys to a single org instead
        ApiKey(
            id=api_key.id,
            datasource_ids=[ds.datasource_id for ds in api_key.datasources],
            key=None,  # Omit key in list response
        )
        for api_key in api_keys
    ]


@router.post("/apikeys")
async def apikeys_create(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(get_user_by_token)],
    body: Annotated[UpdateApiKeyRequest, Body(...)],
) -> ApiKey:
    """Creates an API key for the requested datasources.

    The user must belong to the organizations that own all the requested datasources.

    Raises:
        HTTPException: If the user doesn't have access to any of the requested datasources.
    """
    # Verify user has access to all requested datasources
    stmt = (
        select(Datasource.id)
        .join(Organization)
        .join(Organization.users)
        .where(Datasource.id.in_(body.datasource_ids), User.id == user.id)
    )
    result = session.execute(stmt)
    accessible_ids = {row[0] for row in result}

    invalid_ids = set(body.datasource_ids) - accessible_ids
    if invalid_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You don't have access to datasources: {sorted(invalid_ids)}",
        )

    # Create the API key
    label, key = make_key()
    key_hash = hash_key(key)
    api_key = ApiKeyTable(id=label, key=key_hash)
    api_key.datasources = [
        ApiKeyDatasourceTable(datasource_id=ds_id) for ds_id in body.datasource_ids
    ]
    session.add(api_key)
    session.commit()
    return ApiKey(id=label, datasource_ids=body.datasource_ids, key=key)


@router.delete("/apikeys/{api_key_id}")
async def apikeys_delete(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(get_user_by_token)],
    api_key_id: Annotated[str, Path(...)],
):
    """Deletes the specified API key."""
    stmt = (
        delete(ApiKeyTable)
        .where(ApiKeyTable.id == api_key_id)
        .where(
            ApiKeyTable.id.in_(
                select(ApiKeyTable.id)
                .join(ApiKeyTable.datasources)
                .join(ApiKeyDatasourceTable.datasource)
                .join(Organization)
                .join(Organization.users)
                .where(User.id == user.id)
            )
        )
    )
    session.execute(stmt)
    session.commit()
    return {"status": "success"}


@router.post("/datasources")
async def datasources_create(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(get_user_by_token)],
    body: Annotated[CreateDatasourceRequest, Body(...)],
):
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
            detail="You are not a member of this organization"
        )

    if body.config.webhook_config:
        raise HTTPException(
            status_code=400, detail="Configuring webhooks is disallowed."
        )

    if (
        body.config.dwh.driver == "bigquery"
        and body.config.dwh.credentials.type != "serviceaccountinfo"
    ):
        raise HTTPException(
            status_code=400,
            detail="BigQuery credentials must be specified using type=serviceaccountinfo",
        )

    datasource = Datasource(
        id=str(uuid.uuid4()),
        name=body.name,
        organization_id=org.id,
        # TODO: for now, round-trip through model_dump_json() to persist SecretStr fields.
        config=json.loads(body.config.model_dump_json()),
    )
    session.add(datasource)
    session.commit()

    return {"status": "success", "id": datasource.id}


@router.post("/users/invites")
async def user_create(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(get_user_by_token)],
    body: Annotated[CreateUserRequest, Body(...)],
):
    """Creates a new user in the system.

    Only users with an agency.fund email address can create new users.

    Raises:
        HTTPException: If the caller's email is not from agency.fund domain.
    """
    # TODO: confirm with hd rather than email address
    if not user.email.endswith("@agency.fund"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only agency.fund users can create new users",
        )

    stmt = User(email=body.email)
    session.add(stmt)
    try:
        session.commit()
    except IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User already exists."
        )
    return {"status": "success"}


@router.patch("/apikeys/{api_key_id}")
async def apikeys_update(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(get_user_by_token)],
    api_key_id: Annotated[str, Path(...)],
    body: Annotated[UpdateApiKeyRequest, Body()],
) -> ApiKey:
    """Updates the list of datasources for the specified API key.

    The user must have access to both:
    1. The API key being modified (via their organization's datasources)
    2. All the requested datasources in the update

    Raises:
        HTTPException: If the API key is not found or the user doesn't have required access.
    """
    # First verify the API key exists and user has access to it
    stmt = (
        select(ApiKeyTable)
        .options(selectinload(ApiKeyTable.datasources))
        .join(ApiKeyTable.datasources)
        .join(ApiKeyDatasourceTable.datasource)
        .join(Organization)
        .join(Organization.users)
        .where(ApiKeyTable.id == api_key_id)
        .where(User.id == user.id)
    )
    result = session.execute(stmt)
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(
            status_code=404, detail="API key not found or access denied"
        )

    # Then verify user has access to all requested datasources
    accessible_ids = {
        datasource.id
        for org in user.organizations
        for datasource in org.datasources
        if datasource.id in body.datasource_ids
    }

    invalid_ids = set(body.datasource_ids) - accessible_ids
    if invalid_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You don't have access to datasources: {sorted(invalid_ids)}",
        )

    # Update the API key
    api_key.datasources = [
        ApiKeyDatasourceTable(datasource_id=ds_id) for ds_id in body.datasource_ids
    ]
    session.commit()
    return ApiKey(id=api_key_id, datasource_ids=body.datasource_ids)
