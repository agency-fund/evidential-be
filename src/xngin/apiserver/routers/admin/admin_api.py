"""Implements a basic Admin API."""

import secrets  # noqa: I001
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Path,
    Query,
    Response,
    status,
)
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import QueryableAttribute, selectinload

from xngin.apiserver import constants, flags
from xngin.apiserver.apikeys import hash_key_or_raise, make_key
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.dns.safe_resolve import DnsLookupError, safe_resolve
from xngin.apiserver.dwh.queries import get_participant_metrics
from xngin.apiserver.dwh.inspections import (
    create_inspect_table_response_from_table,
)
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.models import tables
from xngin.apiserver.models.storage_format_converters import ExperimentStorageConverter
from xngin.apiserver.routers.admin.admin_api_types import (
    AddMemberToOrganizationRequest,
    AddWebhookToOrganizationRequest,
    AddWebhookToOrganizationResponse,
    ApiKeySummary,
    CreateApiKeyResponse,
    CreateDatasourceRequest,
    CreateDatasourceResponse,
    CreateOrganizationRequest,
    CreateOrganizationResponse,
    CreateParticipantsTypeRequest,
    CreateParticipantsTypeResponse,
    DatasourceSummary,
    EventSummary,
    GetDatasourceResponse,
    GetOrganizationResponse,
    InspectDatasourceResponse,
    InspectDatasourceTableResponse,
    InspectParticipantTypesResponse,
    ListApiKeysResponse,
    ListDatasourcesResponse,
    ListOrganizationEventsResponse,
    ListOrganizationsResponse,
    ListParticipantsTypeResponse,
    ListWebhooksResponse,
    OrganizationSummary,
    UpdateDatasourceRequest,
    UpdateOrganizationRequest,
    UpdateOrganizationWebhookRequest,
    UpdateParticipantsTypeRequest,
    UpdateParticipantsTypeResponse,
    UserSummary,
    WebhookSummary,
)
from xngin.apiserver.routers.auth.auth_dependencies import require_oidc_token
from xngin.apiserver.routers.auth.principal import Principal
from xngin.apiserver.routers.common_api_types import (
    ArmAnalysis,
    CreateExperimentRequest,
    CreateExperimentResponse,
    ExperimentAnalysis,
    ExperimentConfig,
    GetExperimentAssignmentsResponse,
    GetMetricsResponseElement,
    GetParticipantAssignmentResponse,
    GetStrataResponseElement,
    ListExperimentsResponse,
    MetricAnalysis,
    PowerRequest,
    PowerResponse,
)
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.routers.stateless.stateless_api import (
    power_check_impl,
    validate_schema_metrics_or_raise,
)
from xngin.apiserver.dwh.dwh_session import DwhSession, DwhDatabaseDoesNotExistError
from xngin.apiserver.settings import (
    ParticipantsConfig,
    ParticipantsDef,
    RemoteDatabaseConfig,
)
from xngin.apiserver.testing.testing_dwh import create_user_and_first_datasource
from xngin.stats.analysis import analyze_experiment as analyze_experiment_impl
from xngin.stats.stats_errors import StatsAnalysisError

GENERIC_SUCCESS = Response(status_code=status.HTTP_204_NO_CONTENT)
RESPONSE_CACHE_MAX_AGE_SECONDS = timedelta(minutes=15).seconds


class HTTPExceptionError(BaseModel):
    detail: str


# This defines the response codes we can expect our API to return in the normal course of operation and would be
# useful for our developers to think about.
#
# FastAPI will add a case for 422 (method argument or pydantic validation errors) automatically. 500s are
# intentionally omitted here as they (ideally) should never happen.
STANDARD_ADMIN_RESPONSES: dict[str | int, dict[str, Any]] = {
    # We return 400 when the client's request is invalid.
    "400": {"model": HTTPExceptionError, "description": "The request is invalid."},
    # We return 401 when the user presents an Authorization: header but it is not valid.
    "401": {
        "model": HTTPExceptionError,
        "description": "Authentication credentials are invalid.",
    },
    # 403s are returned by FastAPI's OpenIdConnect helper class when the Authorization: header is missing.
    # 403s are also returned by our code when the authenticated user doesn't have permission to perform the requested
    # action.
    "403": {
        "model": HTTPExceptionError,
        "description": "Requester does not have sufficient privileges to perform this operation or is not authenticated.",
    },
    # We return a 404 when a requested resource is not found. Authenticated users that do not have permission to know
    # whether a requested resource exists or not may also see a 404.
    "404": {
        "model": HTTPExceptionError,
        "description": "Requested content was not found.",
    },
}


def responses_factory(*codes):
    return {
        code: config
        for code, config in STANDARD_ADMIN_RESPONSES.items()
        if code in {str(c) for c in codes}
    }


def cache_is_fresh(updated: datetime | None):
    return updated and datetime.now(UTC) - updated < timedelta(minutes=5)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting router: {__name__} (prefix={router.prefix})")
    yield


router = APIRouter(
    lifespan=lifespan,
    prefix=constants.API_PREFIX_V1 + "/m",
    responses=STANDARD_ADMIN_RESPONSES,
    dependencies=[
        Depends(require_oidc_token)
    ],  # All routes in this router require authentication.
)


async def user_from_token(
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    token_info: Annotated[Principal, Depends(require_oidc_token)],
) -> tables.User:
    """Dependency for fetching the User record matching the authenticated user's email.

    This may raise a 400, 401, or 403.
    """
    result = await session.scalars(
        select(tables.User).filter(tables.User.email == token_info.email)
    )
    user = result.first()
    if user:
        return user

    if token_info.is_privileged():
        new_user = create_user_and_first_datasource(
            session,
            email=token_info.email,
            dsn=flags.XNGIN_DEVDWH_DSN,
            privileged=True,
        )
        await session.commit()
        return new_user

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"No user found with email: {token_info.email}",
    )


async def get_organization_or_raise(
    session: AsyncSession, user: tables.User, organization_id: str
):
    """Reads the requested organization from the database. Raises 404 if disallowed or not found."""
    stmt = (
        select(tables.Organization)
        .join(tables.UserOrganization)
        .where(tables.Organization.id == organization_id)
        .where(tables.UserOrganization.user_id == user.id)
    )
    result = await session.execute(stmt)
    org = result.scalar_one_or_none()
    if org is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found."
        )
    return org


async def get_datasource_or_raise(
    session: AsyncSession,
    user: tables.User,
    datasource_id: str,
    /,
    *,
    preload: list[QueryableAttribute] | None = None,
):
    """Reads the requested datasource from the database.

    Raises 404 if disallowed or not found.
    """
    stmt = (
        select(tables.Datasource)
        .join(tables.Organization)
        .join(tables.UserOrganization)
        .where(
            tables.UserOrganization.user_id == user.id,
            tables.Datasource.id == datasource_id,
        )
    )
    if preload:
        stmt = stmt.options(*[selectinload(f) for f in preload])
    result = await session.execute(stmt)
    ds = result.scalar_one_or_none()
    if ds is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Datasource not found."
        )
    return ds


async def get_experiment_via_ds_or_raise(
    session: AsyncSession,
    ds: tables.Datasource,
    experiment_id: str,
    *,
    preload: list[QueryableAttribute] | None = None,
) -> tables.Experiment:
    """Reads the requested experiment (related to the given datasource) from the database.

    The .arms attribute will be eagerly loaded due to its frequent use and small size.

    Raises 404 if not found.
    """
    stmt = (
        select(tables.Experiment)
        .options(selectinload(tables.Experiment.arms))
        .where(tables.Experiment.datasource_id == ds.id)
        .where(tables.Experiment.id == experiment_id)
    )
    if preload:
        stmt = stmt.options(*[selectinload(f) for f in preload])
    result = await session.execute(stmt)
    exp = result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found."
        )
    return exp


@router.get("/caller-identity")
async def caller_identity(
    token_info: Annotated[Principal, Depends(require_oidc_token)],
) -> Principal:
    """Returns basic metadata about the authenticated caller of this method."""
    return token_info


@router.get("/organizations")
async def list_organizations(
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
) -> ListOrganizationsResponse:
    """Returns a list of organizations that the authenticated user is a member of."""
    stmt = (
        select(tables.Organization)
        .join(tables.Organization.users)
        .where(tables.User.id == user.id)
        .order_by(tables.Organization.name)
    )
    organizations = await session.scalars(stmt)

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
async def create_organizations(
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
    body: Annotated[CreateOrganizationRequest, Body(...)],
) -> CreateOrganizationResponse:
    """Creates a new organization.

    Only users with an @agency.fund email address can create organizations.
    """
    if not user.is_privileged:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only privileged users can create organizations",
        )

    organization = tables.Organization(name=body.name)
    session.add(organization)
    organization.users.append(user)  # Add the creating user to the organization
    await session.commit()

    return CreateOrganizationResponse(id=organization.id)


@router.post("/organizations/{organization_id}/webhooks")
async def add_webhook_to_organization(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
    body: Annotated[AddWebhookToOrganizationRequest, Body(...)],
) -> AddWebhookToOrganizationResponse:
    """Adds a Webhook to an organization."""
    # Verify user has access to the organization
    org = await get_organization_or_raise(session, user, organization_id)

    # Generate a secure auth token
    auth_token = secrets.token_hex(16)

    # Create and save the webhook
    webhook = tables.Webhook(
        type=body.type, url=body.url, auth_token=auth_token, organization_id=org.id
    )
    session.add(webhook)
    await session.commit()

    return AddWebhookToOrganizationResponse(
        id=webhook.id, type=webhook.type, url=webhook.url, auth_token=auth_token
    )


@router.get("/organizations/{organization_id}/webhooks")
async def list_organization_webhooks(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
) -> ListWebhooksResponse:
    """Lists all the webhooks for an organization."""
    # Verify user has access to the organization
    org = await get_organization_or_raise(session, user, organization_id)

    # Query for webhooks
    stmt = select(tables.Webhook).where(tables.Webhook.organization_id == org.id)
    webhooks = await session.scalars(stmt)

    # Convert webhooks to WebhookSummary objects
    webhook_summaries = convert_webhooks_to_webhooksummaries(webhooks)

    return ListWebhooksResponse(items=webhook_summaries)


def convert_webhooks_to_webhooksummaries(webhooks):
    return [
        WebhookSummary(
            id=webhook.id,
            type=webhook.type,
            url=webhook.url,
            auth_token=webhook.auth_token,
        )
        for webhook in webhooks
    ]


@router.patch(
    "/organizations/{organization_id}/webhooks/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def update_organization_webhook(
    organization_id: str,
    webhook_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
    body: Annotated[UpdateOrganizationWebhookRequest, Body(...)],
):
    """Updates a webhook's URL in an organization."""
    # Verify user has access to the organization
    org = await get_organization_or_raise(session, user, organization_id)

    # Find the webhook
    webhook = (
        await session.execute(
            select(tables.Webhook).filter(
                tables.Webhook.id == webhook_id,
                tables.Webhook.organization_id == org.id,
            )
        )
    ).scalar_one_or_none()

    if webhook is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found"
        )

    # Update the webhook URL
    webhook.url = body.url
    await session.commit()
    return GENERIC_SUCCESS


@router.post(
    "/organizations/{organization_id}/webhooks/{webhook_id}/authtoken",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def regenerate_webhook_auth_token(
    organization_id: str,
    webhook_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
):
    """Regenerates the auth token for a webhook in an organization."""
    # Verify user has access to the organization
    org = await get_organization_or_raise(session, user, organization_id)

    # Find the webhook
    webhook = (
        await session.execute(
            select(tables.Webhook).filter(
                tables.Webhook.id == webhook_id,
                tables.Webhook.organization_id == org.id,
            )
        )
    ).scalar_one_or_none()

    if webhook is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found"
        )

    # Generate a new secure auth token
    webhook.auth_token = secrets.token_hex(16)
    await session.commit()
    return GENERIC_SUCCESS


@router.delete(
    "/organizations/{organization_id}/webhooks/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_webhook_from_organization(
    organization_id: str,
    webhook_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
):
    """Removes a Webhook from an organization."""
    # Verify user has access to the organization
    org = await get_organization_or_raise(session, user, organization_id)

    # Find and delete the webhook
    stmt = (
        delete(tables.Webhook)
        .where(tables.Webhook.id == webhook_id)
        .where(tables.Webhook.organization_id == org.id)
    )
    result = await session.execute(stmt)

    if result.rowcount == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found"
        )

    await session.commit()
    return GENERIC_SUCCESS


@router.get("/organizations/{organization_id}/events")
async def list_organization_events(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
) -> ListOrganizationEventsResponse:
    """Returns the most recent 200 events in an organization."""
    # Verify user has access to the organization
    org = await get_organization_or_raise(session, user, organization_id)

    # Query for the most recent 200 events
    stmt = (
        select(tables.Event)
        .where(tables.Event.organization_id == org.id)
        .order_by(tables.Event.created_at.desc())
        .limit(200)
    )
    events = await session.scalars(stmt)

    event_summaries = convert_events_to_eventsummaries(events)
    return ListOrganizationEventsResponse(items=event_summaries)


def convert_events_to_eventsummaries(events):
    event_summaries = []
    for event in events:
        data = event.get_data()
        event_summaries.append(
            EventSummary(
                id=event.id,
                created_at=event.created_at,
                type=event.type,
                summary=data.summarize() if data else "Unknown",
                link=data.link() if data else None,
                details=data.model_dump() if data else None,
            )
        )
    return event_summaries


@router.post(
    "/organizations/{organization_id}/members", status_code=status.HTTP_204_NO_CONTENT
)
async def add_member_to_organization(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
    body: Annotated[AddMemberToOrganizationRequest, Body(...)],
):
    """Adds a new member to an organization.

    The authenticated user must be part of the organization to add members.
    """
    # Check if the organization exists
    org = await session.get(
        tables.Organization,
        organization_id,
        options=[selectinload(tables.Organization.users)],
    )
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )

    if not user.is_privileged:
        # Verify user is a member of the organization
        _authz_check = await get_organization_or_raise(session, user, organization_id)

    # Add the new member
    result = await session.scalars(
        select(tables.User).filter(tables.User.email == body.email)
    )
    new_user = result.first()
    if not new_user:
        new_user = tables.User(email=body.email)
        session.add(new_user)

    org.users.append(new_user)
    await session.commit()
    return GENERIC_SUCCESS


@router.delete(
    "/organizations/{organization_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_member_from_organization(
    organization_id: str,
    user_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
):
    """Removes a member from an organization.

    The authenticated user must be part of the organization to remove members.
    """
    _authz_check = await get_organization_or_raise(session, user, organization_id)
    # Prevent users from removing themselves from an organization
    if user_id == user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot remove yourself from an organization",
        )
    stmt = delete(tables.UserOrganization).where(
        tables.UserOrganization.organization_id == organization_id,
        tables.UserOrganization.user_id == user_id,
    )
    result = await session.execute(stmt)
    if result.rowcount == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User is not a member of this organization",
        )

    await session.commit()
    return GENERIC_SUCCESS


@router.patch("/organizations/{organization_id}")
async def update_organization(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
    body: Annotated[UpdateOrganizationRequest, Body(...)],
):
    """Updates an organization's properties.

    The authenticated user must be a member of the organization.
    Currently only supports updating the organization name.
    """
    org = await get_organization_or_raise(session, user, organization_id)

    if body.name is not None:
        org.name = body.name

    await session.commit()
    return GENERIC_SUCCESS


@router.get("/organizations/{organization_id}")
async def get_organization(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
) -> GetOrganizationResponse:
    """Returns detailed information about a specific organization.

    The authenticated user must be a member of the organization.
    """
    # First get the organization and verify user has access
    org = await get_organization_or_raise(session, user, organization_id)

    # Get users and datasources separately
    users_stmt = (
        select(tables.User)
        .join(tables.UserOrganization)
        .filter(tables.UserOrganization.organization_id == organization_id)
    )
    users = await session.scalars(users_stmt)

    datasources_stmt = select(tables.Datasource).filter(
        tables.Datasource.organization_id == organization_id
    )
    datasources = await session.scalars(datasources_stmt)

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
                driver=ds.get_config().dwh.driver,
                type=ds.get_config().type,
                # Nit: Redundant in this response
                organization_id=ds.organization_id,
                organization_name=org.name,
            )
            for ds in sorted(datasources, key=lambda x: x.name)
        ],
    )


@router.get("/organizations/{organization_id}/datasources")
async def list_organization_datasources(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
) -> ListDatasourcesResponse:
    """Returns a list of datasources accessible to the authenticated user for an org."""
    _authz_check = await get_organization_or_raise(session, user, organization_id)
    stmt = (
        select(tables.Datasource)
        .join(tables.Organization)
        .join(tables.Organization.users)
        .where(tables.User.id == user.id)
    )
    if organization_id is not None:
        stmt = stmt.where(tables.Organization.id == organization_id)

    datasources = await session.scalars(stmt)

    def convert_ds_to_summary(ds: tables.Datasource) -> DatasourceSummary:
        config = ds.get_config()
        return DatasourceSummary(
            id=ds.id,
            name=ds.name,
            driver=config.dwh.driver,
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
async def create_datasource(
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
    body: Annotated[CreateDatasourceRequest, Body(...)],
) -> CreateDatasourceResponse:
    """Creates a new datasource for the specified organization."""
    org = await get_organization_or_raise(session, user, body.organization_id)

    if (
        body.dwh.driver == "bigquery"
        and body.dwh.credentials.type != "serviceaccountinfo"
    ):
        raise HTTPException(
            status_code=400,
            detail="BigQuery credentials must be specified using type=serviceaccountinfo",
        )
    if body.dwh.driver in {"postgresql+psycopg", "postgresql+psycopg2"}:
        try:
            safe_resolve(body.dwh.host)
        except DnsLookupError as err:
            raise HTTPException(
                status_code=400,
                detail="DNS resolution failed. Check datasource hostname and try again.",
            ) from err

    config = RemoteDatabaseConfig(participants=[], type="remote", dwh=body.dwh)

    datasource = tables.Datasource(name=body.name, organization_id=org.id).set_config(
        config
    )
    session.add(datasource)
    await session.commit()

    return CreateDatasourceResponse(id=datasource.id)


@router.patch("/datasources/{datasource_id}")
async def update_datasource(
    datasource_id: str,
    body: UpdateDatasourceRequest,
    user: Annotated[tables.User, Depends(user_from_token)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
):
    ds = await get_datasource_or_raise(session, user, datasource_id)
    if body.name is not None:
        ds.name = body.name
    if body.dwh is not None:
        cfg = ds.get_config()
        cfg.dwh = body.dwh

        # Invalidate cached inspections.
        ds.set_config(cfg)
        ds.clear_table_list()
        invalidate_inspect_tables = delete(tables.DatasourceTablesInspected).where(
            tables.DatasourceTablesInspected.datasource_id == datasource_id
        )
        invalidate_inspect_ptype = delete(tables.ParticipantTypesInspected).where(
            tables.ParticipantTypesInspected.datasource_id == datasource_id
        )
        await session.execute(invalidate_inspect_tables)
        await session.execute(invalidate_inspect_ptype)
    await session.commit()
    return GENERIC_SUCCESS


@router.get("/datasources/{datasource_id}")
async def get_datasource(
    datasource_id: str,
    user: Annotated[tables.User, Depends(user_from_token)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> GetDatasourceResponse:
    """Returns detailed information about a specific datasource."""
    ds = await get_datasource_or_raise(
        session, user, datasource_id, preload=[tables.Datasource.organization]
    )
    config = ds.get_config()
    return GetDatasourceResponse(
        id=ds.id,
        name=ds.name,
        config=config,
        organization_id=ds.organization_id,
        organization_name=ds.organization.name,
    )


@router.get("/datasources/{datasource_id}/inspect")
async def inspect_datasource(
    datasource_id: str,
    user: Annotated[tables.User, Depends(user_from_token)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
) -> InspectDatasourceResponse:
    """Verifies connectivity to a datasource and returns a list of readable tables."""
    ds = await get_datasource_or_raise(session, user, datasource_id)
    if not refresh and cache_is_fresh(ds.table_list_updated):
        return InspectDatasourceResponse(tables=ds.table_list)
    try:
        try:
            config = ds.get_config()
            with DwhSession(config.dwh) as dwh:
                tablenames = dwh.list_tables()

            ds.set_table_list(tablenames)
            await session.commit()
            return InspectDatasourceResponse(tables=tablenames)
        except DwhDatabaseDoesNotExistError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc
    except:
        ds.clear_table_list()
        await session.commit()
        raise


async def invalidate_inspect_table_cache(session, datasource_id):
    """Invalidates all table inspection cache entries for a datasource."""
    await session.execute(
        delete(tables.DatasourceTablesInspected).where(
            tables.DatasourceTablesInspected.datasource_id == datasource_id
        )
    )


@router.get("/datasources/{datasource_id}/inspect/{table_name}")
async def inspect_table_in_datasource(
    datasource_id: str,
    table_name: str,
    user: Annotated[tables.User, Depends(user_from_token)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
) -> InspectDatasourceTableResponse:
    """Inspects a single table in a datasource and returns a summary of its fields."""
    ds = await get_datasource_or_raise(session, user, datasource_id)
    if (
        not refresh
        and (
            cached := await session.get(
                tables.DatasourceTablesInspected, (datasource_id, table_name)
            )
        )
        and cache_is_fresh(cached.response_last_updated)
    ):
        return cached.get_response()

    config = ds.get_config()

    await invalidate_inspect_table_cache(session, datasource_id)
    await session.commit()

    with DwhSession(config.dwh) as dwh:
        # CannotFindTableError will be handled by exceptionhandlers.py.
        table = dwh.infer_table(table_name)
        response = create_inspect_table_response_from_table(table)

    session.add(
        tables.DatasourceTablesInspected(
            datasource_id=datasource_id, table_name=table_name
        ).set_response(response)
    )
    await session.commit()

    return response


@router.delete("/datasources/{datasource_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_datasource(
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
    datasource_id: Annotated[str, Path(...)],
):
    """Deletes a datasource.

    The user must be a member of the organization that owns the datasource.
    """
    # Delete the datasource, but only if the user has access to it
    stmt = (
        delete(tables.Datasource)
        .where(tables.Datasource.id == datasource_id)
        .where(
            tables.Datasource.id.in_(
                select(tables.Datasource.id)
                .join(tables.Organization)
                .join(tables.Organization.users)
                .where(tables.User.id == user.id)
            )
        )
    )
    await session.execute(stmt)
    await session.commit()

    return GENERIC_SUCCESS


@router.get("/datasources/{datasource_id}/participants")
async def list_participant_types(
    datasource_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
) -> ListParticipantsTypeResponse:
    ds = await get_datasource_or_raise(session, user, datasource_id)
    return ListParticipantsTypeResponse(
        items=list(
            sorted(ds.get_config().participants, key=lambda p: p.participant_type)
        )
    )


@router.post("/datasources/{datasource_id}/participants")
async def create_participant_type(
    datasource_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
    body: CreateParticipantsTypeRequest,
) -> CreateParticipantsTypeResponse:
    ds = await get_datasource_or_raise(session, user, datasource_id)
    participants_def = ParticipantsDef(
        type="schema",
        participant_type=body.participant_type,
        table_name=body.schema_def.table_name,
        fields=body.schema_def.fields,
    )
    config = ds.get_config()
    config.participants.append(participants_def)
    ds.set_config(config)
    await session.commit()
    return CreateParticipantsTypeResponse(
        participant_type=participants_def.participant_type,
        schema_def=body.schema_def,
    )


@router.get("/datasources/{datasource_id}/participants/{participant_id}/inspect")
async def inspect_participant_types(
    datasource_id: str,
    participant_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
) -> InspectParticipantTypesResponse:
    """Returns filter, strata, and metric field metadata for a participant type, including exemplars for filter fields."""
    ds = await get_datasource_or_raise(session, user, datasource_id)
    dsconfig = ds.get_config()
    # CannotFindParticipantsError will be handled by exceptionhandlers.
    pconfig = dsconfig.find_participants(participant_id)
    if pconfig.type == "sheet":
        raise HTTPException(
            status_code=405, detail="Sheet schemas cannot be inspected."
        )

    if (
        not refresh
        and (
            cached := await session.get(
                tables.ParticipantTypesInspected, (datasource_id, participant_id)
            )
        )
        and cache_is_fresh(cached.response_last_updated)
    ):
        return cached.get_response()

    await session.execute(
        delete(tables.ParticipantTypesInspected).where(
            tables.ParticipantTypesInspected.datasource_id == datasource_id,
            tables.ParticipantTypesInspected.participant_type == participant_id,
        )
    )
    await session.commit()

    def inspect_participant_types_impl() -> InspectParticipantTypesResponse:
        with DwhSession(dsconfig.dwh) as dwh:
            result = dwh.infer_table_with_descriptors(
                pconfig.table_name, pconfig.get_unique_id_field()
            )
            mapper = dwh.create_filter_meta_mapper(result.db_schema, result.sa_table)

        filter_fields = {c.field_name: c for c in pconfig.fields if c.is_filter}
        strata_fields = {c.field_name: c for c in pconfig.fields if c.is_strata}
        metric_cols = {c.field_name: c for c in pconfig.fields if c.is_metric}

        return InspectParticipantTypesResponse(
            metrics=sorted(
                [
                    GetMetricsResponseElement(
                        data_type=result.db_schema.get(col_name).data_type,
                        field_name=col_name,
                        description=col_descriptor.description,
                    )
                    for col_name, col_descriptor in metric_cols.items()
                    if result.db_schema.get(col_name)
                ],
                key=lambda item: item.field_name,
            ),
            strata=sorted(
                [
                    GetStrataResponseElement(
                        data_type=result.db_schema.get(field_name).data_type,
                        field_name=field_name,
                        description=field_descriptor.description,
                        # For strata columns, we will echo back any extra annotations
                        extra=field_descriptor.extra,
                    )
                    for field_name, field_descriptor in strata_fields.items()
                    if result.db_schema.get(field_name)
                ],
                key=lambda item: item.field_name,
            ),
            filters=sorted(
                [
                    mapper(field_name, field_descriptor)
                    for field_name, field_descriptor in filter_fields.items()
                    if result.db_schema.get(field_name)
                ],
                key=lambda item: item.field_name,
            ),
        )

    response = inspect_participant_types_impl()

    session.add(
        tables.ParticipantTypesInspected(
            datasource_id=datasource_id, participant_type=participant_id
        ).set_response(response)
    )
    await session.commit()

    return response


@router.get("/datasources/{datasource_id}/participants/{participant_id}")
async def get_participant_types(
    datasource_id: str,
    participant_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
) -> ParticipantsConfig:
    ds = await get_datasource_or_raise(session, user, datasource_id)
    # CannotFindParticipantsError will be handled by exceptionhandlers.
    return ds.get_config().find_participants(participant_id)


@router.patch(
    "/datasources/{datasource_id}/participants/{participant_id}",
    response_model=UpdateParticipantsTypeResponse,
)
async def update_participant_type(
    datasource_id: str,
    participant_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
    body: UpdateParticipantsTypeRequest,
):
    ds = await get_datasource_or_raise(session, user, datasource_id)
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

    # Invalidate the participant types cached inspections because the configuration may have been updated and they may
    # be stale.
    await session.execute(
        delete(tables.ParticipantTypesInspected).where(
            tables.ParticipantTypesInspected.datasource_id == datasource_id,
            tables.ParticipantTypesInspected.participant_type == participant_id,
        )
    )
    await session.commit()
    return UpdateParticipantsTypeResponse(
        participant_type=participant.participant_type,
        table_name=participant.table_name,
        fields=participant.fields,
    )


@router.delete(
    "/datasources/{datasource_id}/participants/{participant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_participant(
    datasource_id: str,
    participant_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
):
    ds = await get_datasource_or_raise(session, user, datasource_id)
    config = ds.get_config()
    participant = config.find_participants(participant_id)
    config.participants.remove(participant)
    ds.set_config(config)
    await session.commit()
    return GENERIC_SUCCESS


@router.get("/datasources/{datasource_id}/apikeys")
async def list_api_keys(
    datasource_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
) -> ListApiKeysResponse:
    """Returns API keys that have access to the datasource."""
    ds = await get_datasource_or_raise(
        session,
        user,
        datasource_id,
        preload=[tables.Datasource.api_keys, tables.Datasource.organization],
    )
    return ListApiKeysResponse(
        items=[
            ApiKeySummary(
                id=api_key.id,
                datasource_id=api_key.datasource_id,
                organization_id=ds.organization_id,
                organization_name=ds.organization.name,
            )
            for api_key in sorted(ds.api_keys, key=lambda a: a.id)
        ]
    )


@router.post("/datasources/{datasource_id}/apikeys")
async def create_api_key(
    datasource_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
) -> CreateApiKeyResponse:
    """Creates an API key for the specified datasource.

    The user must belong to the organization that owns the requested datasource.
    """
    ds = await get_datasource_or_raise(session, user, datasource_id)
    label, key = make_key()
    key_hash = hash_key_or_raise(key)
    api_key = tables.ApiKey(id=label, key=key_hash, datasource_id=ds.id)
    session.add(api_key)
    await session.commit()
    return CreateApiKeyResponse(id=label, datasource_id=ds.id, key=key)


@router.delete(
    "/datasources/{datasource_id}/apikeys/{api_key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_api_key(
    datasource_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
    api_key_id: Annotated[str, Path(...)],
):
    """Deletes the specified API key."""
    ds = await get_datasource_or_raise(
        session, user, datasource_id, preload=[tables.Datasource.api_keys]
    )
    ds.api_keys = [a for a in ds.api_keys if a.id != api_key_id]
    session.add(ds)
    await session.commit()
    return GENERIC_SUCCESS


@router.post("/datasources/{datasource_id}/experiments")
async def create_experiment(
    datasource_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
    body: CreateExperimentRequest,
    chosen_n: Annotated[
        int | None, Query(..., description="Number of participants to assign.")
    ] = None,
    stratify_on_metrics: Annotated[
        bool,
        Query(description="Whether to also stratify on metrics during assignment."),
    ] = True,
    random_state: Annotated[
        int | None,
        Query(
            description="Specify a random seed for reproducibility.",
            include_in_schema=False,
        ),
    ] = None,
) -> CreateExperimentResponse:
    datasource = await get_datasource_or_raise(session, user, datasource_id)
    if body.design_spec.ids_are_present():
        raise LateValidationError("Invalid DesignSpec: UUIDs must not be set.")
    ds_config = datasource.get_config()
    participants_cfg = ds_config.find_participants(body.design_spec.participant_type)
    if not isinstance(participants_cfg, ParticipantsDef):
        raise LateValidationError(
            "Invalid ParticipantsConfig: Participants must be of type schema."
        )

    # Get participants and their schema info from the client dwh
    participants = None
    with DwhSession(ds_config.dwh) as dwh:
        if chosen_n is not None:
            result = dwh.get_participants(
                participants_cfg.table_name, body.design_spec.filters, chosen_n
            )
            sa_table, participants = result.sa_table, result.participants
        elif body.design_spec.experiment_type == "preassigned":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Preassigned experiments must have a chosen_n.",
            )
        else:
            sa_table = dwh.infer_table(participants_cfg.table_name)

    return await experiments_common.create_experiment_impl(
        request=body,
        datasource_id=datasource.id,
        participant_unique_id_field=participants_cfg.get_unique_id_field(),
        dwh_sa_table=sa_table,
        dwh_participants=participants,
        random_state=random_state,
        xngin_session=session,
        stratify_on_metrics=stratify_on_metrics,
    )


@router.get("/datasources/{datasource_id}/experiments/{experiment_id}/analyze")
async def analyze_experiment(
    datasource_id: str,
    experiment_id: str,
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
    baseline_arm_id: Annotated[
        str | None,
        Query(
            description="UUID of the baseline arm. If None, the first design spec arm is used.",
        ),
    ] = None,
) -> ExperimentAnalysis:
    ds = await get_datasource_or_raise(xngin_session, user, datasource_id)
    dsconfig = ds.get_config()

    experiment = await get_experiment_via_ds_or_raise(
        xngin_session,
        ds,
        experiment_id,
        preload=[tables.Experiment.arm_assignments],
    )

    participants_cfg = dsconfig.find_participants(experiment.participant_type)
    if not isinstance(participants_cfg, ParticipantsDef):
        raise LateValidationError(
            "Invalid ParticipantsConfig: Participants must be of type schema."
        )
    unique_id_field = participants_cfg.get_unique_id_field()

    with DwhSession(dsconfig.dwh) as dwh:
        sa_table = dwh.infer_table(participants_cfg.table_name)

        design_spec = ExperimentStorageConverter(experiment).get_design_spec()
        metrics = design_spec.metrics
        assignments = experiment.arm_assignments
        participant_ids = [assignment.participant_id for assignment in assignments]
        if len(participant_ids) == 0:
            raise StatsAnalysisError("No participants found for experiment.")

        # Mark the start of the analysis as when we begin pulling outcomes.
        created_at = datetime.now(UTC)
        participant_outcomes = get_participant_metrics(
            dwh.session,
            sa_table,
            metrics,
            unique_id_field,
            participant_ids,
        )

    # We want to notify the user if there are participants assigned to the experiment that are not
    # in the data warehouse. E.g. in an online experiment, perhaps a new user was assigned
    # before their info was synced to the dwh.
    num_participants = len(participant_ids)
    num_missing_participants = num_participants - len(participant_outcomes)

    # Always assume the first arm is the baseline; UI can override this.
    baseline_arm_id = baseline_arm_id or design_spec.arms[0].arm_id
    analyze_results = analyze_experiment_impl(
        assignments, participant_outcomes, baseline_arm_id
    )

    metric_analyses = []
    for metric in design_spec.metrics:
        metric_name = metric.field_name
        arm_analyses = []
        for arm in experiment.arms:
            arm_result = analyze_results[metric_name][arm.id]
            arm_analyses.append(
                ArmAnalysis(
                    arm_id=arm.id,
                    arm_name=arm.name,
                    arm_description=arm.description,
                    is_baseline=arm_result.is_baseline,
                    estimate=arm_result.estimate,
                    p_value=arm_result.p_value,
                    t_stat=arm_result.t_stat,
                    std_error=arm_result.std_error,
                    num_missing_values=arm_result.num_missing_values,
                )
            )
        metric_analyses.append(
            MetricAnalysis(
                metric_name=metric_name, metric=metric, arm_analyses=arm_analyses
            )
        )
    return ExperimentAnalysis(
        experiment_id=experiment.id,
        metric_analyses=metric_analyses,
        num_participants=num_participants,
        num_missing_participants=num_missing_participants,
        created_at=created_at,
    )


EXPERIMENT_STATE_TRANSITION_RESPONSES: dict[int | str, dict[str, Any]] = {
    204: {"model": None, "description": "Experiment state updated successfully."},
    304: {"model": None, "description": "Experiment already in the target state."},
    400: {
        "model": HTTPExceptionError,
        "description": "Experiment is not in a valid state to transition to the target state.",
    },
}


@router.post(
    "/datasources/{datasource_id}/experiments/{experiment_id}/commit",
    responses=EXPERIMENT_STATE_TRANSITION_RESPONSES,
)
async def commit_experiment(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
):
    ds = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(session, ds, experiment_id)
    return await experiments_common.commit_experiment_impl(session, experiment)


@router.post(
    "/datasources/{datasource_id}/experiments/{experiment_id}/abandon",
    responses=EXPERIMENT_STATE_TRANSITION_RESPONSES,
)
async def abandon_experiment(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
):
    ds = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(session, ds, experiment_id)
    return await experiments_common.abandon_experiment_impl(session, experiment)


@router.get("/organizations/{organization_id}/experiments")
async def list_organization_experiments(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
) -> ListExperimentsResponse:
    """Returns a list of experiments in the organization."""
    org = await get_organization_or_raise(session, user, organization_id)
    return await experiments_common.list_organization_experiments_impl(session, org.id)


@router.get("/datasources/{datasource_id}/experiments/{experiment_id}")
async def get_experiment(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
) -> ExperimentConfig:
    """Returns the experiment with the specified ID."""
    ds = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(session, ds, experiment_id)
    converter = ExperimentStorageConverter(experiment)
    assign_summary = await experiments_common.get_assign_summary(
        session, experiment.id, converter.get_balance_check()
    )
    return converter.get_experiment_config(assign_summary)


@router.get("/datasources/{datasource_id}/experiments/{experiment_id}/assignments")
async def get_experiment_assignments(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
) -> GetExperimentAssignmentsResponse:
    ds = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(
        session, ds, experiment_id, preload=[tables.Experiment.arm_assignments]
    )
    return experiments_common.get_experiment_assignments_impl(experiment)


@router.get(
    "/datasources/{datasource_id}/experiments/{experiment_id}/assignments/csv",
    summary=(
        "Export experiment assignments as CSV file; BalanceCheck not included. "
        "csv header form: participant_id,arm_id,arm_name,strata_name1,strata_name2,..."
    ),
)
async def get_experiment_assignments_as_csv(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
) -> StreamingResponse:
    ds = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(
        session,
        ds,
        experiment_id,
        preload=[tables.Experiment.arms, tables.Experiment.arm_assignments],
    )
    return await experiments_common.get_experiment_assignments_as_csv_impl(experiment)


@router.get(
    "/datasources/{datasource_id}/experiments/{experiment_id}/assignments/{participant_id}",
    description="""Get the assignment for a specific participant, excluding strata if any.
    For 'preassigned' experiments, the participant's Assignment is returned if it exists.
    For 'online', returns the assignment if it exists, else generates an assignment.""",
)
async def get_experiment_assignment_for_participant(
    datasource_id: str,
    experiment_id: str,
    participant_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
    create_if_none: Annotated[
        bool,
        Query(
            description="Create an assignment if none exists. Does nothing for preassigned experiments. Override if you just want to check if an assignment exists."
        ),
    ] = True,
    random_state: Annotated[
        int | None,
        Query(
            description="Specify a random seed for reproducibility.",
            include_in_schema=False,
        ),
    ] = None,
) -> GetParticipantAssignmentResponse:
    """Get the assignment for a specific participant in an experiment."""
    # Validate the datasource and experiment exist
    ds = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(session, ds, experiment_id)

    # Look up the participant's assignment if it exists
    assignment = await experiments_common.get_existing_assignment_for_participant(
        session, experiment.id, participant_id
    )
    if not assignment and create_if_none and experiment.stopped_assignments_at is None:
        assignment = await experiments_common.create_assignment_for_participant(
            session, experiment, participant_id, random_state
        )

    return GetParticipantAssignmentResponse(
        experiment_id=experiment_id,
        participant_id=participant_id,
        assignment=assignment,
    )


@router.delete(
    "/datasources/{datasource_id}/experiments/{experiment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_experiment(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
):
    """Deletes the experiment with the specified ID."""
    ds = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(session, ds, experiment_id)
    await session.delete(experiment)
    await session.commit()
    return GENERIC_SUCCESS


@router.post("/datasources/{datasource_id}/power")
async def power_check(
    datasource_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(user_from_token)],
    body: PowerRequest,
) -> PowerResponse:
    ds = await get_datasource_or_raise(session, user, datasource_id)
    dsconfig = ds.get_config()
    participants_cfg = dsconfig.find_participants(body.design_spec.participant_type)
    validate_schema_metrics_or_raise(body.design_spec, participants_cfg)
    return power_check_impl(body, dsconfig, participants_cfg)
