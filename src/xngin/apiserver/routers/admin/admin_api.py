"""
This module defines the internal Evidential UI-facing Admin API endpoints.
(See experiments_api.py for integrator-facing endpoints.)
"""

import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, assert_never

from fastapi import (
    APIRouter,
    BackgroundTasks,
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
from xngin.apiserver.dwh.dwh_session import (
    DwhSession,
    NoDwh,
)
from xngin.apiserver.dwh.inspection_types import ParticipantsSchema
from xngin.apiserver.dwh.inspections import (
    create_inspect_table_response_from_table,
)
from xngin.apiserver.dwh.queries import (
    get_stats_on_filters,
    get_stats_on_metrics,
)
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.admin import admin_api_converters, admin_common, authz
from xngin.apiserver.routers.admin.admin_api_converters import (
    api_dsn_to_settings_dwh,
    convert_api_snapshot_status_to_snapshot_status,
    convert_snapshot_to_api_snapshot,
)
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
    CreateSnapshotResponse,
    DatasourceSummary,
    EventSummary,
    GetDatasourceResponse,
    GetOrganizationResponse,
    GetSnapshotResponse,
    InspectDatasourceResponse,
    InspectDatasourceTableResponse,
    InspectParticipantTypesResponse,
    ListApiKeysResponse,
    ListDatasourcesResponse,
    ListOrganizationEventsResponse,
    ListOrganizationsResponse,
    ListParticipantsTypeResponse,
    ListSnapshotsResponse,
    ListWebhooksResponse,
    OrganizationSummary,
    PostgresDsn,
    RedshiftDsn,
    SnapshotStatus,
    UpdateArmRequest,
    UpdateDatasourceRequest,
    UpdateExperimentRequest,
    UpdateOrganizationRequest,
    UpdateOrganizationWebhookRequest,
    UpdateParticipantsTypeRequest,
    UpdateParticipantsTypeResponse,
    UserSummary,
    WebhookSummary,
)
from xngin.apiserver.routers.admin.admin_common import create_organization_impl
from xngin.apiserver.routers.admin.generic_handlers import handle_delete
from xngin.apiserver.routers.auth.auth_api_types import CallerIdentity
from xngin.apiserver.routers.auth.auth_dependencies import require_user_from_token
from xngin.apiserver.routers.common_api_types import (
    BaseBanditExperimentSpec,
    BaseFrequentistDesignSpec,
    CMABContextInputRequest,
    ContextInput,
    ContextType,
    CreateExperimentRequest,
    CreateExperimentResponse,
    ExperimentAnalysisResponse,
    ExperimentsType,
    GetExperimentAssignmentsResponse,
    GetExperimentResponse,
    GetMetricsResponseElement,
    GetParticipantAssignmentResponse,
    GetStrataResponseElement,
    ListExperimentsResponse,
    PowerRequest,
    PowerResponse,
)
from xngin.apiserver.routers.common_enums import ExperimentState
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.settings import (
    ParticipantsConfig,
    ParticipantsDef,
    RemoteDatabaseConfig,
)
from xngin.apiserver.snapshots import snapshotter
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter
from xngin.stats import check_power

GENERIC_SUCCESS = Response(status_code=status.HTTP_204_NO_CONTENT)
RESPONSE_CACHE_MAX_AGE_SECONDS = timedelta(minutes=15).seconds


# Describes the structure of an error raised via `raise HTTPException`.
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
        "description": (
            "Requester does not have sufficient privileges to perform this operation or is not authenticated."
        ),
    },
    # We return a 404 when a requested resource is not found. Authenticated users that do not have permission to know
    # whether a requested resource exists or not may also see a 404.
    "404": {
        "model": HTTPExceptionError,
        "description": "Requested content was not found.",
    },
}


def cache_is_fresh(updated: datetime | None):
    return updated is not None and datetime.now(UTC) - updated < timedelta(minutes=5)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting router: {__name__} (prefix={router.prefix})")
    yield


@asynccontextmanager
async def clear_db_table_cache_on_error(session: AsyncSession, datasource: tables.Datasource):
    """Context manager that clears a datasource's cached table list on error."""
    try:
        yield
    except:
        datasource.clear_table_list()
        await session.commit()
        raise


router = APIRouter(
    lifespan=lifespan,
    prefix=constants.API_PREFIX_V1 + "/m",
    responses=STANDARD_ADMIN_RESPONSES,
    dependencies=[Depends(require_user_from_token)],  # All routes in this router require authentication.
)


async def get_organization_or_raise(session: AsyncSession, user: tables.User, organization_id: str):
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")
    return org


async def get_datasource_or_raise(
    session: AsyncSession,
    user: tables.User,
    datasource_id: str,
    /,
    *,
    organization_id: str | None = None,
    preload: list[QueryableAttribute] | None = None,
) -> tables.Datasource:
    """Reads the requested datasource from the database.

    Requests that accept organization_id should also pass organization_id= kwarg.

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
    if organization_id:
        stmt = stmt.where(tables.Organization.id == organization_id)
    if preload:
        stmt = stmt.options(*[selectinload(f) for f in preload])
    result = await session.execute(stmt)
    ds = result.scalar_one_or_none()
    if ds is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Datasource not found.")
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found.")
    return exp


async def validate_webhooks(
    session: AsyncSession, organization_id: str, request_webhooks: list[str]
) -> list[tables.Webhook]:
    # Validate webhook IDs exist and belong to organization
    validated_webhooks = []
    if request_webhooks:
        webhooks = await session.scalars(
            select(tables.Webhook)
            .where(tables.Webhook.id.in_(request_webhooks))
            .where(tables.Webhook.organization_id == organization_id)
        )
        validated_webhooks = list(webhooks)
        found_webhook_ids = {w.id for w in validated_webhooks}
        missing_webhook_ids = set(request_webhooks) - found_webhook_ids
        if missing_webhook_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid webhook IDs: {sorted(missing_webhook_ids)}",
            )
    return validated_webhooks


def sort_contexts_by_id_or_raise(context_defns: list[tables.Context], context_inputs: list[ContextInput]):
    context_defns = sorted(context_defns, key=lambda c: c.id)
    context_inputs = sorted(context_inputs, key=lambda c: c.context_id)

    if len(context_inputs) != len(context_defns):
        raise LateValidationError(
            f"Expected {len(context_defns)} context inputs, but got {len(context_inputs)} in "
            f"CreateCMABAssignmentRequest."
        )

    for context_input, context_def in zip(
        context_inputs,
        context_defns,
        strict=True,
    ):
        if context_input.context_id != context_def.id:
            raise LateValidationError(
                f"Context input for id {context_input.context_id} does not match expected context id {context_def.id}",
            )
        if context_def.value_type == ContextType.BINARY.value and context_input.context_value not in {0.0, 1.0}:
            raise LateValidationError(
                f"Context value for id {context_input.context_id} must be binary (0 or 1).",
            )
    return context_inputs


@router.get("/caller-identity")
async def caller_identity(user: Annotated[tables.User, Depends(require_user_from_token)]) -> CallerIdentity:
    """Returns basic metadata about the authenticated caller of this method."""
    return CallerIdentity(
        email=user.email,
        iss=user.iss or "",
        sub=user.sub or "",
        hd="",
        is_privileged=user.is_privileged,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
):
    """Invalidates all previously created session tokens."""
    user.last_logout = datetime.now(UTC)
    await session.commit()
    return GENERIC_SUCCESS


@router.get(
    "/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots/{snapshot_id}"
)
async def get_snapshot(
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    organization_id: Annotated[str, Path()],
    datasource_id: Annotated[str, Path()],
    experiment_id: Annotated[str, Path()],
    snapshot_id: Annotated[str, Path()],
) -> GetSnapshotResponse:
    """Fetches a snapshot by ID."""
    datasource = await get_datasource_or_raise(session, user, datasource_id, organization_id=organization_id)
    experiment = await get_experiment_via_ds_or_raise(session, datasource, experiment_id)

    snapshot = await session.scalar(
        select(tables.Snapshot).where(
            tables.Snapshot.experiment_id == experiment.id,
            tables.Snapshot.id == snapshot_id,
        )
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    return GetSnapshotResponse(snapshot=convert_snapshot_to_api_snapshot(snapshot))


@router.get("/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots")
async def list_snapshots(
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    organization_id: Annotated[str, Path()],
    datasource_id: Annotated[str, Path()],
    experiment_id: Annotated[str, Path()],
    status_: Annotated[
        list[SnapshotStatus] | None,
        Query(
            alias="status",
            description="Filter the returned snapshots to only those of this status. May be specified multiple times.",
        ),
    ] = None,
) -> ListSnapshotsResponse:
    """Lists snapshots for an experiment, ordered by timestamp."""
    datasource = await get_datasource_or_raise(session, user, datasource_id, organization_id=organization_id)
    experiment = await get_experiment_via_ds_or_raise(session, datasource, experiment_id)

    query = (
        select(tables.Snapshot)
        .where(tables.Snapshot.experiment_id == experiment.id)
        .order_by(tables.Snapshot.updated_at)
    )
    if status_:
        query = query.where(
            tables.Snapshot.status.in_([convert_api_snapshot_status_to_snapshot_status(s) for s in status_])
        )
    snapshots = await session.scalars(query)

    return ListSnapshotsResponse(items=[convert_snapshot_to_api_snapshot(snapshot) for snapshot in snapshots])


@router.delete(
    "/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots/{snapshot_id}"
)
async def delete_snapshot(
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    _organization_id: Annotated[str, Path(alias="organization_id")],
    datasource_id: Annotated[str, Path()],
    experiment_id: Annotated[str, Path()],
    snapshot_id: Annotated[str, Path()],
    allow_missing: Annotated[
        bool,
        Query(description="If true, return a 204 even if the resource does not exist."),
    ] = False,
):
    """Deletes a snapshot."""
    resource_query = select(tables.Snapshot).where(
        tables.Snapshot.experiment_id == experiment_id, tables.Snapshot.id == snapshot_id
    )
    return await handle_delete(
        session, allow_missing, authz.is_user_authorized_on_datasource(user, datasource_id), resource_query
    )


@router.post("/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots")
async def create_snapshot(
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    organization_id: Annotated[str, Path()],
    datasource_id: Annotated[str, Path()],
    experiment_id: Annotated[str, Path()],
    background_tasks: BackgroundTasks,
) -> CreateSnapshotResponse:
    """Request the asynchronous creation of a snapshot for an experiment.

    Returns the ID of the snapshot. Poll get_snapshot until the job is completed.
    """
    datasource = await get_datasource_or_raise(session, user, datasource_id, organization_id=organization_id)
    experiment = await get_experiment_via_ds_or_raise(session, datasource, experiment_id)

    if experiment.state != ExperimentState.COMMITTED:
        raise LateValidationError("You can only snapshot committed experiments.")
    # Aligning with the buffer in snapshotter.py, as we wish to capture +/- 1 day on both sides.
    if experiment.end_date < datetime.now(UTC) - timedelta(days=1):
        raise LateValidationError("You can only snapshot active experiments.")

    snapshot = tables.Snapshot(experiment_id=experiment.id)
    session.add(snapshot)
    await session.commit()
    background_tasks.add_task(snapshotter.make_first_snapshot, snapshot.experiment_id, snapshot.id)
    return CreateSnapshotResponse(id=snapshot.id)


@router.get("/organizations")
async def list_organizations(
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
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
    user: Annotated[tables.User, Depends(require_user_from_token)],
    body: Annotated[CreateOrganizationRequest, Body(...)],
) -> CreateOrganizationResponse:
    """Creates a new organization.

    Only privileged users can create organizations.
    """
    if not user.is_privileged:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only privileged users can create organizations",
        )

    organization = await create_organization_impl(session, user, body.name)
    await session.commit()

    return CreateOrganizationResponse(id=organization.id)


@router.post("/organizations/{organization_id}/webhooks")
async def add_webhook_to_organization(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    body: Annotated[AddWebhookToOrganizationRequest, Body(...)],
) -> AddWebhookToOrganizationResponse:
    """Adds a Webhook to an organization."""
    # Verify user has access to the organization
    org = await get_organization_or_raise(session, user, organization_id)

    # Generate a secure auth token
    auth_token = secrets.token_hex(16)

    # Create and save the webhook
    webhook = tables.Webhook(
        type=body.type,
        name=body.name,
        url=body.url,
        auth_token=auth_token,
        organization_id=org.id,
    )
    session.add(webhook)
    await session.commit()

    return AddWebhookToOrganizationResponse(
        id=webhook.id,
        type=webhook.type,
        name=webhook.name,
        url=webhook.url,
        auth_token=auth_token,
    )


@router.get("/organizations/{organization_id}/webhooks")
async def list_organization_webhooks(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
) -> ListWebhooksResponse:
    """Lists all the webhooks for an organization."""
    # Verify user has access to the organization
    org = await get_organization_or_raise(session, user, organization_id)

    # Query for webhooks
    stmt = (
        select(tables.Webhook)
        .where(tables.Webhook.organization_id == org.id)
        .order_by(tables.Webhook.name, tables.Webhook.id)
    )
    webhooks = await session.scalars(stmt)

    # Convert webhooks to WebhookSummary objects
    webhook_summaries = convert_webhooks_to_webhooksummaries(webhooks)

    return ListWebhooksResponse(items=webhook_summaries)


def convert_webhooks_to_webhooksummaries(webhooks):
    return [
        WebhookSummary(
            id=webhook.id,
            type=webhook.type,
            name=webhook.name,
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
    user: Annotated[tables.User, Depends(require_user_from_token)],
    body: Annotated[UpdateOrganizationWebhookRequest, Body(...)],
):
    """Updates a webhook's name and URL in an organization."""
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found")

    webhook.name = body.name
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
    user: Annotated[tables.User, Depends(require_user_from_token)],
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found")

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
    user: Annotated[tables.User, Depends(require_user_from_token)],
    allow_missing: Annotated[
        bool,
        Query(description="If true, return a 204 even if the resource does not exist."),
    ] = False,
):
    """Removes a Webhook from an organization."""
    resource_query = select(tables.Webhook).where(tables.Webhook.id == webhook_id)
    return await handle_delete(
        session,
        allow_missing,
        authz.is_user_authorized_on_organization(user, organization_id),
        resource_query,
    )


@router.get("/organizations/{organization_id}/events")
async def list_organization_events(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
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


@router.post("/organizations/{organization_id}/members", status_code=status.HTTP_204_NO_CONTENT)
async def add_member_to_organization(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    if not user.is_privileged:
        # Verify user is a member of the organization
        _authz_check = await get_organization_or_raise(session, user, organization_id)

    if body.email in {u.email for u in org.users}:
        return GENERIC_SUCCESS

    new_user = (await session.execute(select(tables.User).where(tables.User.email == body.email))).scalar_one_or_none()
    if new_user is None:
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
    user: Annotated[tables.User, Depends(require_user_from_token)],
    allow_missing: Annotated[
        bool,
        Query(description="If true, return a 204 even if the resource does not exist."),
    ] = False,
):
    """Removes a member from an organization.

    The authenticated user must be part of the organization to remove members.
    """
    resource_query = select(tables.UserOrganization).where(
        tables.UserOrganization.organization_id == organization_id,
        tables.UserOrganization.user_id == user_id,
        tables.UserOrganization.user_id != user.id,  # not current authenticated user
    )
    return await handle_delete(
        session,
        allow_missing,
        authz.is_user_authorized_on_organization(user, organization_id),
        resource_query,
    )


@router.patch("/organizations/{organization_id}")
async def update_organization(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
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
    user: Annotated[tables.User, Depends(require_user_from_token)],
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

    datasources_stmt = select(tables.Datasource).filter(tables.Datasource.organization_id == organization_id)
    datasources = await session.scalars(datasources_stmt)

    return GetOrganizationResponse(
        id=org.id,
        name=org.name,
        users=[UserSummary(id=u.id, email=u.email) for u in sorted(users, key=lambda x: x.email)],
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
    user: Annotated[tables.User, Depends(require_user_from_token)],
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
        items=[convert_ds_to_summary(ds) for ds in sorted(datasources, key=lambda d: d.name)]
    )


@router.post("/datasources")
async def create_datasource(
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    body: Annotated[CreateDatasourceRequest, Body(...)],
) -> CreateDatasourceResponse:
    """Creates a new datasource for the specified organization."""
    org = await get_organization_or_raise(session, user, body.organization_id)

    raise_unless_safe_hostname(body.dsn)

    config = RemoteDatabaseConfig(participants=[], type="remote", dwh=api_dsn_to_settings_dwh(body.dsn))

    datasource = await admin_common.create_datasource_impl(session, org, body.name, config)
    await session.commit()

    return CreateDatasourceResponse(id=datasource.id)


@router.patch("/datasources/{datasource_id}", status_code=status.HTTP_204_NO_CONTENT)
async def update_datasource(
    datasource_id: str,
    body: UpdateDatasourceRequest,
    user: Annotated[tables.User, Depends(require_user_from_token)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
):
    ds = await get_datasource_or_raise(session, user, datasource_id)
    if body.name is not None:
        ds.name = body.name
    if body.dsn is not None:
        raise_unless_safe_hostname(body.dsn)
        cfg = ds.get_config()
        cfg.dwh = api_dsn_to_settings_dwh(body.dsn, cfg.dwh)
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
    user: Annotated[tables.User, Depends(require_user_from_token)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> GetDatasourceResponse:
    """Returns detailed information about a specific datasource."""
    ds = await get_datasource_or_raise(session, user, datasource_id, preload=[tables.Datasource.organization])
    config = ds.get_config()
    return GetDatasourceResponse(
        id=ds.id,
        name=ds.name,
        dsn=admin_api_converters.settings_dwh_to_api_dsn(config.dwh),
        organization_id=ds.organization_id,
        organization_name=ds.organization.name,
    )


@router.get("/datasources/{datasource_id}/inspect")
async def inspect_datasource(
    datasource_id: str,
    user: Annotated[tables.User, Depends(require_user_from_token)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
) -> InspectDatasourceResponse:
    """Verifies connectivity to a datasource and returns a list of readable tables."""
    ds = await get_datasource_or_raise(session, user, datasource_id)

    if ds.get_config().dwh.driver == "none":
        return InspectDatasourceResponse(tables=[])

    if not refresh and cache_is_fresh(ds.table_list_updated) and ds.table_list is not None:
        return InspectDatasourceResponse(tables=ds.table_list)

    async with clear_db_table_cache_on_error(session, ds):
        config = ds.get_config()
        async with DwhSession(config.dwh) as dwh:
            tablenames = await dwh.list_tables()
        ds.set_table_list(tablenames)
        await session.commit()
        return InspectDatasourceResponse(tables=tablenames)


async def invalidate_inspect_table_cache(session, datasource_id):
    """Invalidates all table inspection cache entries for a datasource."""
    await session.execute(
        delete(tables.DatasourceTablesInspected).where(tables.DatasourceTablesInspected.datasource_id == datasource_id)
    )


@router.get("/datasources/{datasource_id}/inspect/{table_name}")
async def inspect_table_in_datasource(
    datasource_id: str,
    table_name: str,
    user: Annotated[tables.User, Depends(require_user_from_token)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
) -> InspectDatasourceTableResponse:
    """Inspects a single table in a datasource and returns a summary of its fields."""
    ds = await get_datasource_or_raise(session, user, datasource_id)
    if (
        not refresh
        and (cached := await session.get(tables.DatasourceTablesInspected, (datasource_id, table_name)))
        and cache_is_fresh(cached.response_last_updated)
        and cached.response is not None
    ):
        return InspectDatasourceTableResponse.model_validate(cached.response)

    config = ds.get_config()

    await invalidate_inspect_table_cache(session, datasource_id)
    await session.commit()

    async with DwhSession(config.dwh) as dwh:
        # CannotFindTableError will be handled by exceptionhandlers.py.
        table = await dwh.inspect_table(table_name)
    response = create_inspect_table_response_from_table(table)

    session.add(
        tables.DatasourceTablesInspected(
            datasource_id=datasource_id,
            table_name=table_name,
            response=response.model_dump(),
            response_last_updated=datetime.now(UTC),
        )
    )
    await session.commit()

    return response


@router.delete(
    "/organizations/{organization_id}/datasources/{datasource_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_datasource(
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    organization_id: Annotated[str, Path(...)],
    datasource_id: Annotated[str, Path(...)],
    allow_missing: Annotated[
        bool,
        Query(description="If true, return a 204 even if the resource does not exist."),
    ] = False,
):
    """Deletes a datasource.

    The user must be a member of the organization that owns the datasource.
    """
    resource_query = select(tables.Datasource).where(tables.Datasource.id == datasource_id)
    return await handle_delete(
        session,
        allow_missing,
        authz.is_user_authorized_on_organization(user, organization_id),
        resource_query,
    )


@router.get("/datasources/{datasource_id}/participants")
async def list_participant_types(
    datasource_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
) -> ListParticipantsTypeResponse:
    ds = await get_datasource_or_raise(session, user, datasource_id)
    return ListParticipantsTypeResponse(
        items=list(sorted(ds.get_config().participants, key=lambda p: p.participant_type))
    )


@router.post("/datasources/{datasource_id}/participants")
async def create_participant_type(
    datasource_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
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
    user: Annotated[tables.User, Depends(require_user_from_token)],
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
) -> InspectParticipantTypesResponse:
    """Returns filter, strata, and metric field metadata for a participant type, including exemplars for
    filter fields."""
    ds = await get_datasource_or_raise(session, user, datasource_id)
    dsconfig = ds.get_config()
    # CannotFindParticipantsError will be handled by exceptionhandlers.
    pconfig = dsconfig.find_participants(participant_id)
    if pconfig.type == "sheet":
        raise HTTPException(status_code=405, detail="Sheet schemas cannot be inspected.")

    if (
        not refresh
        and (cached := await session.get(tables.ParticipantTypesInspected, (datasource_id, participant_id)))
        and cache_is_fresh(cached.response_last_updated)
        and cached.response is not None
    ):
        return InspectParticipantTypesResponse.model_validate(cached.response)

    await session.execute(
        delete(tables.ParticipantTypesInspected).where(
            tables.ParticipantTypesInspected.datasource_id == datasource_id,
            tables.ParticipantTypesInspected.participant_type == participant_id,
        )
    )
    await session.commit()

    filter_fields = {c.field_name: c for c in pconfig.fields if c.is_filter}
    strata_fields = {c.field_name: c for c in pconfig.fields if c.is_strata}
    metric_cols = {c.field_name: c for c in pconfig.fields if c.is_metric}

    async def inspect_participant_types_impl() -> InspectParticipantTypesResponse:
        async with DwhSession(dsconfig.dwh) as dwh:
            result = await dwh.inspect_table_with_descriptors(pconfig.table_name, pconfig.get_unique_id_field())
            filter_data = await asyncio.to_thread(
                get_stats_on_filters,
                dwh.session,
                result.sa_table,
                result.db_schema,
                filter_fields,
            )

        return InspectParticipantTypesResponse(
            metrics=sorted(
                [
                    GetMetricsResponseElement(
                        data_type=result.db_schema.get(col_name).data_type,  # type: ignore[union-attr]
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
                        data_type=result.db_schema.get(field_name).data_type,  # type: ignore[union-attr]
                        field_name=field_name,
                        description=field_descriptor.description,
                    )
                    for field_name, field_descriptor in strata_fields.items()
                    if result.db_schema.get(field_name)
                ],
                key=lambda item: item.field_name,
            ),
            filters=sorted(
                filter_data,
                key=lambda item: item.field_name,
            ),
        )

    response = await inspect_participant_types_impl()

    session.add(
        tables.ParticipantTypesInspected(
            datasource_id=datasource_id,
            participant_type=participant_id,
            # This value may contain Python datetime objects. The default JSON serializer doesn't serialize them
            # but the Pydantic serializer turns them into ISO8601 strings. This could be better.
            response=json.loads(response.model_dump_json()),
            response_last_updated=datetime.now(UTC),
        )
    )

    await session.commit()

    return response


@router.get("/datasources/{datasource_id}/participants/{participant_id}")
async def get_participant_types(
    datasource_id: str,
    participant_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
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
    user: Annotated[tables.User, Depends(require_user_from_token)],
    body: UpdateParticipantsTypeRequest,
):
    ds = await get_datasource_or_raise(session, user, datasource_id)
    config = ds.get_config()
    participant = config.find_participants(participant_id)
    config.participants.remove(participant)
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
    user: Annotated[tables.User, Depends(require_user_from_token)],
    allow_missing: Annotated[
        bool,
        Query(description="If true, return a 204 even if the resource does not exist."),
    ] = False,
):
    async def get_participants_or_none(session_: AsyncSession):
        resource = (
            await session_.execute(select(tables.Datasource).where(tables.Datasource.id == datasource_id))
        ).scalar_one_or_none()
        if resource is None:
            return None
        config = resource.get_config()
        participant = config.find_participants_or_none(participant_id)
        if participant is None:
            return None
        return resource

    async def deleter(_session: AsyncSession, resource: tables.Datasource):
        config = resource.get_config()
        participant = config.find_participants(participant_id)
        config.participants.remove(participant)
        resource.set_config(config)

    return await handle_delete(
        session,
        allow_missing,
        authz.is_user_authorized_on_datasource(user, datasource_id),
        get_participants_or_none,
        deleter,
    )


@router.get("/datasources/{datasource_id}/apikeys")
async def list_api_keys(
    datasource_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
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
    user: Annotated[tables.User, Depends(require_user_from_token)],
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
    user: Annotated[tables.User, Depends(require_user_from_token)],
    api_key_id: Annotated[str, Path(...)],
    allow_missing: Annotated[
        bool,
        Query(description="If true, return a 204 even if the resource does not exist."),
    ] = False,
):
    """Deletes the specified API key."""
    resource_query = (
        select(tables.ApiKey)
        .join(tables.Datasource)
        .where(tables.Datasource.id == datasource_id, tables.ApiKey.id == api_key_id)
    )
    return await handle_delete(
        session,
        allow_missing,
        authz.is_user_authorized_on_datasource(user, datasource_id),
        resource_query,
    )


@router.post("/datasources/{datasource_id}/experiments")
async def create_experiment(
    datasource_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    body: CreateExperimentRequest,
    chosen_n: Annotated[int | None, Query(..., description="Number of participants to assign.", ge=0)] = None,
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
    """Creates a new experiment in the specified datasource."""
    datasource = await get_datasource_or_raise(session, user, datasource_id)

    if body.design_spec.ids_are_present():
        raise LateValidationError("Invalid DesignSpec: UUIDs must not be set.")

    # Validate webhook IDs exist and belong to organization
    organization_id = datasource.organization_id
    validated_webhooks = await validate_webhooks(
        session=session, organization_id=organization_id, request_webhooks=body.webhooks
    )

    return await experiments_common.create_experiment_impl(
        request=body,
        datasource=datasource,
        xngin_session=session,
        chosen_n=chosen_n,
        stratify_on_metrics=stratify_on_metrics,
        random_state=random_state,
        validated_webhooks=validated_webhooks,
    )


@router.get(
    "/datasources/{datasource_id}/experiments/{experiment_id}/analyze",
    description="""
    For preassigned experiments, and online experiments (except contextual bandits),
    returns an analysis of the experiment's performance, given datasource and experiment ID.""",
)
async def analyze_experiment(
    datasource_id: str,
    experiment_id: str,
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    baseline_arm_id: Annotated[
        str | None,
        Query(
            description="UUID of the baseline arm. If None, the first design spec arm is used.",
        ),
    ] = None,
) -> ExperimentAnalysisResponse:
    ds = await get_datasource_or_raise(xngin_session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(
        xngin_session,
        ds,
        experiment_id,
        preload=[tables.Experiment.arm_assignments, tables.Experiment.draws, tables.Experiment.contexts],
    )

    design_spec = await ExperimentStorageConverter(experiment).get_design_spec()
    match design_spec:
        case BaseBanditExperimentSpec():
            if experiment.experiment_type != ExperimentsType.MAB_ONLINE.value:
                raise LateValidationError(
                    """Invalid experiment type for bandit analysis; for CMAB experiments,
                    use the corresponding POST endpoint.""",
                )
            return experiments_common.analyze_experiment_bandit_impl(experiment)

        case BaseFrequentistDesignSpec():
            # Always assume the first arm is the baseline; UI can override this.
            baseline_arm_id = baseline_arm_id or design_spec.arms[0].arm_id
            assert baseline_arm_id is not None
            return await experiments_common.analyze_experiment_freq_impl(
                ds.get_config(), experiment, baseline_arm_id, design_spec.metrics
            )
        case _:
            assert_never()


@router.post(
    "/datasources/{datasource_id}/experiments/{experiment_id}/analyze_cmab",
    description="""
    For contextual bandit experiments, returns an analysis of the experiment's performance,
    given datasource and experiment ID and context values as input.""",
)
async def analyze_cmab_experiment(
    datasource_id: str,
    experiment_id: str,
    body: CMABContextInputRequest,
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
) -> ExperimentAnalysisResponse:
    ds = await get_datasource_or_raise(xngin_session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(
        xngin_session,
        ds,
        experiment_id,
        preload=[tables.Experiment.draws, tables.Experiment.contexts],
    )

    if experiment.experiment_type != ExperimentsType.CMAB_ONLINE.value:
        raise LateValidationError(
            f"Experiment {experiment.id} is a {experiment.experiment_type} experiment, and not a "
            f"{ExperimentsType.CMAB_ONLINE.value} experiment. Please use the corresponding GET endpoint to "
            f"retrieve an experiment analysis."
        )

    context_inputs = body.context_inputs
    context_defns = experiment.contexts
    sorted_context_inputs = sort_contexts_by_id_or_raise(context_defns, context_inputs)

    return experiments_common.analyze_experiment_bandit_impl(
        experiment, context_vals=[ci.context_value for ci in sorted_context_inputs]
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
    status_code=status.HTTP_204_NO_CONTENT,
)
async def commit_experiment(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
):
    ds = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(session, ds, experiment_id)
    return await experiments_common.commit_experiment_impl(session, experiment)


@router.post(
    "/datasources/{datasource_id}/experiments/{experiment_id}/abandon",
    responses=EXPERIMENT_STATE_TRANSITION_RESPONSES,
    status_code=status.HTTP_204_NO_CONTENT,
)
async def abandon_experiment(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
):
    ds = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(session, ds, experiment_id)
    return await experiments_common.abandon_experiment_impl(session, experiment)


@router.get("/organizations/{organization_id}/experiments")
async def list_organization_experiments(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
) -> ListExperimentsResponse:
    """Returns a list of experiments in the organization."""
    org = await get_organization_or_raise(session, user, organization_id)
    return await experiments_common.list_organization_or_datasource_experiments_impl(
        xngin_session=session, organization_id=org.id
    )


@router.get("/datasources/{datasource_id}/experiments/{experiment_id}")
async def get_experiment_for_ui(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
) -> GetExperimentResponse:
    """Returns the experiment with the specified ID."""
    ds = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(
        session,
        ds,
        experiment_id,
        preload=[tables.Experiment.webhooks, tables.Experiment.contexts],
    )
    return await experiments_common.get_experiment_impl(session, experiment)


@router.get("/datasources/{datasource_id}/experiments/{experiment_id}/assignments")
async def get_experiment_assignments_for_ui(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
) -> GetExperimentAssignmentsResponse:
    ds = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(
        session,
        ds,
        experiment_id,
        preload=[tables.Experiment.arm_assignments, tables.Experiment.draws],
    )
    return experiments_common.get_experiment_assignments_impl(experiment)


@router.get(
    "/datasources/{datasource_id}/experiments/{experiment_id}/assignments/csv",
    summary=(
        "Export experiment assignments as CSV file; BalanceCheck not included. "
        "csv header form: participant_id,arm_id,arm_name,strata_name1,strata_name2,..."
    ),
)
async def get_experiment_assignments_as_csv_for_ui(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
) -> StreamingResponse:
    # TODO: update for bandits
    ds = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(
        session,
        ds,
        experiment_id,
        preload=[tables.Experiment.arm_assignments],
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
    user: Annotated[tables.User, Depends(require_user_from_token)],
    create_if_none: Annotated[
        bool,
        Query(
            description=(
                "Create an assignment if none exists. Does nothing for preassigned experiments. "
                "Override if you just want to check if an assignment exists."
            )
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

    return await experiments_common.get_or_create_assignment_for_participant(
        xngin_session=session,
        experiment=experiment,
        participant_id=participant_id,
        create_if_none=create_if_none,
        properties=None,
        random_state=random_state,
    )


@router.patch("/datasources/{datasource_id}/experiments/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def update_experiment(
    datasource_id: str,
    experiment_id: str,
    body: UpdateExperimentRequest,
    user: Annotated[tables.User, Depends(require_user_from_token)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
):
    datasource = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(session, datasource, experiment_id)

    if experiment.state != ExperimentState.COMMITTED:
        raise LateValidationError("Experiment must have been committed to be updated.")

    if body.name is not None:
        experiment.name = body.name
    if body.description is not None:
        experiment.description = body.description
    if body.design_url is not None:
        experiment.design_url = body.design_url
    if body.start_date is not None:
        end_date = body.end_date or experiment.end_date
        if end_date <= body.start_date:
            raise LateValidationError("New start date must be before end date.")
        experiment.start_date = body.start_date
    if body.end_date is not None:
        if body.end_date <= experiment.start_date:
            raise LateValidationError("New end date must be after start date.")
        experiment.end_date = body.end_date
    if body.decision is not None:
        experiment.decision = body.decision
    if body.impact is not None:
        experiment.impact = body.impact

    await session.commit()
    return GENERIC_SUCCESS


@router.delete(
    "/datasources/{datasource_id}/experiments/{experiment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_experiment(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    allow_missing: Annotated[
        bool,
        Query(description="If true, return a 204 even if the resource does not exist."),
    ] = False,
):
    """Deletes the experiment with the specified ID."""
    resource_query = select(tables.Experiment).where(tables.Experiment.id == experiment_id)
    return await handle_delete(
        session,
        allow_missing,
        authz.is_user_authorized_on_datasource(user, datasource_id),
        resource_query,
    )


@router.patch(
    "/datasources/{datasource_id}/experiments/{experiment_id}/arms/{arm_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def update_arm(
    datasource_id: str,
    experiment_id: str,
    arm_id: str,
    body: UpdateArmRequest,
    user: Annotated[tables.User, Depends(require_user_from_token)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
):
    datasource = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(session, datasource, experiment_id)

    if experiment.state != ExperimentState.COMMITTED:
        raise LateValidationError("Experiment must have been committed to update arms.")

    arm = next((arm for arm in experiment.arms if arm.id == arm_id), None)
    if arm is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Arm not found.")

    if body.name is not None:
        arm.name = body.name
    if body.description is not None:
        arm.description = body.description

    await session.commit()
    return GENERIC_SUCCESS


@router.post("/datasources/{datasource_id}/power")
async def power_check(
    datasource_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    body: PowerRequest,
) -> PowerResponse:
    """Performs a power check for the specified datasource."""
    design_spec = body.design_spec
    if not isinstance(design_spec, BaseFrequentistDesignSpec):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Power checks are only supported for frequentist experiments",
        )
    ds = await get_datasource_or_raise(session, user, datasource_id)
    if isinstance(ds.config, NoDwh):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Power checks are not supported for datasources without a data warehouse.",
        )
    dsconfig = ds.get_config()
    participants_cfg = dsconfig.find_participants(design_spec.participant_type)

    validate_schema_metrics_or_raise(design_spec, participants_cfg)
    async with DwhSession(dsconfig.dwh) as dwh:
        sa_table = await dwh.inspect_table(participants_cfg.table_name)

        metric_stats = await asyncio.to_thread(
            get_stats_on_metrics,
            dwh.session,
            sa_table,
            design_spec.metrics,
            design_spec.filters,
        )

    arm_weights = design_spec.get_validated_arm_weights()

    return PowerResponse(
        analyses=check_power(
            metrics=metric_stats,
            n_arms=len(design_spec.arms),
            power=design_spec.power,
            alpha=design_spec.alpha,
            arm_weights=arm_weights,
        )
    )


def validate_schema_metrics_or_raise(design_spec: BaseFrequentistDesignSpec, schema: ParticipantsSchema):
    metric_fields = {m.field_name for m in schema.fields if m.is_metric}
    metrics_requested = {m.field_name for m in design_spec.metrics}
    invalid_metrics = metrics_requested - metric_fields
    if len(invalid_metrics) > 0:
        raise LateValidationError(
            f"Invalid DesignSpec metrics (check your Datasource configuration): {invalid_metrics}"
        )


def raise_unless_safe_hostname(dsn):
    """Raises a 400 if the DNS name in dsn is possibly attempting to connect to resources on local network."""
    if flags.DISABLE_SAFEDNS_CHECK:
        return
    if isinstance(dsn, PostgresDsn | RedshiftDsn):
        try:
            safe_resolve(dsn.host)
        except DnsLookupError as err:
            raise HTTPException(
                status_code=400,
                detail="DNS resolution failed. Check datasource hostname and try again.",
            ) from err
