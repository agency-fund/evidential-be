"""
This module defines the internal Evidential UI-facing Admin API endpoints.
(See experiments_api.py for integrator-facing endpoints.)
"""

import secrets
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal, assert_never

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
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import delete, func, literal, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, selectinload

from xngin.apiserver import constants
from xngin.apiserver.apikeys import hash_key_or_raise, make_key
from xngin.apiserver.dependencies import xngin_db_session, xngin_sync_db_session
from xngin.apiserver.dns.safe_resolve import DnsLookupError, safe_resolve
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.dwh.inspections import create_inspect_table_response_from_table
from xngin.apiserver.dwh.queries import get_stats_on_metrics
from xngin.apiserver.exceptionhandlers import XHTTPValidationError
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.limits import MAX_LENGTH_OF_EMAIL_VALUE, MAX_LENGTH_OF_NAME_VALUE
from xngin.apiserver.pagination import (
    PaginationQuery,
    SortField,
    build_next_page_token,
    paginate,
    pagination_query_params,
    unbounded_pagination_query_params,
)
from xngin.apiserver.routers.admin import admin_api_converters, admin_common, authz
from xngin.apiserver.routers.admin import admin_dependencies as adeps
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
    CreateSnapshotResponse,
    CreateUserRequest,
    CreateUserResponse,
    DatasourceSummary,
    DeleteExperimentDataRequest,
    EventSummary,
    GetDatasourceResponse,
    GetExperimentForUiResponse,
    GetOrganizationResponse,
    GetSnapshotResponse,
    GetUserResponse,
    InspectDatasourceResponse,
    InspectDatasourceTableResponse,
    ListApiKeysResponse,
    ListDatasourcesResponse,
    ListOrganizationEventsResponse,
    ListOrganizationsResponse,
    ListSnapshotsResponse,
    ListUsersResponse,
    ListWebhooksResponse,
    OrganizationListItem,
    OrganizationSummary,
    PatchUserRequest,
    PostgresDsn,
    RedshiftDsn,
    SnapshotStatus,
    UpdateArmRequest,
    UpdateDatasourceRequest,
    UpdateExperimentRequest,
    UpdateOrganizationRequest,
    UpdateOrganizationWebhookRequest,
    UserDetail,
    UserSummary,
    WebhookSummary,
)
from xngin.apiserver.routers.admin.admin_common import create_organization_impl
from xngin.apiserver.routers.admin.generic_handlers import handle_delete
from xngin.apiserver.routers.auth.auth_api_types import CallerIdentity
from xngin.apiserver.routers.auth.auth_dependencies import require_user_from_token
from xngin.apiserver.routers.common_api_types import (
    CMABContextInputRequest,
    CMABExperimentSpec,
    ContextInput,
    ContextType,
    CreateExperimentRequest,
    CreateExperimentResponse,
    ExperimentAnalysisResponse,
    ExperimentsType,
    Filter,
    ListExperimentsResponse,
    MABDwhExperimentSpec,
    MABExperimentSpec,
    OnlineFrequentistExperimentSpec,
    PowerRequest,
    PowerResponse,
    PreassignedFrequentistExperimentSpec,
    Relation,
)
from xngin.apiserver.routers.common_enums import ExperimentState
from xngin.apiserver.routers.experiments import experiments_common, experiments_common_csv
from xngin.apiserver.routers.experiments.experiments_common import (
    AbandonExperimentResult,
    convert_table_to_fields_or_raise,
    make_schema_from_experiment,
)
from xngin.apiserver.routers.experiments.experiments_common_csv import CsvStreamingResponse
from xngin.apiserver.routers.power_adapters import calculate_cluster_stats_from_database
from xngin.apiserver.settings import NoDwh, RemoteDatabaseConfig
from xngin.apiserver.snapshots import snapshotter
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter
from xngin.events.webhook_sent import WebhookSentEvent
from xngin.stats import check_power
from xngin.tq.task_payload_types import WEBHOOK_OUTBOUND_TASK_TYPE

GENERIC_SUCCESS = Response(status_code=status.HTTP_204_NO_CONTENT)
RESPONSE_CACHE_MAX_AGE_SECONDS = timedelta(minutes=15).seconds


# Describes the structure of an error raised via `raise HTTPException`.
class HTTPExceptionError(BaseModel):
    detail: str


class MessageError(BaseModel):
    message: str


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

VALIDATION_RESPONSE: dict[str, Any] = {
    "model": XHTTPValidationError,
    "description": "Validation error.",
}

DWH_CONNECTION_RESPONSES: dict[str | int, dict[str, Any]] = {
    "422": VALIDATION_RESPONSE,
    "502": {
        "model": MessageError,
        "description": "Unable to connect to the datasource.",
    },
    "504": {
        "model": MessageError,
        "description": "Datasource request timed out.",
    },
}

DWH_CONNECTION_AND_NOT_FOUND_RESPONSES: dict[str | int, dict[str, Any]] = {
    **DWH_CONNECTION_RESPONSES,
    "404": {
        "model": MessageError,
        "description": "Requested datasource metadata could not be found.",
    },
}


def cache_is_fresh(updated: datetime | None):
    return updated is not None and datetime.now(UTC) - updated < timedelta(minutes=5)


def ensure_not_last_privileged_user(session: Session, exclude_user_id: str) -> None:
    """Raises 403 if no privileged users remain after excluding the given user_id.

    Use this guard before revoking privilege from, or deleting, a user that is privileged.
    """
    remaining = session.scalar(
        select(func.count())
        .select_from(tables.User)
        .where(tables.User.is_privileged.is_(True), tables.User.id != exclude_user_id)
    )
    if not remaining:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="At least one privileged user must remain.",
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting router: {__name__} (prefix={router.prefix})")
    yield


@contextmanager
def clear_db_table_cache_on_error(session: Session, datasource: tables.Datasource):
    """Context manager that clears a datasource's cached table list on error."""
    try:
        yield
    except:
        datasource.clear_table_list()
        session.commit()
        raise


router = APIRouter(
    lifespan=lifespan,
    prefix=constants.API_PREFIX_V1 + "/m",
    responses=STANDARD_ADMIN_RESPONSES,
    dependencies=[Depends(require_user_from_token)],  # All routes in this router require authentication.
)


def validate_webhooks(session: Session, organization_id: str, request_webhooks: list[str]) -> list[tables.Webhook]:
    # Validate webhook IDs exist and belong to organization
    validated_webhooks = []
    if request_webhooks:
        webhooks = session.scalars(
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
def caller_identity(user: Annotated[tables.User, Depends(require_user_from_token)]) -> CallerIdentity:
    """Returns basic metadata about the authenticated caller of this method."""
    return CallerIdentity(
        email=user.email,
        iss=user.iss or "",
        sub=user.sub or "",
        hd="",
        is_privileged=user.is_privileged,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
):
    """Invalidates all previously created session tokens."""
    user.last_logout = datetime.now(UTC)
    session.commit()
    return GENERIC_SUCCESS


@router.post("/users")
def create_user(
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    _user: Annotated[tables.User, Depends(adeps.privileged_caller)],
    body: Annotated[CreateUserRequest, Body(...)],
) -> CreateUserResponse:
    """Creates a User record by email. Privileged users only.

    Idempotent: if a user with the given email already exists, the existing user's id is returned
    and nothing else is changed. Newly-created users have `is_privileged=false` and no organization
    memberships. When they next sign in via OIDC, the existing user record is bound to their OIDC
    identity automatically.
    """
    user_id = (
        session.execute(
            pg_insert(tables.User)
            .values(email=body.email)
            .on_conflict_do_update(
                index_elements=[tables.User.email],
                set_={"email": body.email},
            )
            .returning(tables.User.id)
        )
    ).scalar_one()
    session.commit()
    return CreateUserResponse(id=user_id)


@router.get("/users")
def list_users(
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    user: Annotated[tables.User, Depends(adeps.privileged_caller)],
    pagination: Annotated[PaginationQuery, Depends(pagination_query_params)],
    email_contains: Annotated[
        str | None,
        Query(
            max_length=MAX_LENGTH_OF_EMAIL_VALUE,
            description="Optional case-insensitive substring filter on the user's email address.",
        ),
    ] = None,
    scope: Annotated[
        Literal["all", "mine"],
        Query(
            description=(
                "`all` (default) returns every user in the system. `mine` returns only users that share "
                "at least one organization with the caller."
            )
        ),
    ] = "all",
) -> ListUsersResponse:
    """Lists users in the system. Privileged users only.

    Sorted by email ascending.
    """
    stmt = select(tables.User).options(selectinload(tables.User.organizations))
    if scope == "mine":
        callers_org_ids = (
            select(tables.UserOrganization.organization_id).where(tables.UserOrganization.user_id == user.id).subquery()
        )
        stmt = stmt.where(
            tables.User.id.in_(
                select(tables.UserOrganization.user_id).where(
                    tables.UserOrganization.organization_id.in_(select(callers_org_ids))
                )
            )
        )
    if email_contains:
        stmt = stmt.where(tables.User.email.icontains(email_contains, autoescape=True))

    ordering = [
        SortField(column=tables.User.email, attr="email", direction="asc"),
        SortField(column=tables.User.id, attr="id", direction="asc"),
    ]
    stmt = paginate(stmt, ordering, pagination)
    rows = list(session.scalars(stmt))
    rows, next_page_token = build_next_page_token(rows, pagination.page_size, ordering)

    return ListUsersResponse(
        items=[
            UserDetail(
                id=u.id,
                email=u.email,
                is_privileged=u.is_privileged,
                organizations=[
                    OrganizationSummary(id=o.id, name=o.name) for o in sorted(u.organizations, key=lambda o: o.name)
                ],
                last_logout=u.last_logout,
                has_logged_in=u.iss is not None,
                created_at=u.created_at,
            )
            for u in rows
        ],
        next_page_token=next_page_token,
    )


@router.get("/users/{user_id}")
def get_user(
    target: Annotated[tables.User, Depends(adeps.privileged_target_user)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
) -> GetUserResponse:
    """Fetches details for a single user, including the organizations they belong to.

    Privileged users only. Each returned organization carries summary counts (number of users,
    number of experiments), matching the shape used on the organizations list page.
    """
    membership_rows = (
        session.execute(
            select(tables.UserOrganization, tables.Organization)
            .join(tables.Organization, tables.UserOrganization.organization_id == tables.Organization.id)
            .where(tables.UserOrganization.user_id == target.id)
            .order_by(tables.Organization.name)
        )
    ).all()
    org_ids = [org.id for _, org in membership_rows]
    user_counts: dict[str, int] = {}
    experiment_counts: dict[str, int] = {}
    if org_ids:
        user_count_rows = session.execute(
            select(tables.UserOrganization.organization_id, func.count())
            .where(tables.UserOrganization.organization_id.in_(org_ids))
            .group_by(tables.UserOrganization.organization_id)
        )
        user_counts = {org_id: count for org_id, count in user_count_rows}

        experiment_count_rows = session.execute(
            select(tables.Datasource.organization_id, func.count(tables.Experiment.id))
            .join(tables.Experiment, tables.Experiment.datasource_id == tables.Datasource.id)
            .where(tables.Datasource.organization_id.in_(org_ids))
            .group_by(tables.Datasource.organization_id)
        )
        experiment_counts = {org_id: count for org_id, count in experiment_count_rows}

    return GetUserResponse(
        id=target.id,
        email=target.email,
        is_privileged=target.is_privileged,
        organizations=[
            OrganizationListItem(
                id=org.id,
                name=org.name,
                created_at=org.created_at,
                user_count=user_counts.get(org.id, 0),
                experiment_count=experiment_counts.get(org.id, 0),
                joined_at=membership.created_at,
            )
            for membership, org in membership_rows
        ],
        last_logout=target.last_logout,
        has_logged_in=target.iss is not None,
        created_at=target.created_at,
    )


@router.patch("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def patch_user(
    target: Annotated[tables.User, Depends(adeps.privileged_target_user)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    body: Annotated[PatchUserRequest, Body(...)],
):
    """Updates a user's properties. Privileged users only.

    Currently only supports updating `is_privileged`. Revoking privilege from the last privileged
    user in the system is rejected with a 400.
    """
    if body.is_privileged is not None:
        if target.is_privileged and not body.is_privileged:
            ensure_not_last_privileged_user(session, target.id)
        target.is_privileged = body.is_privileged

    session.commit()
    return GENERIC_SUCCESS


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    target: Annotated[tables.User, Depends(adeps.privileged_target_user)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    user: Annotated[tables.User, Depends(adeps.privileged_caller)],
):
    """Deletes a user. Privileged users only.

    Cascades to remove all organization memberships. Rejects deleting yourself, and rejects deleting
    the last privileged user in the system.
    """
    if target.id == user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot delete yourself.",
        )

    # Defense-in-depth: under normal flow, the privilege requirement + the self-delete check above guarantee
    # this can't be the last privileged user (deleting the last priv user would require being them,
    # and self-delete is blocked). Kept in case those checks are reordered or relaxed in the future.
    if target.is_privileged:
        ensure_not_last_privileged_user(session, target.id)

    session.delete(target)
    session.commit()
    return GENERIC_SUCCESS


@router.get(
    "/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots/{snapshot_id}"
)
def get_snapshot(
    snapshot: Annotated[tables.Snapshot, Depends(adeps.snapshot)],
) -> GetSnapshotResponse:
    """Fetches a snapshot by ID."""
    return GetSnapshotResponse(snapshot=convert_snapshot_to_api_snapshot(snapshot))


@router.get("/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots")
def list_snapshots(
    experiment: Annotated[tables.Experiment, Depends(adeps.org_experiment)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    pagination: Annotated[PaginationQuery, Depends(unbounded_pagination_query_params)],
    status_: Annotated[
        list[SnapshotStatus] | None,
        Query(
            alias="status",
            description="Filter the returned snapshots to only those of this status. May be specified multiple times.",
        ),
    ] = None,
) -> ListSnapshotsResponse:
    """Lists snapshots for an experiment, ordered by timestamp."""
    query = select(tables.Snapshot).where(tables.Snapshot.experiment_id == experiment.id)
    if status_:
        query = query.where(
            tables.Snapshot.status.in_([convert_api_snapshot_status_to_snapshot_status(s) for s in status_])
        )
    ordering = [
        SortField.timestamp(
            column=tables.Snapshot.updated_at,
            attr="updated_at",
            direction="desc",
        ),
        SortField(column=tables.Snapshot.id, attr="id", direction="desc"),
    ]
    query = paginate(query, ordering, pagination)
    snapshots = list(session.scalars(query))
    snapshots, next_page_token = build_next_page_token(snapshots, pagination.page_size, ordering)

    latest_failure = session.scalar(
        select(tables.Snapshot.updated_at)
        .where(tables.Snapshot.experiment_id == experiment.id)
        .where(tables.Snapshot.status == convert_api_snapshot_status_to_snapshot_status(SnapshotStatus.FAILED))
        .order_by(tables.Snapshot.updated_at.desc())
        .limit(1)
    )

    return ListSnapshotsResponse(
        items=[convert_snapshot_to_api_snapshot(snapshot) for snapshot in snapshots],
        latest_failure=latest_failure,
        next_page_token=next_page_token,
    )


@router.delete(
    "/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots/{snapshot_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_snapshot(
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    organization_id: Annotated[str, Path()],
    datasource_id: Annotated[str, Path()],
    experiment_id: Annotated[str, Path()],
    snapshot_id: Annotated[str, Path()],
    allow_missing: Annotated[
        bool,
        Query(description="If true, return a 204 even if the resource does not exist."),
    ] = False,
):
    """Deletes a snapshot."""
    resource_query = (
        select(tables.Snapshot)
        .join(tables.Experiment, tables.Snapshot.experiment_id == tables.Experiment.id)
        .join(tables.Datasource, tables.Experiment.datasource_id == tables.Datasource.id)
        .where(
            tables.Datasource.organization_id == organization_id,
            tables.Experiment.datasource_id == datasource_id,
            tables.Snapshot.experiment_id == experiment_id,
            tables.Snapshot.id == snapshot_id,
        )
    )
    response = handle_delete(
        session, allow_missing, authz.is_user_authorized_on_datasource(user, datasource_id), resource_query
    )
    session.commit()
    return response


@router.post("/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots")
def create_snapshot(
    experiment: Annotated[tables.Experiment, Depends(adeps.org_experiment)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    background_tasks: BackgroundTasks,
) -> CreateSnapshotResponse:
    """Request the asynchronous creation of a snapshot for an experiment.

    Returns the ID of the snapshot. Poll get_snapshot until the job is completed.
    """
    if experiment.state != ExperimentState.COMMITTED:
        raise LateValidationError("You can only snapshot committed experiments.")
    # Aligning with the buffer in snapshotter.py, as we wish to capture +/- 1 day on both sides.
    if experiment.end_date < datetime.now(UTC) - timedelta(days=1):
        raise LateValidationError("You can only snapshot active experiments.")

    snapshot = tables.Snapshot(experiment_id=experiment.id)
    session.add(snapshot)
    session.commit()
    background_tasks.add_task(snapshotter.make_first_snapshot, snapshot.experiment_id, snapshot.id)
    return CreateSnapshotResponse(id=snapshot.id)


@router.get("/organizations")
def list_organizations(
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    pagination: Annotated[PaginationQuery, Depends(pagination_query_params)],
    scope: Annotated[
        Literal["mine", "all"],
        Query(
            description=(
                "`mine` (default) returns organizations the caller is a member of. `all` returns every "
                "organization in the system and requires the caller to be privileged."
            )
        ),
    ] = "mine",
    name_contains: Annotated[
        str | None,
        Query(
            max_length=MAX_LENGTH_OF_NAME_VALUE,
            description="Optional case-insensitive substring filter on the organization's name.",
        ),
    ] = None,
    include_stats: Annotated[
        bool,
        Query(
            description=(
                "When true, populate `user_count` and `experiment_count` on each item. When false "
                "(the default), those fields are returned as null."
            )
        ),
    ] = False,
) -> ListOrganizationsResponse:
    """Returns the list of organizations the caller can see, scoped by the `scope` query param.

    Sorted by name ascending.
    """
    if scope == "all":
        adeps.raise_unless_privileged(user)
        stmt = select(tables.Organization)
    else:
        stmt = select(tables.Organization).join(tables.Organization.users).where(tables.User.id == user.id)

    if name_contains:
        stmt = stmt.where(tables.Organization.name.icontains(name_contains, autoescape=True))

    ordering = [
        SortField(column=tables.Organization.name, attr="name", direction="asc"),
        SortField(column=tables.Organization.id, attr="id", direction="asc"),
    ]
    stmt = paginate(stmt, ordering, pagination)
    rows = list(session.scalars(stmt))
    rows, next_page_token = build_next_page_token(rows, pagination.page_size, ordering)

    user_counts: dict[str, int] = {}
    experiment_counts: dict[str, int] = {}
    if include_stats and rows:
        org_ids = [o.id for o in rows]
        user_count_rows = session.execute(
            select(tables.UserOrganization.organization_id, func.count())
            .where(tables.UserOrganization.organization_id.in_(org_ids))
            .group_by(tables.UserOrganization.organization_id)
        )
        user_counts = {org_id: count for org_id, count in user_count_rows}

        experiment_count_rows = session.execute(
            select(tables.Datasource.organization_id, func.count(tables.Experiment.id))
            .join(tables.Experiment, tables.Experiment.datasource_id == tables.Datasource.id)
            .where(tables.Datasource.organization_id.in_(org_ids))
            .group_by(tables.Datasource.organization_id)
        )
        experiment_counts = {org_id: count for org_id, count in experiment_count_rows}

    return ListOrganizationsResponse(
        items=[
            OrganizationListItem(
                id=org.id,
                name=org.name,
                created_at=org.created_at,
                user_count=user_counts.get(org.id, 0) if include_stats else None,
                experiment_count=experiment_counts.get(org.id, 0) if include_stats else None,
            )
            for org in rows
        ],
        next_page_token=next_page_token,
    )


@router.post("/organizations")
def create_organizations(
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    body: Annotated[CreateOrganizationRequest, Body(...)],
) -> CreateOrganizationResponse:
    """Creates a new organization.

    Any authenticated user may create an organization. The creator is automatically added as a
    member of the new organization.
    """
    organization = create_organization_impl(session, user, body.name)
    session.commit()

    return CreateOrganizationResponse(id=organization.id)


@router.post("/organizations/{organization_id}/webhooks")
def add_webhook_to_organization(
    organization: Annotated[tables.Organization, Depends(adeps.organization)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    body: Annotated[AddWebhookToOrganizationRequest, Body(...)],
) -> AddWebhookToOrganizationResponse:
    """Adds a Webhook to an organization."""
    auth_token, webhook = admin_common.create_webhook_impl(session, organization.id, body)
    session.commit()

    return AddWebhookToOrganizationResponse(
        id=webhook.id,
        direction=webhook.direction,
        type=webhook.type,
        name=webhook.name,
        url=webhook.url,
        auth_token=auth_token,
    )


@router.get("/organizations/{organization_id}/webhooks")
def list_organization_webhooks(
    organization: Annotated[tables.Organization, Depends(adeps.organization)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
) -> ListWebhooksResponse:
    """Lists all the webhooks for an organization."""
    stmt = (
        select(tables.Webhook)
        .where(tables.Webhook.organization_id == organization.id)
        .order_by(tables.Webhook.name, tables.Webhook.id)
    )
    webhooks = session.scalars(stmt)

    # Convert webhooks to WebhookSummary objects
    webhook_summaries = convert_webhooks_to_webhooksummaries(webhooks)

    return ListWebhooksResponse(items=webhook_summaries)


def convert_webhooks_to_webhooksummaries(webhooks):
    return [
        WebhookSummary(
            id=webhook.id,
            direction=webhook.direction,
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
def update_organization_webhook(
    webhook: Annotated[tables.Webhook, Depends(adeps.webhook)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    body: Annotated[UpdateOrganizationWebhookRequest, Body(...)],
):
    """Updates a webhook's name and URL in an organization."""
    webhook.name = body.name
    webhook.url = body.url
    session.commit()
    return GENERIC_SUCCESS


@router.post(
    "/organizations/{organization_id}/webhooks/{webhook_id}/authtoken",
    status_code=status.HTTP_204_NO_CONTENT,
)
def regenerate_webhook_auth_token(
    webhook: Annotated[tables.Webhook, Depends(adeps.webhook)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
):
    """Regenerates the auth token for a webhook in an organization."""
    webhook.auth_token = secrets.token_hex(16)
    session.commit()
    return GENERIC_SUCCESS


@router.delete(
    "/organizations/{organization_id}/webhooks/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_webhook_from_organization(
    organization_id: str,
    webhook_id: str,
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    allow_missing: Annotated[
        bool,
        Query(description="If true, return a 204 even if the resource does not exist."),
    ] = False,
):
    """Removes a Webhook from an organization."""
    resource_query = select(tables.Webhook).where(
        tables.Webhook.organization_id == organization_id,
        tables.Webhook.id == webhook_id,
    )
    response = handle_delete(
        session,
        allow_missing,
        authz.is_user_authorized_on_organization(user, organization_id),
        resource_query,
    )
    session.commit()
    return response


@router.get("/organizations/{organization_id}/events")
def list_organization_events(
    organization: Annotated[tables.Organization, Depends(adeps.organization)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    pagination: Annotated[PaginationQuery, Depends(pagination_query_params)],
) -> ListOrganizationEventsResponse:
    """Returns events in an organization, newest first."""
    stmt = select(tables.Event).where(tables.Event.organization_id == organization.id)
    ordering = [
        SortField.timestamp(
            column=tables.Event.created_at,
            attr="created_at",
            direction="desc",
        ),
        SortField(column=tables.Event.id, attr="id", direction="desc"),
    ]
    stmt = paginate(stmt, ordering, pagination)
    events = list(session.scalars(stmt))
    events, next_page_token = build_next_page_token(events, pagination.page_size, ordering)

    event_summaries = convert_events_to_eventsummaries(events)
    return ListOrganizationEventsResponse(items=event_summaries, next_page_token=next_page_token)


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
                details=data.sanitize().model_dump() if data else None,
                status_icon=data.status_icon() if data else "info",
            )
        )
    return event_summaries


@router.post(
    "/organizations/{organization_id}/events/{event_id}/resend",
    status_code=status.HTTP_204_NO_CONTENT,
)
def resend_organization_event(
    event: Annotated[tables.Event, Depends(adeps.event)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
):
    """Re-enqueues the outbound webhook task that produced a webhook.sent event."""
    data = event.get_data()
    if not isinstance(data, WebhookSentEvent):
        # Only webhook.sent events can be resent.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event type cannot be resent.")
    session.add(
        tables.Task(
            task_type=WEBHOOK_OUTBOUND_TASK_TYPE,
            payload=data.request.model_dump(),
        )
    )
    session.commit()
    return GENERIC_SUCCESS


@router.post("/organizations/{organization_id}/members", status_code=status.HTTP_204_NO_CONTENT)
def add_member_to_organization(
    organization: Annotated[tables.Organization, Depends(adeps.organization_with_members)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    body: Annotated[AddMemberToOrganizationRequest, Body(...)],
):
    """Adds a new member to an organization.

    The authenticated user must be part of the organization to add members.
    """
    if body.email in {u.email for u in organization.users}:
        return GENERIC_SUCCESS

    new_user = session.execute(select(tables.User).where(tables.User.email == body.email)).scalar_one_or_none()
    if new_user is None:
        new_user = tables.User(email=body.email)
        session.add(new_user)
    organization.users.append(new_user)
    session.commit()
    return GENERIC_SUCCESS


@router.delete(
    "/organizations/{organization_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def remove_member_from_organization(
    organization_id: str,
    user_id: str,
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    allow_missing: Annotated[
        bool,
        Query(description="If true, return a 204 even if the resource does not exist."),
    ] = False,
):
    """Removes a member from an organization.

    The authenticated user must be part of the organization to remove members. Privileged users may
    remove members from any organization. A user cannot remove themselves from an organization.
    """
    if user_id == user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot remove yourself from an organization.",
        )

    resource_query = select(tables.UserOrganization).where(
        tables.UserOrganization.organization_id == organization_id,
        tables.UserOrganization.user_id == user_id,
    )
    is_authorized = (
        select(literal(True)) if user.is_privileged else authz.is_user_authorized_on_organization(user, organization_id)
    )
    response = handle_delete(session, allow_missing, is_authorized, resource_query)
    session.commit()
    return response


@router.patch("/organizations/{organization_id}")
def update_organization(
    organization: Annotated[tables.Organization, Depends(adeps.organization)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    body: Annotated[UpdateOrganizationRequest, Body(...)],
):
    """Updates an organization's properties.

    The authenticated user must be a member of the organization.
    Currently only supports updating the organization name.
    """
    if body.name is not None:
        organization.name = body.name

    session.commit()
    return GENERIC_SUCCESS


@router.get("/organizations/{organization_id}")
def get_organization(
    organization: Annotated[tables.Organization, Depends(adeps.organization)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
) -> GetOrganizationResponse:
    """Returns detailed information about a specific organization.

    The authenticated user must be a member of the organization.
    """
    users_stmt = (
        select(tables.User)
        .join(tables.UserOrganization)
        .filter(tables.UserOrganization.organization_id == organization.id)
    )
    users = session.scalars(users_stmt)

    datasources_stmt = select(tables.Datasource).filter(tables.Datasource.organization_id == organization.id)
    datasources = session.scalars(datasources_stmt)

    return GetOrganizationResponse(
        id=organization.id,
        name=organization.name,
        users=[
            UserSummary(id=u.id, email=u.email, is_privileged=u.is_privileged)
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
                organization_name=organization.name,
            )
            for ds in sorted(datasources, key=lambda x: x.name)
        ],
    )


@router.get("/organizations/{organization_id}/datasources")
def list_organization_datasources(
    organization: Annotated[tables.Organization, Depends(adeps.organization)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
) -> ListDatasourcesResponse:
    """Returns a list of datasources accessible to the authenticated user for an org."""
    experiment_count = (
        select(tables.Experiment.datasource_id, func.count().label("experiment_count"))
        .group_by(tables.Experiment.datasource_id)
        .subquery()
    )
    stmt = (
        select(tables.Datasource)
        .join(tables.Organization)
        .join(tables.Organization.users)
        .outerjoin(experiment_count, tables.Datasource.id == experiment_count.c.datasource_id)
        .where(tables.User.id == user.id)
        .where(tables.Organization.id == organization.id)
        .order_by(
            func.coalesce(experiment_count.c.experiment_count, 0).desc(),
            tables.Datasource.name.asc(),
        )
    )

    datasources = session.scalars(stmt)

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

    return ListDatasourcesResponse(items=[convert_ds_to_summary(ds) for ds in datasources])


@router.post(
    "/datasources",
    responses=DWH_CONNECTION_RESPONSES,
)
def create_datasource(
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    body: Annotated[CreateDatasourceRequest, Body(...)],
    connectivity_check: Annotated[
        bool,
        Query(description="When true, validate datasource connectivity before creation."),
    ] = False,
) -> CreateDatasourceResponse:
    """Creates a new datasource for the specified organization."""
    org = adeps.load_organization_or_raise(session, user, body.organization_id)

    raise_unless_safe_hostname(body.dsn)

    config = RemoteDatabaseConfig(type="remote", dwh=api_dsn_to_settings_dwh(body.dsn))
    if connectivity_check and config.dwh.driver != "none":
        with DwhSession.open(config.dwh) as dwh:
            dwh.connectivity_check()

    datasource = admin_common.create_datasource_impl(session, org, body.name, config)
    session.commit()

    return CreateDatasourceResponse(id=datasource.id)


@router.patch("/datasources/{datasource_id}", status_code=status.HTTP_204_NO_CONTENT)
def update_datasource(
    datasource: Annotated[tables.Datasource, Depends(adeps.datasource)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    body: UpdateDatasourceRequest,
):
    if body.name is not None:
        datasource.name = body.name
    if body.dsn is not None:
        raise_unless_safe_hostname(body.dsn)
        cfg = datasource.get_config()
        cfg.dwh = api_dsn_to_settings_dwh(body.dsn, cfg.dwh)
        datasource.set_config(cfg)

    datasource.clear_table_list()
    invalidate_inspect_tables = delete(tables.DatasourceTablesInspected).where(
        tables.DatasourceTablesInspected.datasource_id == datasource.id
    )
    session.execute(invalidate_inspect_tables)

    session.commit()
    return GENERIC_SUCCESS


@router.get("/datasources/{datasource_id}")
def get_datasource(
    datasource: Annotated[tables.Datasource, Depends(adeps.datasource_with_organization)],
) -> GetDatasourceResponse:
    """Returns detailed information about a specific datasource."""
    config = datasource.get_config()
    return GetDatasourceResponse(
        id=datasource.id,
        name=datasource.name,
        dsn=admin_api_converters.settings_dwh_to_api_dsn(config.dwh),
        organization_id=datasource.organization_id,
        organization_name=datasource.organization.name,
    )


@router.get(
    "/datasources/{datasource_id}/inspect",
    responses=DWH_CONNECTION_AND_NOT_FOUND_RESPONSES,
)
def inspect_datasource(
    datasource: Annotated[tables.Datasource, Depends(adeps.datasource)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
) -> InspectDatasourceResponse:
    """Verifies connectivity to a datasource and returns a list of readable tables."""
    if datasource.get_config().dwh.driver == "none":
        return InspectDatasourceResponse(tables=[])

    if not refresh and cache_is_fresh(datasource.table_list_updated) and datasource.table_list is not None:
        return InspectDatasourceResponse(tables=datasource.table_list)

    with clear_db_table_cache_on_error(session, datasource):
        config = datasource.get_config()
        with DwhSession.open(config.dwh) as dwh:
            tablenames = dwh.list_tables()
        datasource.set_table_list(tablenames)
        session.commit()
        return InspectDatasourceResponse(tables=tablenames)


def invalidate_inspect_table_cache(session: Session, datasource_id: str) -> None:
    """Invalidates all table inspection cache entries for a datasource."""
    session.execute(
        delete(tables.DatasourceTablesInspected).where(tables.DatasourceTablesInspected.datasource_id == datasource_id)
    )


@router.get(
    "/datasources/{datasource_id}/inspect/{table_name}",
    responses=DWH_CONNECTION_AND_NOT_FOUND_RESPONSES,
)
def inspect_table_in_datasource(
    datasource: Annotated[tables.Datasource, Depends(adeps.datasource)],
    table_name: str,
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
) -> InspectDatasourceTableResponse:
    """Inspects a single table in a datasource and returns a summary of its fields."""
    datasource_id = datasource.id
    if (
        not refresh
        and (cached := session.get(tables.DatasourceTablesInspected, (datasource_id, table_name)))
        and cache_is_fresh(cached.response_last_updated)
        and cached.response is not None
    ):
        return InspectDatasourceTableResponse.model_validate(cached.response)

    config = datasource.get_config()

    if config.dwh.driver == "none":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only remote datasources may be inspected.",
        )

    invalidate_inspect_table_cache(session, datasource_id)
    session.commit()

    with DwhSession.open(config.dwh) as dwh:
        # CannotFindTableError will be handled by exceptionhandlers.py.
        table = dwh.inspect_table(table_name)
    response = create_inspect_table_response_from_table(table)

    session.add(
        tables.DatasourceTablesInspected(
            datasource_id=datasource_id,
            table_name=table_name,
            response=response.model_dump(),
            response_last_updated=datetime.now(UTC),
        )
    )
    session.commit()

    return response


@router.delete(
    "/organizations/{organization_id}/datasources/{datasource_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_datasource(
    session: Annotated[Session, Depends(xngin_sync_db_session)],
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
    resource_query = select(tables.Datasource).where(
        tables.Datasource.organization_id == organization_id,
        tables.Datasource.id == datasource_id,
    )

    response = handle_delete(
        session,
        allow_missing,
        authz.is_user_authorized_on_organization(user, organization_id),
        resource_query,
    )
    session.commit()
    return response


@router.get("/datasources/{datasource_id}/apikeys")
def list_api_keys(
    datasource: Annotated[tables.Datasource, Depends(adeps.datasource_with_api_keys)],
) -> ListApiKeysResponse:
    """Returns API keys that have access to the datasource."""
    return ListApiKeysResponse(
        items=[
            ApiKeySummary(
                id=api_key.id,
                datasource_id=api_key.datasource_id,
                organization_id=datasource.organization_id,
                organization_name=datasource.organization.name,
            )
            for api_key in sorted(datasource.api_keys, key=lambda a: a.id)
        ]
    )


@router.post("/datasources/{datasource_id}/apikeys")
def create_api_key(
    datasource: Annotated[tables.Datasource, Depends(adeps.datasource)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
) -> CreateApiKeyResponse:
    """Creates an API key for the specified datasource.

    The user must belong to the organization that owns the requested datasource.
    """
    label, key = make_key()
    key_hash = hash_key_or_raise(key)
    api_key = tables.ApiKey(id=label, key=key_hash, datasource_id=datasource.id)
    session.add(api_key)
    session.commit()
    return CreateApiKeyResponse(id=label, datasource_id=datasource.id, key=key)


@router.delete(
    "/datasources/{datasource_id}/apikeys/{api_key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_api_key(
    datasource_id: str,
    session: Annotated[Session, Depends(xngin_sync_db_session)],
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
    response = handle_delete(
        session,
        allow_missing,
        authz.is_user_authorized_on_datasource(user, datasource_id),
        resource_query,
    )
    session.commit()
    return response


@router.post("/datasources/{datasource_id}/experiments")
def create_experiment(
    datasource: Annotated[tables.Datasource, Depends(adeps.datasource)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    body: CreateExperimentRequest,
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
    if body.design_spec.ids_are_present():
        raise LateValidationError("Invalid DesignSpec: UUIDs must not be set.")

    # Validate webhook IDs exist and belong to organization
    organization_id = datasource.organization_id
    validated_webhooks = validate_webhooks(
        session=session, organization_id=organization_id, request_webhooks=body.webhooks
    )

    response = experiments_common.create_experiment_impl(
        request=body,
        datasource=datasource,
        xngin_session=session,
        stratify_on_metrics=stratify_on_metrics,
        random_state=random_state,
        validated_webhooks=validated_webhooks,
    )
    session.commit()
    return response


@router.get(
    "/datasources/{datasource_id}/experiments/{experiment_id}/analyze",
    description="""
    For preassigned experiments, and online experiments (except contextual bandits),
    returns an analysis of the experiment's performance, given datasource and experiment ID.""",
)
def analyze_experiment(
    experiment: Annotated[tables.Experiment, Depends(adeps.experiment_for_analysis)],
    datasource: Annotated[tables.Datasource, Depends(adeps.datasource)],
    xngin_session: Annotated[Session, Depends(xngin_sync_db_session)],
    baseline_arm_id: Annotated[
        str | None,
        Query(
            description="UUID of the baseline arm. If None, the first design spec arm is used.",
        ),
    ] = None,
) -> ExperimentAnalysisResponse:
    design_spec = ExperimentStorageConverter(experiment).get_design_spec()
    match design_spec:
        case PreassignedFrequentistExperimentSpec() | OnlineFrequentistExperimentSpec():
            # Always assume the first arm is the baseline; UI can override this.
            baseline_arm_id = baseline_arm_id or design_spec.arms[0].arm_id
            assert baseline_arm_id is not None
            return experiments_common.analyze_experiment_freq_impl(
                xngin_session, datasource.get_config(), experiment, baseline_arm_id, design_spec.metrics
            )
        case MABExperimentSpec() | MABDwhExperimentSpec():
            return experiments_common.analyze_experiment_bandit_impl(xngin_session, experiment)
        case CMABExperimentSpec():
            raise LateValidationError(
                """Invalid experiment type for bandit analysis; for CMAB experiments,
                use the corresponding POST endpoint.""",
            )
        case _:
            assert_never(design_spec)


@router.post(
    "/datasources/{datasource_id}/experiments/{experiment_id}/analyze_cmab",
    description="""
    For contextual bandit experiments, returns an analysis of the experiment's performance,
    given datasource and experiment ID and context values as input.""",
)
def analyze_cmab_experiment(
    experiment: Annotated[tables.Experiment, Depends(adeps.experiment_with_contexts)],
    xngin_session: Annotated[Session, Depends(xngin_sync_db_session)],
    body: CMABContextInputRequest,
) -> ExperimentAnalysisResponse:
    if experiment.experiment_type != ExperimentsType.CMAB_ONLINE.value:
        raise LateValidationError(
            f"Experiment {experiment.id} is a {experiment.experiment_type} experiment, and not a "
            f"{ExperimentsType.CMAB_ONLINE.value} experiment. Please use the corresponding GET endpoint to "
            f"retrieve an experiment analysis."
        )

    if body.context_inputs is None:
        raise LateValidationError("context_inputs must be provided when analyzing a CMAB experiment.")
    sorted_context_inputs = sort_contexts_by_id_or_raise(experiment.contexts, body.context_inputs)

    return experiments_common.analyze_experiment_bandit_impl(
        xngin_session, experiment, context_vals=[ci.context_value for ci in sorted_context_inputs]
    )


EXPERIMENT_STATE_TRANSITION_RESPONSES: dict[int | str, dict[str, Any]] = {
    204: {"model": None, "description": "Experiment state updated successfully."},
    409: {
        "model": HTTPExceptionError,
        "description": "Experiment is not in a valid state to transition to the target state.",
    },
}


@router.post(
    "/datasources/{datasource_id}/experiments/{experiment_id}/commit",
    responses=EXPERIMENT_STATE_TRANSITION_RESPONSES,
    status_code=status.HTTP_204_NO_CONTENT,
)
def commit_experiment(
    experiment: Annotated[tables.Experiment, Depends(adeps.experiment_for_commit)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
):
    result = experiments_common.commit_experiment_impl(session, experiment)
    if result == experiments_common.CommitExperimentResult.INVALID_STATE:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Invalid state: {experiment.state}")
    session.commit()
    return GENERIC_SUCCESS


@router.post(
    "/datasources/{datasource_id}/experiments/{experiment_id}/abandon",
    responses=EXPERIMENT_STATE_TRANSITION_RESPONSES,
    status_code=status.HTTP_204_NO_CONTENT,
)
def abandon_experiment(
    experiment: Annotated[tables.Experiment, Depends(adeps.experiment)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
):
    result = experiments_common.abandon_experiment_impl(experiment)
    if result == AbandonExperimentResult.INVALID_STATE:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Invalid state: {experiment.state}")
    session.commit()
    return GENERIC_SUCCESS


@router.get("/organizations/{organization_id}/experiments")
def list_organization_experiments(
    organization: Annotated[tables.Organization, Depends(adeps.organization)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
) -> ListExperimentsResponse:
    """Returns a list of experiments in the organization."""
    return experiments_common.list_organization_or_datasource_experiments_impl(
        xngin_session=session, organization_id=organization.id
    )


@router.get("/datasources/{datasource_id}/experiments/{experiment_id}")
def get_experiment_for_ui(
    experiment: Annotated[tables.Experiment, Depends(adeps.experiment_for_ui)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
) -> GetExperimentForUiResponse:
    """Returns the experiment with the specified ID."""
    return GetExperimentForUiResponse(
        config=experiments_common.get_experiment_impl(session, experiment),
        experiment_schema=make_schema_from_experiment(experiment),
    )


@router.get(
    "/datasources/{datasource_id}/experiments/{experiment_id}/assignments/csv",
    summary=(
        "Export experiment assignments as CSV file; BalanceCheck not included. "
        "csv header form: participant_id,[cluster_key,]arm_id,arm_name,strata_name1,strata_name2,..."
    ),
    response_class=CsvStreamingResponse,
)
async def get_experiment_assignments_as_csv_for_ui(
    experiment: Annotated[tables.Experiment, Depends(adeps.experiment_for_csv_export)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> CsvStreamingResponse:
    # TODO: update for bandits
    return experiments_common_csv.get_experiment_assignments_as_csv_impl(session, experiment)


@router.patch("/datasources/{datasource_id}/experiments/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def update_experiment(
    experiment: Annotated[tables.Experiment, Depends(adeps.experiment)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    body: UpdateExperimentRequest,
):
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

    session.commit()
    return GENERIC_SUCCESS


@router.delete(
    "/datasources/{datasource_id}/experiments/{experiment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_experiment(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    allow_missing: Annotated[
        bool,
        Query(description="If true, return a 204 even if the resource does not exist."),
    ] = False,
):
    """Deletes the experiment with the specified ID."""
    resource_query = select(tables.Experiment).where(
        tables.Experiment.datasource_id == datasource_id,
        tables.Experiment.id == experiment_id,
    )
    response = handle_delete(
        session, allow_missing, authz.is_user_authorized_on_datasource(user, datasource_id), resource_query
    )
    session.commit()
    return response


@router.delete(
    "/datasources/{datasource_id}/experiments/{experiment_id}/data",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_experiment_data(
    experiment: Annotated[tables.Experiment, Depends(adeps.experiment)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    body: DeleteExperimentDataRequest,
):
    """Deletes specific data associated with an experiment."""
    if body.assignments:
        etype = ExperimentsType(experiment.experiment_type)
        if etype.is_freq():
            session.execute(delete(tables.ArmAssignment).where(tables.ArmAssignment.experiment_id == experiment.id))
        else:
            session.execute(delete(tables.Draw).where(tables.Draw.experiment_id == experiment.id))
        session.execute(
            delete(tables.ArmStats).where(
                tables.ArmStats.arm_id.in_(select(tables.Arm.id).where(tables.Arm.experiment_id == experiment.id))
            )
        )

    if body.snapshots:
        session.execute(delete(tables.Snapshot).where(tables.Snapshot.experiment_id == experiment.id))

    session.commit()
    return GENERIC_SUCCESS


@router.patch(
    "/datasources/{datasource_id}/experiments/{experiment_id}/arms/{arm_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def update_arm(
    experiment: Annotated[tables.Experiment, Depends(adeps.experiment)],
    arm_id: str,
    body: UpdateArmRequest,
    session: Annotated[Session, Depends(xngin_sync_db_session)],
):
    if experiment.state != ExperimentState.COMMITTED:
        raise LateValidationError("Experiment must have been committed to update arms.")

    arm = next((arm for arm in experiment.arms if arm.id == arm_id), None)
    if arm is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Arm not found.")

    if body.name is not None:
        arm.name = body.name
    if body.description is not None:
        arm.description = body.description

    session.commit()
    return GENERIC_SUCCESS


@router.post(
    "/datasources/{datasource_id}/power",
    responses=DWH_CONNECTION_AND_NOT_FOUND_RESPONSES,
)
def power_check(
    datasource: Annotated[tables.Datasource, Depends(adeps.datasource)],
    body: PowerRequest,
) -> PowerResponse:
    """Performs a power check for the specified datasource."""
    design_spec = body.design_spec
    if isinstance(datasource.config, NoDwh):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Power checks are not supported for datasources without a data warehouse.",
        )
    dsconfig = datasource.get_config()

    with DwhSession.open(dsconfig.dwh) as dwh:
        sa_table = dwh.inspect_table(design_spec.table_name)
        # Validate the fields used in the design spec are present in the table and that filter values are valid.
        _ = convert_table_to_fields_or_raise(sa_table, design_spec)

        filters = design_spec.filters
        cluster_key = None
        desired_n_clusters = None
        if isinstance(design_spec, PreassignedFrequentistExperimentSpec):
            cluster_key = design_spec.cluster_key
            desired_n_clusters = design_spec.desired_n_clusters
        # Exclude rows without a valid cluster key.
        if cluster_key is not None:
            filters = [*filters, Filter(field_name=cluster_key, relation=Relation.EXCLUDES, value=[None])]

        metric_stats = dwh.run(get_stats_on_metrics, sa_table, design_spec.metrics, filters)

        # Augment with cluster-level stats if this is a cluster-randomized design.
        if cluster_key is not None:
            request_metrics_by_name = {m.field_name: m for m in design_spec.metrics}
            # Derive stats from the dwh only for metrics without user-provided ICC, in one query.
            db_derived_metrics = [
                metric_stat.field_name
                for metric_stat in metric_stats
                if request_metrics_by_name[metric_stat.field_name].icc is None
            ]
            db_cluster_stats = (
                dwh.run(
                    calculate_cluster_stats_from_database,
                    sa_table,
                    cluster_key,
                    db_derived_metrics,
                    filters,
                )
                if db_derived_metrics
                else {}
            )
            for metric_stat in metric_stats:
                req_metric = request_metrics_by_name[metric_stat.field_name]
                # If the user provided ICC, avg_cluster_size, and cv, use them instead of deriving from the dwh.
                if req_metric.icc is not None:
                    metric_stat.icc = req_metric.icc
                    metric_stat.avg_cluster_size = req_metric.avg_cluster_size
                    metric_stat.cv = req_metric.cv
                else:
                    cluster_stats = db_cluster_stats[metric_stat.field_name]
                    metric_stat.icc = cluster_stats["icc"]
                    metric_stat.avg_cluster_size = cluster_stats["avg_cluster_size"]
                    metric_stat.cv = cluster_stats["cv"]

    arm_weights = design_spec.get_validated_arm_weights()

    return PowerResponse(
        analyses=check_power(
            metrics=metric_stats,
            n_arms=len(design_spec.arms),
            power=design_spec.power,
            alpha=design_spec.alpha,
            arm_weights=arm_weights,
            desired_n=design_spec.desired_n,
            desired_n_clusters=desired_n_clusters,
        )
    )


def raise_unless_safe_hostname(dsn):
    """Raises a 400 if the DNS name in dsn is possibly attempting to connect to resources on local network."""
    if isinstance(dsn, PostgresDsn | RedshiftDsn):
        try:
            safe_resolve(dsn.host)
        except DnsLookupError as err:
            raise HTTPException(
                status_code=400,
                detail="DNS resolution failed. Check datasource hostname and try again.",
            ) from err
