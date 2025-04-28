"""Implements a basic Admin API."""

import secrets
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Annotated
import uuid

import google.api_core.exceptions
import sqlalchemy
import sqlalchemy.orm
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
from pydantic import BaseModel
from sqlalchemy import delete, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from xngin.apiserver import flags, settings
from xngin.apiserver.routers.stateless_api_types import (
    ArmAnalysis,
    DataType,
    ExperimentAnalysis,
    GetMetricsResponseElement,
    GetStrataResponseElement,
    MetricAnalysis,
    PowerRequest,
    PowerResponse,
)
from xngin.apiserver.apikeys import hash_key, make_key
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.dns.safe_resolve import safe_resolve
from xngin.apiserver.dwh.queries import get_participant_metrics, query_for_participants
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.models.tables import (
    ApiKey,
    Datasource,
    DatasourceTablesInspected,
    Event,
    Experiment,
    Organization,
    ParticipantTypesInspected,
    User,
    UserOrganization,
    Webhook,
)
from xngin.apiserver.routers import experiments, experiments_api_types
from xngin.apiserver.routers.admin_api_types import (
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
    FieldMetadata,
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
    UpdateParticipantsTypeRequest,
    UpdateParticipantsTypeResponse,
    UserSummary,
    WebhookSummary,
)
from xngin.apiserver.routers.stateless_api import (
    create_col_to_filter_meta_mapper,
    generate_field_descriptors,
    power_check_impl,
    validate_schema_metrics_or_raise,
)
from xngin.apiserver.routers.experiments_api_types import (
    ExperimentConfig,
    GetParticipantAssignmentResponse,
)
from xngin.apiserver.routers.oidc_dependencies import TokenInfo, require_oidc_token
from xngin.apiserver.settings import (
    Dsn,
    ParticipantsConfig,
    ParticipantsDef,
    RemoteDatabaseConfig,
    infer_table,
)
from xngin.stats.analysis import analyze_experiment as analyze_experiment_impl

GENERIC_SUCCESS = Response(status_code=status.HTTP_204_NO_CONTENT)
RESPONSE_CACHE_MAX_AGE_SECONDS = timedelta(minutes=15).seconds


class HTTPExceptionError(BaseModel):
    detail: str


# This defines the response codes we can expect our API to return in the normal course of operation and would be
# useful for our developers to think about.
#
# FastAPI will add a case for 422 (method argument or pydantic validation errors) automatically. 500s are
# intentionally omitted here as they (ideally) should never happen.
STANDARD_ADMIN_RESPONSES = {
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


def is_enabled():
    """Feature flag: Returns true iff OIDC is enabled."""
    return flags.ENABLE_ADMIN


def cache_is_fresh(updated: datetime | None):
    return updated and datetime.now(UTC) - updated < timedelta(minutes=5)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


router = APIRouter(
    lifespan=lifespan,
    prefix="/m",
    responses=STANDARD_ADMIN_RESPONSES,
    dependencies=[
        Depends(require_oidc_token)
    ],  # All routes in this router require authentication.
)


def user_from_token(
    session: Annotated[Session, Depends(xngin_db_session)],
    token_info: Annotated[TokenInfo, Depends(require_oidc_token)],
) -> User:
    """Dependency for fetching the User record matching the authenticated user's email.

    This may raise a 400, 401, or 403.
    """
    user = session.query(User).filter(User.email == token_info.email).first()
    if not user:
        # Privileged users will have a user and an organization created on the fly.
        if token_info.is_privileged():
            user = User(email=token_info.email, is_privileged=True)
            session.add(user)
            organization = Organization(name="My Organization")
            session.add(organization)
            organization.users.append(user)
            if dev_dsn := flags.XNGIN_DEVDWH_DSN:
                # TODO: Also add a default participant type.
                config = RemoteDatabaseConfig(
                    participants=[], type="remote", dwh=Dsn.from_url(dev_dsn)
                )
                datasource = Datasource(
                    name="Local DWH", organization=organization
                ).set_config(config)
                session.add(datasource)
            session.commit()
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"No user found with email: {token_info.email}",
            )
    return user


def get_organization_or_raise(session: Session, user: User, organization_id: str):
    """Reads the requested organization from the database. Raises 404 if disallowed or not found."""
    stmt = (
        select(Organization)
        .join(UserOrganization)
        .where(Organization.id == organization_id)
        .where(UserOrganization.user_id == user.id)
    )
    org = session.execute(stmt).scalar_one_or_none()
    if org is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found."
        )
    return org


def get_datasource_or_raise(session: Session, user: User, datasource_id: str):
    """Reads the requested datasource from the database. Raises 404 if disallowed or not found."""
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


def get_experiment_via_ds_or_raise(
    session: Session, ds: Datasource, experiment_id: str
) -> Experiment:
    """Reads the requested experiment (related to the given datasource) from the database. Raises 404 if not found."""
    stmt = (
        select(Experiment)
        .where(Experiment.datasource_id == ds.id)
        .where(Experiment.id == experiment_id)
    )
    exp = session.execute(stmt).scalar_one_or_none()
    if exp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found."
        )
    return exp


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
    user: Annotated[User, Depends(user_from_token)],
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

    organization = Organization(name=body.name)
    session.add(organization)
    organization.users.append(user)  # Add the creating user to the organization
    session.commit()

    return CreateOrganizationResponse(id=organization.id)


@router.post("/organizations/{organization_id}/webhooks")
def add_webhook_to_organization(
    organization_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
    body: Annotated[AddWebhookToOrganizationRequest, Body(...)],
) -> AddWebhookToOrganizationResponse:
    """Adds a Webhook to an organization."""
    # Verify user has access to the organization
    org = get_organization_or_raise(session, user, organization_id)

    # Generate a secure auth token
    auth_token = secrets.token_hex(16)

    # Create and save the webhook
    webhook = Webhook(
        type=body.type, url=body.url, auth_token=auth_token, organization_id=org.id
    )
    session.add(webhook)
    session.commit()

    return AddWebhookToOrganizationResponse(
        id=webhook.id, type=webhook.type, url=webhook.url, auth_token=auth_token
    )


@router.get("/organizations/{organization_id}/webhooks")
def list_organization_webhooks(
    organization_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
) -> ListWebhooksResponse:
    """Lists all the webhooks for an organization."""
    # Verify user has access to the organization
    org = get_organization_or_raise(session, user, organization_id)

    # Query for webhooks
    stmt = select(Webhook).where(Webhook.organization_id == org.id)
    webhooks = session.scalars(stmt).all()

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


@router.delete(
    "/organizations/{organization_id}/webhooks/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_webhook_from_organization(
    organization_id: str,
    webhook_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
):
    """Removes a Webhook from an organization."""
    # Verify user has access to the organization
    org = get_organization_or_raise(session, user, organization_id)

    # Find and delete the webhook
    stmt = (
        delete(Webhook)
        .where(Webhook.id == webhook_id)
        .where(Webhook.organization_id == org.id)
    )
    result = session.execute(stmt)

    if result.rowcount == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found"
        )

    session.commit()
    return GENERIC_SUCCESS


@router.get("/organizations/{organization_id}/events")
def list_organization_events(
    organization_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
) -> ListOrganizationEventsResponse:
    """Returns the most recent 200 events in an organization."""
    # Verify user has access to the organization
    org = get_organization_or_raise(session, user, organization_id)

    # Query for the most recent 200 events
    stmt = (
        select(Event)
        .where(Event.organization_id == org.id)
        .order_by(Event.created_at.desc())
        .limit(200)
    )
    events = session.scalars(stmt).all()

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
def add_member_to_organization(
    organization_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
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

    if not user.is_privileged:
        # Verify user is a member of the organization
        _authz_check = get_organization_or_raise(session, user, organization_id)

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
    _authz_check = get_organization_or_raise(session, user, organization_id)
    # Prevent users from removing themselves from an organization
    if user_id == user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot remove yourself from an organization",
        )
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


@router.patch("/organizations/{organization_id}")
def update_organization(
    organization_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
    body: Annotated[UpdateOrganizationRequest, Body(...)],
):
    """Updates an organization's properties.

    The authenticated user must be a member of the organization.
    Currently only supports updating the organization name.
    """
    org = get_organization_or_raise(session, user, organization_id)

    if body.name is not None:
        org.name = body.name

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
def list_organization_datasources(
    organization_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
) -> ListDatasourcesResponse:
    """Returns a list of datasources accessible to the authenticated user for an org."""
    _authz_check = get_organization_or_raise(session, user, organization_id)
    stmt = (
        select(Datasource)
        .join(Organization)
        .join(Organization.users)
        .where(User.id == user.id)
    )
    if organization_id is not None:
        stmt = stmt.where(Organization.id == organization_id)

    result = session.execute(stmt)
    datasources = result.scalars().all()

    def convert_ds_to_summary(ds: Datasource) -> DatasourceSummary:
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
def create_datasource(
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
    body: Annotated[CreateDatasourceRequest, Body(...)],
) -> CreateDatasourceResponse:
    """Creates a new datasource for the specified organization."""
    org = get_organization_or_raise(session, user, body.organization_id)

    if (
        body.dwh.driver == "bigquery"
        and body.dwh.credentials.type != "serviceaccountinfo"
    ):
        raise HTTPException(
            status_code=400,
            detail="BigQuery credentials must be specified using type=serviceaccountinfo",
        )
    if body.dwh.driver in {"postgresql+psycopg", "postgresql+psycopg2"}:
        _ = safe_resolve(body.dwh.host)  # TODO: handle this exception more gracefully

    config = RemoteDatabaseConfig(participants=[], type="remote", dwh=body.dwh)

    datasource = Datasource(name=body.name, organization_id=org.id).set_config(config)
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

        # Invalidate cached inspections.
        ds.set_config(cfg)
        ds.clear_table_list()
        invalidate_inspect_tables = delete(DatasourceTablesInspected).where(
            DatasourceTablesInspected.datasource_id == datasource_id
        )
        invalidate_inspect_ptype = delete(ParticipantTypesInspected).where(
            ParticipantTypesInspected.datasource_id == datasource_id
        )
        session.execute(invalidate_inspect_tables)
        session.execute(invalidate_inspect_ptype)
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
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
) -> InspectDatasourceResponse:
    """Verifies connectivity to a datasource and returns a list of readable tables."""
    ds = get_datasource_or_raise(session, user, datasource_id)
    if not refresh and cache_is_fresh(ds.table_list_updated):
        return InspectDatasourceResponse(tables=ds.table_list)
    try:
        try:
            config = ds.get_config()

            # Hack for redshift's lack of reflection support.
            if config.dwh.is_redshift():
                query = text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema IN (:search_path) ORDER BY table_name"
                )
                with config.dbsession() as dwh_session:
                    result = dwh_session.execute(
                        query, {"search_path": config.dwh.search_path or "public"}
                    )
                    tables = result.scalars().all()
            else:
                inspected = sqlalchemy.inspect(config.dbengine())
                tables = list(
                    sorted(inspected.get_table_names() + inspected.get_view_names())
                )

            ds.set_table_list(tables)
            session.commit()
            return InspectDatasourceResponse(tables=tables)
        except OperationalError as exc:
            if is_postgres_database_not_found_error(exc):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
                ) from exc
            raise
        except google.api_core.exceptions.NotFound as exc:
            # Google returns a 404 when authentication succeeds but when the specified datasource does not exist.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc
    except:
        ds.clear_table_list()
        session.commit()
        raise


def is_postgres_database_not_found_error(exc):
    return (
        exc.args
        and isinstance(exc.args[0], str)
        and "FATAL:  database" in exc.args[0]
        and "does not exist" in exc.args[0]
    )


def create_inspect_table_response_from_table(
    table: sqlalchemy.Table,
) -> InspectDatasourceTableResponse:
    """Creates an InspectDatasourceTableResponse from a sqlalchemy.Table.

    This is similar to config_sheet.create_schema_from_table but tailored to use in the API.
    """
    possible_id_columns = {
        c.name
        for c in table.columns.values()
        if c.name.endswith("id") or isinstance(c.type, sqlalchemy.sql.sqltypes.UUID)
    }
    primary_key_columns = {c.name for c in table.columns.values() if c.primary_key}
    if len(primary_key_columns) > 1:
        # If there is more than one PK, it probably isn't usable for experiments.
        primary_key_columns = set()
    possible_id_columns |= primary_key_columns

    collected = []
    for column in table.columns.values():
        type_hint = column.type
        data_type = DataType.match(type_hint)
        if data_type.is_supported():
            collected.append(
                FieldMetadata(
                    field_name=column.name,
                    data_type=data_type,
                    description=column.comment or "",
                )
            )

    return InspectDatasourceTableResponse(
        detected_unique_id_fields=list(sorted(possible_id_columns)),
        fields=list(sorted(collected, key=lambda f: f.field_name)),
    )


def invalidate_inspect_table_cache(session, datasource_id):
    """Invalidates all table inspection cache entries for a datasource."""
    session.execute(
        delete(DatasourceTablesInspected).where(
            DatasourceTablesInspected.datasource_id == datasource_id
        )
    )


@router.get("/datasources/{datasource_id}/inspect/{table_name}")
def inspect_table_in_datasource(
    datasource_id: str,
    table_name: str,
    user: Annotated[User, Depends(user_from_token)],
    session: Annotated[Session, Depends(xngin_db_session)],
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
) -> InspectDatasourceTableResponse:
    """Inspects a single table in a datasource and returns a summary of its fields."""
    ds = get_datasource_or_raise(session, user, datasource_id)
    if (
        not refresh
        and (
            cached := session.get(
                DatasourceTablesInspected, (datasource_id, table_name)
            )
        )
        and cache_is_fresh(cached.response_last_updated)
    ):
        return cached.get_response()

    config = ds.get_config()

    invalidate_inspect_table_cache(session, datasource_id)
    session.commit()

    engine = config.dbengine()
    # CannotFindTableError will be handled by exceptionhandlers.py.
    table = settings.infer_table(
        engine, table_name, use_reflection=config.supports_reflection()
    )
    response = create_inspect_table_response_from_table(table)

    session.add(
        DatasourceTablesInspected(
            datasource_id=datasource_id, table_name=table_name
        ).set_response(response)
    )
    session.commit()

    return response


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


@router.get("/datasources/{datasource_id}/participants/{participant_id}/inspect")
def inspect_participant_types(
    datasource_id: str,
    participant_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
) -> InspectParticipantTypesResponse:
    """Returns filter, strata, and metric field metadata for a participant type, including exemplars for filter fields."""
    dsconfig = get_datasource_or_raise(session, user, datasource_id).get_config()
    # CannotFindParticipantsError will be handled by exceptionhandlers.
    pconfig = dsconfig.find_participants(participant_id)
    if pconfig.type == "sheet":
        raise HTTPException(
            status_code=405, detail="Sheet schemas cannot be inspected."
        )

    if (
        not refresh
        and (
            cached := session.get(
                ParticipantTypesInspected, (datasource_id, participant_id)
            )
        )
        and cache_is_fresh(cached.response_last_updated)
    ):
        return cached.get_response()

    session.execute(
        delete(ParticipantTypesInspected).where(
            ParticipantTypesInspected.datasource_id == datasource_id,
            ParticipantTypesInspected.participant_type == participant_id,
        )
    )
    session.commit()

    def inspect_participant_types_impl() -> InspectParticipantTypesResponse:
        with dsconfig.dbsession() as dwh_session:
            sa_table = infer_table(
                dwh_session.get_bind(),
                pconfig.table_name,
                dsconfig.supports_reflection(),
            )
        db_schema = generate_field_descriptors(sa_table, pconfig.get_unique_id_field())
        mapper = create_col_to_filter_meta_mapper(db_schema, sa_table, dwh_session)

        filter_fields = {c.field_name: c for c in pconfig.fields if c.is_filter}
        strata_fields = {c.field_name: c for c in pconfig.fields if c.is_strata}
        metric_cols = {c.field_name: c for c in pconfig.fields if c.is_metric}

        return InspectParticipantTypesResponse(
            metrics=sorted(
                [
                    GetMetricsResponseElement(
                        data_type=db_schema.get(col_name).data_type,
                        field_name=col_name,
                        description=col_descriptor.description,
                    )
                    for col_name, col_descriptor in metric_cols.items()
                    if db_schema.get(col_name)
                ],
                key=lambda item: item.field_name,
            ),
            strata=sorted(
                [
                    GetStrataResponseElement(
                        data_type=db_schema.get(field_name).data_type,
                        field_name=field_name,
                        description=field_descriptor.description,
                        # For strata columns, we will echo back any extra annotations
                        extra=field_descriptor.extra,
                    )
                    for field_name, field_descriptor in strata_fields.items()
                    if db_schema.get(field_name)
                ],
                key=lambda item: item.field_name,
            ),
            filters=sorted(
                [
                    mapper(field_name, field_descriptor)
                    for field_name, field_descriptor in filter_fields.items()
                    if db_schema.get(field_name)
                ],
                key=lambda item: item.field_name,
            ),
        )

    response = inspect_participant_types_impl()

    session.add(
        ParticipantTypesInspected(
            datasource_id=datasource_id, participant_type=participant_id
        ).set_response(response)
    )
    session.commit()

    return response


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

    # Invalidate the participant types cached inspections because the configuration may have been updated and they may
    # be stale.
    session.execute(
        delete(ParticipantTypesInspected).where(
            ParticipantTypesInspected.datasource_id == datasource_id,
            ParticipantTypesInspected.participant_type == participant_id,
        )
    )
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


@router.get("/datasources/{datasource_id}/apikeys")
def list_api_keys(
    datasource_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
) -> ListApiKeysResponse:
    """Returns API keys that have access to the datasource."""
    ds = get_datasource_or_raise(session, user, datasource_id)
    return ListApiKeysResponse(
        items=[
            ApiKeySummary(
                id=api_key.id,
                datasource_id=api_key.datasource_id,
                organization_id=api_key.datasource.organization_id,
                organization_name=api_key.datasource.organization.name,
            )
            for api_key in sorted(ds.api_keys, key=lambda a: a.id)
        ]
    )


@router.post("/datasources/{datasource_id}/apikeys")
def create_api_key(
    datasource_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
) -> CreateApiKeyResponse:
    """Creates an API key for the specified datasource.

    The user must belong to the organization that owns the requested datasource.
    """
    ds = get_datasource_or_raise(session, user, datasource_id)
    label, key = make_key()
    key_hash = hash_key(key)
    api_key = ApiKey(id=label, key=key_hash, datasource_id=ds.id)
    session.add(api_key)
    session.commit()
    return CreateApiKeyResponse(id=label, datasource_id=ds.id, key=key)


@router.delete(
    "/datasources/{datasource_id}/apikeys/{api_key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_api_key(
    datasource_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
    api_key_id: Annotated[str, Path(...)],
):
    """Deletes the specified API key."""
    ds = get_datasource_or_raise(session, user, datasource_id)
    ds.api_keys = [a for a in ds.api_keys if a.id != api_key_id]
    session.add(ds)
    session.commit()
    return GENERIC_SUCCESS


@router.post("/datasources/{datasource_id}/experiments")
def create_experiment_with_assignment(
    datasource_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
    body: experiments_api_types.CreateExperimentRequest,
    chosen_n: Annotated[
        int, Query(..., description="Number of participants to assign.")
    ],
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
) -> experiments_api_types.CreateExperimentResponse:
    datasource = get_datasource_or_raise(session, user, datasource_id)
    if body.design_spec.ids_are_present():
        raise LateValidationError("Invalid DesignSpec: UUIDs must not be set.")
    ds_config = datasource.get_config()
    participants_cfg = ds_config.find_participants(body.audience_spec.participant_type)
    if not isinstance(participants_cfg, ParticipantsDef):
        raise LateValidationError(
            "Invalid ParticipantsConfig: Participants must be of type schema."
        )

    # Get participants and their schema info from the client dwh
    with ds_config.dbsession() as dwh_session:
        sa_table = infer_table(
            dwh_session.get_bind(),
            participants_cfg.table_name,
            ds_config.supports_reflection(),
        )
        participants = query_for_participants(
            dwh_session, sa_table, body.audience_spec, chosen_n
        )

    return experiments.create_experiment_with_assignment_impl(
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
def analyze_experiment(
    datasource_id: str,
    experiment_id: str,
    xngin_session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
    baseline_arm_id: Annotated[
        str | None,
        Query(
            description="UUID of the baseline arm. If None, the first design spec arm is used.",
        ),
    ] = None,
) -> ExperimentAnalysis:
    ds = get_datasource_or_raise(xngin_session, user, datasource_id)
    dsconfig = ds.get_config()

    experiment = get_experiment_via_ds_or_raise(xngin_session, ds, experiment_id)

    participants_cfg = dsconfig.find_participants(
        experiment.get_audience_spec().participant_type
    )
    if not isinstance(participants_cfg, ParticipantsDef):
        raise LateValidationError(
            "Invalid ParticipantsConfig: Participants must be of type schema."
        )
    unique_id_field = participants_cfg.get_unique_id_field()

    with dsconfig.dbsession() as dwh_session:
        sa_table = infer_table(
            dwh_session.get_bind(),
            participants_cfg.table_name,
            dsconfig.supports_reflection(),
        )

        design_spec = experiment.get_design_spec()
        metrics = design_spec.metrics
        assignments = experiment.arm_assignments
        participant_ids = [assignment.participant_id for assignment in assignments]
        participant_outcomes = get_participant_metrics(
            dwh_session,
            sa_table,
            metrics,
            unique_id_field,
            participant_ids,
        )

    # Always assume the first arm is the baseline; UI can override this.
    baseline_arm_id = baseline_arm_id or design_spec.arms[0].arm_id
    analyze_results = analyze_experiment_impl(
        assignments, participant_outcomes, baseline_arm_id
    )

    metric_analyses = []
    for metric in experiment.get_design_spec().metrics:
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
                )
            )
        metric_analyses.append(
            MetricAnalysis(
                metric_name=metric_name, metric=metric, arm_analyses=arm_analyses
            )
        )
    return ExperimentAnalysis(
        experiment_id=uuid.UUID(experiment.id), metric_analyses=metric_analyses
    )


@router.post(
    "/datasources/{datasource_id}/experiments/{experiment_id}/commit",
    status_code=status.HTTP_204_NO_CONTENT,
)
def commit_experiment(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
):
    ds = get_datasource_or_raise(session, user, datasource_id)
    experiment = get_experiment_via_ds_or_raise(session, ds, experiment_id)
    return experiments.commit_experiment_impl(session, experiment)


@router.post(
    "/datasources/{datasource_id}/experiments/{experiment_id}/abandon",
    status_code=status.HTTP_204_NO_CONTENT,
)
def abandon_experiment(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
):
    ds = get_datasource_or_raise(session, user, datasource_id)
    experiment = get_experiment_via_ds_or_raise(session, ds, experiment_id)
    return experiments.abandon_experiment_impl(session, experiment)


@router.get("/datasources/{datasource_id}/experiments")
def list_experiments(
    datasource_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
) -> experiments_api_types.ListExperimentsResponse:
    """Returns the list of experiments in the datasource."""
    ds = get_datasource_or_raise(session, user, datasource_id)
    return experiments.list_experiments_impl(session, ds.id)


@router.get("/datasources/{datasource_id}/experiments/{experiment_id}")
def get_experiment(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
) -> experiments_api_types.ExperimentConfig:
    """Returns the experiment with the specified ID."""
    ds = get_datasource_or_raise(session, user, datasource_id)
    experiment = get_experiment_via_ds_or_raise(session, ds, experiment_id)
    return ExperimentConfig(
        datasource_id=experiment.datasource_id,
        state=experiment.state,
        design_spec=experiment.get_design_spec(),
        audience_spec=experiment.get_audience_spec(),
        power_analyses=experiment.get_power_analyses(),
        assign_summary=experiments.get_assign_summary(experiment),
    )


@router.get("/datasources/{datasource_id}/experiments/{experiment_id}/assignments")
def get_experiment_assignments(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
) -> experiments_api_types.GetExperimentAssignmentsResponse:
    ds = get_datasource_or_raise(session, user, datasource_id)
    experiment = get_experiment_via_ds_or_raise(session, ds, experiment_id)
    return experiments.get_experiment_assignments_impl(experiment)


@router.get(
    "/datasources/{datasource_id}/experiments/{experiment_id}/assignments/csv",
    summary=(
        "Export experiment assignments as CSV file; BalanceCheck not included. "
        "csv header form: participant_id,arm_id,arm_name,strata_name1,strata_name2,..."
    ),
)
def get_experiment_assignments_as_csv(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
) -> StreamingResponse:
    ds = get_datasource_or_raise(session, user, datasource_id)
    experiment = get_experiment_via_ds_or_raise(session, ds, experiment_id)
    return experiments.get_experiment_assignments_as_csv_impl(experiment)


@router.get(
    "/datasources/{datasource_id}/experiments/{experiment_id}/assignments/{participant_id}",
    description="""Get the assignment for a specific participant, excluding strata if any.
    For 'preassigned' experiments, the participant's Assignment is returned if it exists.
    For 'online', returns the assignment if it exists, else generates an assignment.""",
)
def get_experiment_assignment_for_participant(
    datasource_id: str,
    experiment_id: str,
    participant_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
    random_state: Annotated[
        int | None,
        Query(
            description="Specify a random seed for reproducibility.",
            include_in_schema=False,
        ),
    ] = None,
) -> experiments_api_types.GetParticipantAssignmentResponse:
    """Get the assignment for a specific participant in an experiment."""
    # Validate the datasource and experiment exist
    ds = get_datasource_or_raise(session, user, datasource_id)

    # Look up the participant's assignment if it exists
    assignment = experiments.get_existing_assignment_for_participant(
        session, experiment_id, participant_id
    )
    if not assignment:
        experiment = get_experiment_via_ds_or_raise(session, ds, experiment_id)
        assignment = experiments.create_assignment_for_participant(
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
def delete_experiment(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
):
    """Deletes the experiment with the specified ID."""
    ds = get_datasource_or_raise(session, user, datasource_id)
    experiment = get_experiment_via_ds_or_raise(session, ds, experiment_id)
    session.delete(experiment)
    session.commit()
    return GENERIC_SUCCESS


@router.post("/datasources/{datasource_id}/power")
def power_check(
    datasource_id: str,
    session: Annotated[Session, Depends(xngin_db_session)],
    user: Annotated[User, Depends(user_from_token)],
    body: PowerRequest,
) -> PowerResponse:
    ds = get_datasource_or_raise(session, user, datasource_id)
    dsconfig = ds.get_config()
    participants_cfg = dsconfig.find_participants(body.audience_spec.participant_type)
    validate_schema_metrics_or_raise(body.design_spec, participants_cfg)
    return power_check_impl(body, dsconfig, participants_cfg)
