"""
This module defines the internal Evidential UI-facing Admin API endpoints
for configuring specific third-party integrations.

It currently includes endpoints supporting the Turn.io integration,
which allow admins to set up and manage the connection to Turn.io and configure
the mapping from experiment arms to Turn.io journeys.

(See integrations_api.py for endpoints that specific third-party tools can hit.)
"""

import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx2
from fastapi import (
    APIRouter,
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    status,
)
from loguru import logger
from pydantic import ValidationError
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import constants
from xngin.apiserver.dependencies import retrying_httpx_dependency, xngin_db_session
from xngin.apiserver.routers.admin import admin_common, authz
from xngin.apiserver.routers.admin.admin_api import (
    GENERIC_SUCCESS,
    STANDARD_ADMIN_RESPONSES,
    HTTPExceptionError,
    get_datasource_or_raise,
    get_experiment_via_ds_or_raise,
    get_organization_or_raise,
)
from xngin.apiserver.routers.admin.admin_api_types import (
    AddTurnJourneysChangedWebhookRequest,
    AddWebhookToOrganizationResponse,
)
from xngin.apiserver.routers.admin.generic_handlers import handle_delete
from xngin.apiserver.routers.admin_integrations.admin_integrations_api_types import (
    GetTurnArmJourneyMappingResponse,
    GetTurnConnectionResponse,
    GetTurnJourneysResponse,
    Journey,
    SetConnectionToTurnRequest,
    SetTurnArmJourneyMappingRequest,
)
from xngin.apiserver.routers.auth.auth_dependencies import require_user_from_token
from xngin.apiserver.routers.common_api_types import SampleCalls
from xngin.apiserver.routers.experiments.experiments_common import make_sample_calls
from xngin.apiserver.sqla import tables

TURN_JOURNEYS_URL = "https://whatsapp.turn.io/v1/stacks"

TURN_JOURNEYS_RESPONSES: dict[str | int, dict[str, Any]] = {
    "404": {
        "model": HTTPExceptionError,
        "description": "No Turn.io connection has been configured for the organization.",
    },
    "502": {
        "model": HTTPExceptionError,
        "description": "Failed to reach Turn.io API, or Turn.io returned a non-200 response.",
    },
    "422": {
        "model": HTTPExceptionError,
        "description": "The retrieved journeys from Turn.io did not have the expected fields 'name' and 'uuid'.",
    },
}

TURN_ARM_JOURNEY_MAPPING_RESPONSES: dict[str | int, dict[str, Any]] = {
    "400": {
        "model": HTTPExceptionError,
        "description": "Malformed request body, or request arm IDs don't match the experiment's arms.",
    },
    "409": {
        "model": HTTPExceptionError,
        "description": "A Turn.io connection must be configured before journey mappings can be saved.",
    },
}


async def _call_turn_api(
    httpx_client: httpx2.AsyncClient,
    turn_api_token: str,
    method: str,
) -> list[Journey]:
    """
    Wrapper for outbound Turn.io API calls to standardize error handling.

    Any non-2xx response from Turn.io and httpx2.RequestErrors (e.g. network issues, timeouts) are logged
    and re-raised as 502 HTTP exceptions, but with appropriate status codes and error messages
    reproduced for debugging.
    """
    headers = {"Authorization": f"Bearer {turn_api_token}"}

    try:
        response = await httpx_client.request(
            method,
            TURN_JOURNEYS_URL,
            headers=headers,
        )
        response.raise_for_status()
        journeys = [Journey.model_validate(journey) for journey in response.json()]

    except httpx2.RequestError as exc:
        logger.error(f"Error calling Turn.io API at {method} {TURN_JOURNEYS_URL}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to reach Turn.io API. Details: {exc}",
        ) from exc
    except httpx2.HTTPStatusError as exc:
        logger.error(
            f"Turn.io API returned non-2xx status at {method} {TURN_JOURNEYS_URL}:"
            + f"{exc.response.status_code} - {exc.response.text}"
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Turn.io API returned non-2xx status. Details: {exc.response.status_code} - {exc.response.text}",
        ) from exc
    except ValidationError as exc:
        logger.error(f"Turn.io API returned unexpected response structure at {method} {TURN_JOURNEYS_URL}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"The retrieved journeys from Turn.io did not have the expected fields 'name' and 'uuid'. "
            f"Details: {exc}",
        ) from exc
    return journeys


async def refresh_journeys_dict(turn_api_token: str, httpx_client: httpx2.AsyncClient) -> list[Journey]:
    """Refreshes the cached Turn.io journeys on the TurnConnection.

    Returns the updated journey list.
    """
    return await _call_turn_api(httpx_client=httpx_client, turn_api_token=turn_api_token, method="GET")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting router: {__name__} (prefix={router.prefix})")
    yield


router = APIRouter(
    lifespan=lifespan,
    prefix=constants.API_PREFIX_V1 + "/m",
    responses=STANDARD_ADMIN_RESPONSES,
    dependencies=[Depends(require_user_from_token)],  # All routes in this router require authentication.
)


async def get_turn_webhook_or_raise(
    session: AsyncSession,
    allow_missing: bool = False,
    organization_id: str | None = None,
    webhook_id: str | None = None,
) -> tables.Webhook | None:
    """Returns the Turn.io webhook for the organization, if it exists."""
    if organization_id is None and webhook_id is None:
        raise ValueError("Either organization_id or webhook_id must be provided.")

    stmt = select(tables.Webhook).where(
        tables.Webhook.type == "turn.journeys_changed",
    )
    if organization_id is not None:
        stmt = stmt.where(tables.Webhook.organization_id == organization_id)
    if webhook_id is not None:
        stmt = stmt.where(tables.Webhook.id == webhook_id)

    webhook = (await session.execute(stmt)).scalar_one_or_none()

    if not webhook and not allow_missing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Turn.io webhook not found")

    return webhook


@router.put(
    "/integrations/turn-connection/{organization_id}",
    responses=TURN_JOURNEYS_RESPONSES,
)
async def set_organization_turn_connection(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    body: Annotated[SetConnectionToTurnRequest, Body(...)],
    httpx_client: Annotated[httpx2.AsyncClient, Depends(retrying_httpx_dependency)],
) -> AddWebhookToOrganizationResponse:
    """Sets (or rotates) the Turn.io API token for an organization.

    Creates a Turn connection for the organization if one does not yet exist, otherwise
    overwrites the existing token. An organization has at most one Turn connection.

    Also ensures a "turn.journeys_changed" webhook exists for the organization, creating one
    if it does not already exist.

    The stored list of Turn.io journeys for the organization is also refreshed when the token is set or updated.
    A separate webhook endpoint for refreshing journeys ("/integrations/turn/webhook/{webhook_id}/config-updated")
    can be used to notify Evidential of updates to the Journeys from the Turn.io Evidential App, or
    "/integrations/turn/webhook/{webhook_id}/refresh-journeys" can be used to manually trigger a refresh without
    rotating the token.

    NB: It's important to maintain the order of setting token -> refreshing journeys in a single
    transaction, since the validity of the journeys list depends on the token. This ensures that
    we don't end up in a state where the token is updated but the journeys list is out of sync
    with the new token.
    """
    org = await get_organization_or_raise(session, user, organization_id)

    turn_connection = (
        await session.execute(select(tables.TurnConnection).where(tables.TurnConnection.organization_id == org.id))
    ).scalar_one_or_none()

    if turn_connection is None:
        turn_connection = tables.TurnConnection(organization_id=org.id)
        turn_connection.set_turn_api_token(body.turn_api_token)
        session.add(turn_connection)

    turn_webhook = await get_turn_webhook_or_raise(session, organization_id=org.id, allow_missing=True)

    token_changed = turn_connection.get_turn_api_token() != body.turn_api_token
    should_refresh_journeys = turn_webhook is None or token_changed

    if turn_webhook is None:
        _, turn_webhook = admin_common.create_webhook_impl(
            session=session,
            org_id=org.id,
            webhook=AddTurnJourneysChangedWebhookRequest(
                name="Turn Journeys Changed Webhook",
            ),
        )

    elif token_changed:
        # This could make the arm_journey_ids stored in ExperimentTurnConfig.arm_journey_map invalid.
        # We expose that drift to the UI via get_..mapping()'s reporting of stale journey IDs.
        turn_connection.set_turn_api_token(body.turn_api_token)

    else:
        logger.info(
            f"Turn.io API token provided in request is the same as the existing token for organization {org.id}. "
            f"Not updating token."
        )

    if should_refresh_journeys:
        journeys = await refresh_journeys_dict(turn_api_token=body.turn_api_token, httpx_client=httpx_client)
        turn_connection.journeys_dict = {journey.name: journey.uuid for journey in journeys}

    await session.commit()

    return AddWebhookToOrganizationResponse(
        id=turn_webhook.id,
        direction=turn_webhook.direction,
        type=turn_webhook.type,
        name=turn_webhook.name,
        url=turn_webhook.url,
        auth_token=turn_webhook.auth_token,
    )


@router.get("/integrations/turn-connection/{organization_id}")
async def get_organization_turn_connection(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    allow_missing: Annotated[
        bool, Query(description="If true, return a 200 with default/empty field values if the resource does not exist.")
    ] = False,
) -> GetTurnConnectionResponse:
    """Returns a preview of the organization's configured Turn.io API token.

    Raises 404 if no Turn connection has been configured for the organization (or if the
    organization does not exist / the user does not have access to it).
    """
    org = await get_organization_or_raise(session, user, organization_id)

    turn_connection = (
        await session.execute(select(tables.TurnConnection).where(tables.TurnConnection.organization_id == org.id))
    ).scalar_one_or_none()
    if turn_connection is None and not allow_missing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Turn connection not found")

    webhook = await get_turn_webhook_or_raise(session, organization_id=org.id, allow_missing=allow_missing)

    return GetTurnConnectionResponse(
        turn_api_token_preview=turn_connection.turn_api_token_preview if turn_connection else "",
        id=webhook.id if webhook else "",
        type=webhook.type if webhook else "turn.journeys_changed",
        direction=webhook.direction if webhook else "inbound",
        name=webhook.name if webhook else "",
        auth_token=webhook.auth_token if webhook else "",
    )


@router.put(
    "/integrations/turn-connection/{organization_id}/regenerate-webhook-token", responses=TURN_JOURNEYS_RESPONSES
)
async def regenerate_turn_webhook_token(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    allow_missing: Annotated[
        bool, Query(description="If true, return a 200 with default/empty field values if the resource does not exist.")
    ] = False,
) -> AddWebhookToOrganizationResponse:
    """Regenerates the auth token for the organization's turn.journeys_changed webhook.

    This is used when the user wants to change their current token. It does not change
    the Turn.io API token or the stored journeys in any way.

    Raises 404 if no Turn connection or webhook has been configured for the organization (or if the
    organization does not exist / the user does not have access to it). If allow_missing is True,
    returns a 200 response with all string fields set to empty string and url set to None instead.
    """
    org = await get_organization_or_raise(session, user, organization_id)

    turn_webhook = await get_turn_webhook_or_raise(session, organization_id=org.id, allow_missing=allow_missing)

    if turn_webhook is not None:
        turn_webhook.auth_token = secrets.token_hex(16)
        await session.commit()
    return AddWebhookToOrganizationResponse(
        id=turn_webhook.id if turn_webhook else "",
        direction=turn_webhook.direction if turn_webhook else "inbound",
        type=turn_webhook.type if turn_webhook else "turn.journeys_changed",
        name=turn_webhook.name if turn_webhook else "",
        url=turn_webhook.url if turn_webhook else None,
        auth_token=turn_webhook.auth_token if turn_webhook else "",
    )


@router.delete(
    "/integrations/turn-connection/{organization_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_turn_connection_from_organization(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    allow_missing: Annotated[
        bool,
        Query(description="If true, return a 204 even if the resource does not exist."),
    ] = False,
):
    """Removes an organization's Turn.io connection."""

    async def _delete_turn_connection_and_mapping(session: AsyncSession, turn_connection: tables.TurnConnection):
        await session.delete(turn_connection)
        # Cascade delete all Turn journey mappings for experiments under this organization,
        # since they become invalid without a Turn connection.
        await session.execute(
            delete(tables.ExperimentTurnConfig).where(
                tables.ExperimentTurnConfig.experiment_id.in_(
                    select(tables.Experiment.id)
                    .join(tables.Datasource, tables.Experiment.datasource_id == tables.Datasource.id)
                    .where(tables.Datasource.organization_id == organization_id)
                )
            )
        )
        # Cascade delete the Turn journeys changed webhook for this organization,
        # since it becomes invalid without a Turn connection.
        await session.execute(
            delete(tables.Webhook).where(
                tables.Webhook.organization_id == organization_id, tables.Webhook.type == "turn.journeys_changed"
            )
        )

    resource_query = select(tables.TurnConnection).where(tables.TurnConnection.organization_id == organization_id)
    response = await handle_delete(
        session,
        allow_missing,
        authz.is_user_authorized_on_organization(user, organization_id),
        resource_query,
        deleter=_delete_turn_connection_and_mapping,
    )

    await session.commit()
    return response


@router.get("/integrations/turn-connection/{organization_id}/journeys", responses=TURN_JOURNEYS_RESPONSES)
async def get_organization_turn_journeys(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
) -> GetTurnJourneysResponse:
    """Returns a {name: uuid} map of Turn.io journeys available to the organization.

    The stored Journey list is returned to avoid repeatedly calling the Turn.io API.
    """
    org = await get_organization_or_raise(session, user, organization_id)

    turn_connection = (
        await session.execute(select(tables.TurnConnection).where(tables.TurnConnection.organization_id == org.id))
    ).scalar_one_or_none()
    if turn_connection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Turn connection not found")

    if turn_connection.journeys_dict is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No stored journeys found for Turn connection. "
            "Re-set the Turn.io API token to fetch Journeys from Turn.io and store them.",
        )

    journeys = sorted(
        [Journey(name=name, uuid=uuid) for name, uuid in turn_connection.journeys_dict.items()], key=lambda j: j.uuid
    )

    return GetTurnJourneysResponse(journeys=journeys)


@router.put(
    "/integrations/turn-journey-mapping/datasources/{datasource_id}/experiments/{experiment_id}",
    responses=TURN_ARM_JOURNEY_MAPPING_RESPONSES,
    status_code=status.HTTP_204_NO_CONTENT,
)
async def set_turn_arm_journey_mapping(
    datasource_id: str,
    experiment_id: str,
    body: SetTurnArmJourneyMappingRequest,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
):
    """Adds or updates the mapping from each arm ID of the experiment to a Turn.io Journey ID."""
    ds = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(session, ds, experiment_id)
    turn_connection = (
        await session.execute(
            select(tables.TurnConnection).where(tables.TurnConnection.organization_id == ds.organization_id)
        )
    ).scalar_one_or_none()
    if not turn_connection:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "No Turn.io connection configured for organization. "
                "Please set up a Turn.io connection before configuring turn/arm to journey mappings."
            ),
        )

    # Validate Arm IDs in the request body are part of the experiment
    arm_ids = {arm.id for arm in experiment.arms}
    input_arm_ids = set(body.arm_to_journeys.keys())
    missing_arm_ids = arm_ids - input_arm_ids
    extra_arm_ids = input_arm_ids - arm_ids
    if missing_arm_ids or extra_arm_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error in arm IDs config. Missing: {missing_arm_ids}, Extra: {extra_arm_ids}",
        )

    turn_config = (
        await session.execute(
            select(tables.ExperimentTurnConfig).where(
                tables.ExperimentTurnConfig.experiment_id == experiment_id,
            )
        )
    ).scalar_one_or_none()

    if turn_config is None:
        turn_config = tables.ExperimentTurnConfig(
            experiment_id=experiment_id,
            arm_journey_map=body.arm_to_journeys,
        )
        session.add(turn_config)
    else:
        turn_config.arm_journey_map = body.arm_to_journeys

    await session.commit()
    return GENERIC_SUCCESS


@router.get("/integrations/turn-journey-mapping/datasources/{datasource_id}/experiments/{experiment_id}")
async def get_turn_arm_journey_mapping(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
) -> GetTurnArmJourneyMappingResponse:
    """Returns the current mapping from each arm ID of the experiment to a Turn.io Journey ID, if it exists."""
    ds = await get_datasource_or_raise(session, user, datasource_id)
    await get_experiment_via_ds_or_raise(session, ds, experiment_id)
    turn_config = (
        await session.execute(
            select(tables.ExperimentTurnConfig).where(
                tables.ExperimentTurnConfig.experiment_id == experiment_id,
            )
        )
    ).scalar_one_or_none()

    if turn_config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No Turn.io journey mapping found for experiment.",
        )

    turn_journeys_row = await session.execute(
        select(tables.TurnConnection.journeys_dict).where(
            tables.TurnConnection.organization_id == ds.organization_id,
        )
    )
    turn_journeys = turn_journeys_row.scalar_one_or_none()
    uuids = list(turn_journeys.values()) if turn_journeys is not None else []

    stale_arm_ids = []
    if turn_journeys is not None:
        stale_arm_ids = [
            arm_id for arm_id, journey_uuid in turn_config.arm_journey_map.items() if journey_uuid not in uuids
        ]

    return GetTurnArmJourneyMappingResponse(arm_to_journeys=turn_config.arm_journey_map, stale_arm_ids=stale_arm_ids)


@router.delete(
    "/integrations/turn-journey-mapping/datasources/{datasource_id}/experiments/{experiment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_turn_arm_journey_mapping(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    allow_missing: Annotated[
        bool,
        Query(description="If true, return a 204 even if the resource does not exist."),
    ] = False,
):
    """Deletes the mapping from each arm ID of the experiment to a Turn.io Journey ID, if it exists."""
    ds = await get_datasource_or_raise(session, user, datasource_id)
    await get_experiment_via_ds_or_raise(session, ds, experiment_id)
    turn_config_query = select(tables.ExperimentTurnConfig).where(
        tables.ExperimentTurnConfig.experiment_id == experiment_id
    )

    response = await handle_delete(
        session,
        allow_missing,
        authz.is_user_authorized_on_datasource(user, datasource_id),
        turn_config_query,
    )
    await session.commit()
    return response


@router.get("/integrations/datasources/{datasource_id}/experiments/{experiment_id}/sample-calls")
async def get_experiment_sample_calls(
    datasource_id: str,
    experiment_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
) -> SampleCalls | None:
    """Return example API calls for integrating with this experiment.

    Integration/onboarding documentation shown in the integration guide: a get-assignment example
    for every type, plus a report-outcome example for bandits and an assign-with-filters example for
    freq_online experiments that have eligibility filters. Returns null for experiment types without
    a meaningful example (currently CMAB, whose assignment needs a context vector).
    """
    ds = await get_datasource_or_raise(session, user, datasource_id)
    experiment = await get_experiment_via_ds_or_raise(
        session,
        ds,
        experiment_id,
        preload=[tables.Experiment.experiment_fields, tables.Experiment.experiment_filters],
    )
    return make_sample_calls(experiment)
