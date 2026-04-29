"""
This module defines the internal Evidential UI-facing Admin API endpoints
for configuring specific third-party integrations.

It currently includes endpoints supporting the Turn.io integration,
which allow admins to set up and manage the connection to Turn.io and configure
the mapping from experiment arms to Turn.io journeys.

(See integrations_api.py for endpoints that specific third-party tools can hit.)
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Annotated

import httpx
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
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import constants
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.limits import TURN_JOURNEYS_CACHE_TTL_SECONDS, TURN_REQUEST_TIMEOUT_SECONDS
from xngin.apiserver.routers.admin import authz
from xngin.apiserver.routers.admin.admin_api import (
    GENERIC_SUCCESS,
    STANDARD_ADMIN_RESPONSES,
    get_datasource_or_raise,
    get_experiment_via_ds_or_raise,
    get_organization_or_raise,
)
from xngin.apiserver.routers.admin.generic_handlers import handle_delete
from xngin.apiserver.routers.admin_integrations.admin_integrations_api_types import (
    GetTurnArmJourneyMappingResponse,
    GetTurnConnectionResponse,
    GetTurnJourneysResponse,
    SetConnectionToTurnRequest,
    SetTurnArmJourneyMappingRequest,
)
from xngin.apiserver.routers.auth.auth_dependencies import require_user_from_token
from xngin.apiserver.sqla import tables

TURN_JOURNEYS_URL = "https://whatsapp.turn.io/v1/stacks"


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


@router.put(
    "/integrations/turn-connection/{organization_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def set_organization_turn_connection(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    body: Annotated[SetConnectionToTurnRequest, Body(...)],
):
    """Sets (or rotates) the Turn.io API token for an organization.

    Creates a Turn connection for the organization if one does not yet exist, otherwise
    overwrites the existing token. An organization has at most one Turn connection.
    """
    org = await get_organization_or_raise(session, user, organization_id)

    turn_connection = (
        await session.execute(select(tables.TurnConnection).where(tables.TurnConnection.organization_id == org.id))
    ).scalar_one_or_none()
    if turn_connection is None:
        turn_connection = tables.TurnConnection(organization_id=org.id)
        turn_connection.set_turn_api_token(body.turn_api_token)
        session.add(turn_connection)
        await session.commit()

    elif turn_connection.get_turn_api_token() != body.turn_api_token:
        turn_connection.set_turn_api_token(body.turn_api_token)

        # Journey UUIDs belong to a specific Turn workspace, so any stored arm->journey
        # mappings become stale the moment the token changes. Wipe them for every
        # experiment under this organization's datasources.
        await session.execute(
            delete(tables.ExperimentTurnConfig).where(
                tables.ExperimentTurnConfig.experiment_id.in_(
                    select(tables.Experiment.id)
                    .join(tables.Datasource, tables.Experiment.datasource_id == tables.Datasource.id)
                    .where(tables.Datasource.organization_id == org.id)
                )
            )
        )

        await session.commit()
    else:
        logger.debug(f"Turn.io API token for organization {org.id} is unchanged. No update performed.")
    return GENERIC_SUCCESS


@router.get("/integrations/turn-connection/{organization_id}")
async def get_organization_turn_connection(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
    allow_missing: Annotated[
        bool, Query(description="If true, return a 200 with null body if the resource does not exist.")
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

    return GetTurnConnectionResponse(token_preview=turn_connection.turn_api_token_preview if turn_connection else "")


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
    resource_query = select(tables.TurnConnection).where(tables.TurnConnection.organization_id == organization_id)
    response = await handle_delete(
        session,
        allow_missing,
        authz.is_user_authorized_on_organization(user, organization_id),
        resource_query,
    )

    # Cascade delete all Turn journey mappings for experiments under this organization,
    # since they become invalid without a Turn connection.
    mapping_query = select(tables.ExperimentTurnConfig).where(
        tables.ExperimentTurnConfig.experiment_id.in_(
            select(tables.Experiment.id)
            .join(tables.Datasource, tables.Experiment.datasource_id == tables.Datasource.id)
            .where(tables.Datasource.organization_id == organization_id)
        )
    )
    await handle_delete(
        session,
        True,
        authz.is_user_authorized_on_organization(user, organization_id),
        mapping_query,
    )
    await session.commit()
    return response


@router.get("/integrations/turn-connection/{organization_id}/journeys")
async def get_organization_turn_journeys(
    organization_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
) -> GetTurnJourneysResponse:
    """Returns a {name: uuid} map of Turn.io journeys available to the organization.

    The result is cached on the TurnConnection row and refreshed when older than
    TURN_JOURNEYS_CACHE_TTL_SECONDS, or when the API token is rotated.
    """
    org = await get_organization_or_raise(session, user, organization_id)

    turn_connection = (
        await session.execute(select(tables.TurnConnection).where(tables.TurnConnection.organization_id == org.id))
    ).scalar_one_or_none()
    if turn_connection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Turn connection not found")

    now = datetime.now(UTC)
    if (
        turn_connection.cached_journeys is not None
        and turn_connection.cached_journeys_updated_at is not None
        and now - turn_connection.cached_journeys_updated_at < timedelta(seconds=TURN_JOURNEYS_CACHE_TTL_SECONDS)
    ):
        return GetTurnJourneysResponse(journeys=turn_connection.cached_journeys)

    try:
        async with httpx.AsyncClient(timeout=TURN_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(
                TURN_JOURNEYS_URL,
                headers={"Authorization": f"Bearer {turn_connection.get_turn_api_token()}"},
            )
    except httpx.RequestError as exc:
        logger.error(f"Error fetching Turn.io journeys: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to reach Turn.io API to fetch journeys. Details: {exc}",
        ) from exc

    if response.status_code != 200:
        logger.error(f"Non-200 response from Turn.io journeys endpoint: {response.status_code} - {response.text}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Turn.io API returned non-200 status code when fetching journeys. "
                f"Details: {response.status_code} - {response.text}"
            ),
        )

    journey_dict = {journey["name"]: journey["uuid"] for journey in response.json()}
    turn_connection.cached_journeys = journey_dict
    turn_connection.cached_journeys_updated_at = now
    await session.commit()
    return GetTurnJourneysResponse(journeys=journey_dict)


@router.put(
    "/integrations/turn-journey-mapping/datasources/{datasource_id}/experiments/{experiment_id}",
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

    mapping = (
        await session.execute(
            select(tables.ExperimentTurnConfig).where(
                tables.ExperimentTurnConfig.experiment_id == experiment_id,
            )
        )
    ).scalar_one_or_none()

    if mapping is None:
        mapping = tables.ExperimentTurnConfig(
            experiment_id=experiment_id,
            arm_journey_map=body.arm_to_journeys,
        )
        session.add(mapping)
    else:
        mapping.arm_journey_map = body.arm_to_journeys

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
    mapping = (
        await session.execute(
            select(tables.ExperimentTurnConfig).where(
                tables.ExperimentTurnConfig.experiment_id == experiment_id,
            )
        )
    ).scalar_one_or_none()

    if mapping is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No Turn.io journey mapping found for experiment.",
        )

    return GetTurnArmJourneyMappingResponse(arm_to_journeys=mapping.arm_journey_map)


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
    mapping_query = select(tables.ExperimentTurnConfig).where(
        tables.ExperimentTurnConfig.experiment_id == experiment_id
    )

    response = await handle_delete(
        session,
        allow_missing,
        authz.is_user_authorized_on_datasource(user, datasource_id),
        mapping_query,
    )
    await session.commit()
    return response
