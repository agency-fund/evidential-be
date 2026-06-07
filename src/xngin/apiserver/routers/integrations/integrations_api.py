"""
This module defines the public API for clients to integrate with specific third-party tools.
Currently, we only support Turn.io, so this includes endpoints for the Turn.io Evidential App config

(See admin_integrations_api.py for Evidential UI-facing integration endpoints.)
"""

from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import constants
from xngin.apiserver.dependencies import retrying_httpx_dependency
from xngin.apiserver.routers.admin_integrations.admin_integrations_api import refresh_journeys_dict
from xngin.apiserver.routers.common_api_types import TurnConfigResponse, TurnJourneysWebhookRequest
from xngin.apiserver.routers.experiments.dependencies import (
    datasource_dependency,
    experiment_dependency,
    xngin_db_session,
)
from xngin.apiserver.routers.experiments.experiments_api import STANDARD_INTEGRATION_RESPONSES
from xngin.apiserver.settings import (
    Datasource,
)
from xngin.apiserver.sqla import tables


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting router: {__name__} (prefix={router.prefix})")
    yield


router = APIRouter(
    lifespan=lifespan,
    prefix=constants.API_PREFIX_V1,
    responses=STANDARD_INTEGRATION_RESPONSES,
    strict_content_type=False,  # for backwards compatibility, do not require content-type: request headers.
)


@router.get(
    "/integrations/experiments/{experiment_id}/turn-app-config",
    summary="Get the Turn.io arm to journey mapping configuration for the experiment, if it exists.",
    description="""
    Returns the Turn.io journey ID mapped to each arm of the experiment, if a mapping exists. This mapping is used by
    the Turn.io integration to determine which Journey to assign a participant to based on their experiment arm.

    If the experiment has no Turn.io mapping configured, a 404 error is returned.
    To configure the Turn.io mapping for an experiment, please use the admin API endpoint
    `PUT /integrations/turn-journey-mapping/datasources/{datasource_id}/experiments/{experiment_id}`
""",
    response_model=TurnConfigResponse,
)
async def get_turn_app_config(
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    experiment: Annotated[tables.Experiment, Depends(experiment_dependency)],
) -> TurnConfigResponse:
    """Returns the current mapping from each arm ID of the experiment to a Turn.io Journey ID, if it exists."""

    mapping = (
        await session.execute(
            select(tables.ExperimentTurnConfig).where(tables.ExperimentTurnConfig.experiment_id == experiment.id)
        )
    ).scalar_one_or_none()

    if not mapping:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No Turn.io mapping configured for this experiment."
        )

    return TurnConfigResponse(
        experiment_id=experiment.id,
        experiment_name=experiment.name,
        arm_journey_map=mapping.arm_journey_map,
    )


@router.post(
    "/integrations/turn-journeys-webhook",
    summary="Webhook endpoint to receive updates about Turn.io journeys.",
    description="Turn-App webhook: notifies Evidential when the journey list changes.",
    response_model=dict,
)
async def turn_journeys_webhook(
    request: TurnJourneysWebhookRequest,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    httpx_client: Annotated[httpx.AsyncClient, Depends(retrying_httpx_dependency)],
    background_tasks: BackgroundTasks,
    datasource: Annotated[Datasource, Depends(datasource_dependency)],
) -> dict:
    """
    Endpoint to receive webhook calls from the Turn.io app when the list of journeys changes. This allows us to
    proactively update the stored list of journeys for each organization.

    The request body should include a SHA-256 hex digest of the current list of journeys in Turn.io,
    which we can compare to the digest of the stored journeys to determine if there has been a change.
    If there is a change, we refresh the stored journeys and their digest.

    Note: this webhook is intended to be called by the Turn.io app whenever there is a change to the list of journeys,
    but it can also be called manually with a valid journeys digest to trigger an update if needed
    """
    turn_connection = (
        await session.execute(
            select(tables.TurnConnection).where(tables.TurnConnection.organization_id == datasource.organization_id)
        )
    ).scalar_one_or_none()

    if not turn_connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No Turn.io connection found for organization {datasource.organization_id}.",
        )

    if request.journeys_uuid_digest != turn_connection.journeys_uuid_digest:
        logger.info(
            f"Received Turn.io journeys webhook for organization {datasource.organization_id} "
            f"with a different journeys digest than the stored digest. "
            f"Refreshing stored journeys and digest."
        )
        # Turn.io has a 5s timeout for webhooks, so we refresh the journeys in the background
        # after responding to the webhook to avoid timing out the request.
        background_tasks.add_task(refresh_journeys_dict, session, turn_connection, httpx_client)
    else:
        logger.info(
            f"Received Turn.io journeys webhook for organization {datasource.organization_id} "
            f"with the same journeys digest as the stored digest. "
            f"No update needed."
        )
    return {"message": "Turn.io journeys webhook received successfully."}
