"""
This module defines the public API for clients to integrate with specific third-party tools.
Currently, we only support Turn.io, so this includes endpoints for the Turn.io Evidential App config

(See admin_integrations_api.py for Evidential UI-facing integration endpoints.)
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import constants
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.routers.admin.admin_api import GENERIC_SUCCESS
from xngin.apiserver.routers.common_api_types import TurnConfigResponse
from xngin.apiserver.routers.experiments.dependencies import experiment_dependency
from xngin.apiserver.routers.experiments.experiments_api import STANDARD_INTEGRATION_RESPONSES
from xngin.apiserver.sqla import tables
from xngin.tq.task_payload_types import TURN_JOURNEYS_CHANGED_TASK_TYPE


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

    turn_config = (
        await session.execute(
            select(tables.ExperimentTurnConfig).where(tables.ExperimentTurnConfig.experiment_id == experiment.id)
        )
    ).scalar_one_or_none()

    if not turn_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No Turn.io mapping configured for this experiment."
        )

    return TurnConfigResponse(
        experiment_id=experiment.id,
        experiment_name=experiment.name,
        arm_journey_map=turn_config.arm_journey_map,
    )


@router.post(
    "/integrations/turn/webhook/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        "401": {
            "model": dict,
            "description": "The provided Webhook-Token does not match the token configured for this webhook.",
        },
        "404": {
            "model": dict,
            "description": "The specified webhook was not found.",
        },
    },
)
async def turn_webhook(
    webhook_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    webhook_token: Annotated[str | None, Header(alias=constants.HEADER_WEBHOOK_TOKEN)] = None,
):
    """
    This endpoint is used as the webhook URL for Turn.io to notify us of changes to the Journeys.
    It is not intended to be called directly by clients, and will return a 400 error if called without
    a valid Turn.io webhook auth token.
    """
    webhook = await session.get(tables.Webhook, webhook_id)

    if not webhook:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found.")

    if webhook_token != webhook.auth_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook auth token.")

    # Process the webhook payload here
    session.add(
        tables.Task(
            task_type=TURN_JOURNEYS_CHANGED_TASK_TYPE,
            payload={"organization_id": webhook.organization_id},
        )
    )
    await session.commit()
    return GENERIC_SUCCESS
