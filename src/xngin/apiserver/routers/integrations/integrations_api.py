"""
This module defines the public API for clients to integrate with specific third-party tools.
Currently, we only support Turn.io, so this includes endpoints for the Turn.io Evidential App config

(See admin_integrations_api.py for Evidential UI-facing integration endpoints.)
"""

from contextlib import asynccontextmanager
from typing import Annotated

import httpx2
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import constants
from xngin.apiserver.dependencies import retrying_httpx_dependency, xngin_db_session
from xngin.apiserver.routers.admin.admin_api import GENERIC_SUCCESS
from xngin.apiserver.routers.admin_integrations.admin_integrations_api import (
    get_turn_webhook_or_raise,
    refresh_journeys_dict,
)
from xngin.apiserver.routers.common_api_types import TurnConfigResponse
from xngin.apiserver.routers.experiments.dependencies import experiment_dependency
from xngin.apiserver.routers.experiments.experiments_api import STANDARD_INTEGRATION_RESPONSES
from xngin.apiserver.sqla import tables
from xngin.tq.task_payload_types import TURN_JOURNEYS_CHANGED_TASK_TYPE, TurnJourneysChangedTask


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


def check_webhook_auth_token(auth_token: str | None, webhook: tables.Webhook | None) -> tables.Webhook:
    if not auth_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing webhook auth token.")
    if not webhook or auth_token != webhook.auth_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook auth token.")
    return webhook


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
    "/integrations/turn/webhook/{webhook_id}/config-updated",
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
async def receive_turn_journey_update_notification(
    webhook_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    auth_token: Annotated[str | None, Header(alias=constants.HEADER_WEBHOOK_TOKEN)] = None,
):
    """
    This endpoint is used as the webhook URL for Turn.io to notify us of changes to the Journeys.
    It is not intended to be called directly by clients, and will return a 400 error if called without
    a valid Turn.io webhook auth token.

    This endpoint only enqueues a task; it does not perform the refresh itself. A `tq` worker picks up
    the task and triggers the actual refresh by calling `refetch_journeys_from_turn` below, over HTTP.
    This is intentional: Turn.io's webhook calls timeout in ~5s, and large refreshes may take longer.
    See `make_turn_journeys_changed_handler` in `xngin/tq/handlers.py` for the worker side logic.
    """
    turn_webhook = await get_turn_webhook_or_raise(session, webhook_id=webhook_id, allow_missing=False)
    turn_webhook = check_webhook_auth_token(auth_token, turn_webhook)

    session.add(
        tables.Task(
            task_type=TURN_JOURNEYS_CHANGED_TASK_TYPE,
            payload=TurnJourneysChangedTask(
                organization_id=turn_webhook.organization_id, webhook_id=webhook_id, webhook_auth_token=auth_token
            ).model_dump(),
        )
    )
    await session.commit()
    return GENERIC_SUCCESS


@router.post(
    "/integrations/turn/webhook/{webhook_id}/refresh-journeys",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        "404": {
            "model": dict,
            "description": "The specified webhook or Turn.io connection was not found.",
        },
    },
)
async def refetch_journeys_from_turn(
    webhook_id: str,
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    httpx_client: Annotated[httpx2.AsyncClient, Depends(retrying_httpx_dependency)],
    auth_token: Annotated[str | None, Header(alias=constants.HEADER_WEBHOOK_TOKEN)] = None,
):
    """
    Refreshes the cached Turn.io journeys for the organization owning this webhook.

    This endpoint is called in two ways:
    1. By the `tq` worker's `turn_journeys_changed` handler, in response to a Turn.io webhook
       notification (see `receive_turn_journey_update_notification` above). The worker calls this
       endpoint over HTTP, authenticating with the same Webhook-Token any other caller would use,
       rather than invoking the refresh logic in-process — `tq` intentionally has no dependency on
       Evidential's business logic or database models, so it treats this as an external API call.
    2. Directly, as a way to manually trigger a refresh without waiting for a Turn.io webhook or
       rotating the connection's token (see `set_organization_turn_connection` in
       `admin_integrations_api.py`).
    """
    webhook = await get_turn_webhook_or_raise(session, webhook_id=webhook_id, allow_missing=False)
    webhook = check_webhook_auth_token(auth_token, webhook)

    turn_connection = (
        await session.execute(
            select(tables.TurnConnection).where(tables.TurnConnection.organization_id == webhook.organization_id)
        )
    ).scalar_one_or_none()

    if not turn_connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No Turn.io connection configured for this organization."
        )

    journeys = await refresh_journeys_dict(turn_connection.get_turn_api_token(), httpx_client)
    turn_connection.journeys_dict = {journey.name: journey.uuid for journey in journeys}
    await session.commit()

    return GENERIC_SUCCESS
