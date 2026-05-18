"""
This module defines the public API for clients to integrate with specific third-party tools.
Currently, we only support Turn.io, so this includes endpoints for the Turn.io Evidential App config

(See admin_integrations_api.py for Evidential UI-facing integration endpoints.)
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import constants
from xngin.apiserver.dependencies import (
    xngin_db_session,
)
from xngin.apiserver.routers.common_api_types import TurnConfigResponse
from xngin.apiserver.routers.experiments.dependencies import (
    experiment_dependency,
)
from xngin.apiserver.routers.experiments.experiments_api import STANDARD_INTEGRATION_RESPONSES
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
