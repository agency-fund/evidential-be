from typing import Annotated

from fastapi import Depends, HTTPException, Path
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from starlette import status

from xngin.apiserver import constants
from xngin.apiserver.apikeys import hash_key_or_raise
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.models import tables


async def experiment_dependency(
    experiment_id: Annotated[
        str, Path(..., description="The ID of the experiment to fetch.")
    ],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
    api_key: Annotated[
        str | None,
        Depends(APIKeyHeader(name=constants.HEADER_API_KEY, auto_error=False)),
    ],
) -> tables.Experiment:
    """
    Returns the Experiment db object for experiment_id, if the API key grants access to its
    datasource.

    Raises:
        ApiKeyError: If the API key is invalid/missing.
        HTTPException: 404 if the experiment is not found or the API key is invalid for the experiment's datasource.
    """
    key_hash = hash_key_or_raise(api_key)
    # We use joinedload(arms) because we anticipate that inspecting the arms of the experiment will be common, and it
    # is also used in the online experiment assignment flow which is sensitive to database roundtrips.
    query = (
        select(tables.Experiment)
        .join(
            tables.ApiKey,
            tables.Experiment.datasource_id == tables.ApiKey.datasource_id,
        )
        .options(joinedload(tables.Experiment.arms))
        .where(
            tables.Experiment.id == experiment_id,
            tables.ApiKey.key == key_hash,
        )
    )
    experiment = (await xngin_session.scalars(query)).unique().one_or_none()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found or not authorized.",
        )

    return experiment
