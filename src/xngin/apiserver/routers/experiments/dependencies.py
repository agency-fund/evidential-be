from typing import Annotated

from fastapi import Depends, HTTPException, Path
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import QueryableAttribute, joinedload, selectinload
from starlette import status

from xngin.apiserver import constants
from xngin.apiserver.apikeys import hash_key_or_raise
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.sqla import tables


class ExperimentDependency:
    """
    Parameterizable db Experiment dependency (instances are callable) for endpoints that require API keys.

    When constructing the dependency, you can provide a list of experiment attributes to preload to avoid N+1 queries.
    See __call__ for additional injected parameters when called as a dependency.
    """

    def __init__(self, preload: list[QueryableAttribute] | None = None) -> None:
        self.preload = preload

    async def __call__(
        self,
        experiment_id: Annotated[str, Path(..., description="The ID of the experiment to fetch.")],
        api_key: Annotated[
            str | None,
            Depends(APIKeyHeader(name=constants.HEADER_API_KEY, auto_error=False)),
        ],
        xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
    ) -> tables.Experiment:
        """
        Returns the Experiment db object for experiment_id, if the API key grants access to its datasource.
        - experiment_id is pulled from the endpoint's path.
        - api_key is pulled from the API key header.
        - xngin_session is our injected database session.

        Raises:
            ApiKeyError: If the API key is invalid/missing.
            HTTPException: 404 if the experiment is not found or the API key is invalid for the experiment's datasource.
        """
        key_hash = hash_key_or_raise(api_key)
        # We use joinedload(arms) because we anticipate that inspecting the arms of the
        # experiment will be common. It is also used in the online experiment assignment
        # flow, which is sensitive to database roundtrips.
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
        if self.preload:
            query = query.options(*[selectinload(f) for f in self.preload])
        experiment = (await xngin_session.scalars(query)).unique().one_or_none()

        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or not authorized.",
            )

        return experiment


# Default dependency for experiments that only need arms joined in.
experiment_dependency = ExperimentDependency()

# Use this version when a full GetExperimentResponse is needed.
experiment_response_dependency = ExperimentDependency(preload=[tables.Experiment.webhooks, tables.Experiment.contexts])

# This version is used with processing assignments, e.g. exporting them.
experiment_with_assignments_dependency = ExperimentDependency(
    preload=[tables.Experiment.arm_assignments, tables.Experiment.draws],
)
