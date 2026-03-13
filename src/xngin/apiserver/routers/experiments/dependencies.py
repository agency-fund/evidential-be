from typing import Annotated

from fastapi import Depends, Header, HTTPException, Path
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import QueryableAttribute, joinedload, selectinload
from starlette import status

from xngin.apiserver import apikeys, constants
from xngin.apiserver.apikeys import hash_key_or_raise, require_valid_api_key
from xngin.apiserver.dependencies import CannotFindDatasourceError, xngin_db_session
from xngin.apiserver.settings import (
    Datasource,
)
from xngin.apiserver.sqla import tables


class DatasourceApiKeyHeader(APIKeyHeader):
    """Defines the request header for the API key in the OpenAPI spec and requires it to exist on a request.

    This does not validate the key; it only checks that it is present.
    """

    def __init__(self):
        super().__init__(
            name=constants.HEADER_API_KEY,
            description=f"The datasource-specific API key. These keys are managed in Settings > Datasources. "
            f"Datasource keys begin with `{apikeys.API_KEY_PREFIX}`.",
            scheme_name="DatasourceApiKey",
            auto_error=False,
        )

    def check_api_key(self, api_key: str | None) -> str | None:
        """Confirms that the API key is present and matches the expected structure."""
        _ = apikeys.validate_api_key(api_key)
        if api_key is None:
            return None
        return api_key


async def datasource_dependency(
    datasource_id: Annotated[
        str,
        Header(
            examples=["testing"],
            alias=constants.HEADER_CONFIG_ID,
            description="The ID of the datasource to operate on.",
        ),
    ],
    xngin_session: Annotated[AsyncSession, Depends(xngin_db_session)],
    api_key: Annotated[
        str,
        Depends(DatasourceApiKeyHeader()),
    ],
):
    """Returns the configuration for the current request, as determined by the Datasource-ID HTTP request header."""
    if not datasource_id:
        raise CannotFindDatasourceError(f"{constants.HEADER_CONFIG_ID} is required.")

    if from_db := await xngin_session.get(tables.Datasource, datasource_id):
        await require_valid_api_key(xngin_session, api_key, datasource_id)
        dsconfig = from_db.get_config()
        return Datasource(id=datasource_id, config=dsconfig)

    raise CannotFindDatasourceError("Datasource not found.")


class ExperimentDependency:
    """
    Parameterizable db Experiment dependency (instances are callable) for endpoints that require API keys.

    When constructing the dependency, you can provide a list of experiment attributes to preload to
    avoid N+1 queries.  If a *list* is provided as an item in preload, we treat it as pre-loading a
    set of nested relationships.

    See __call__ for additional injected parameters when called as a dependency.
    """

    def __init__(self, preload: list[QueryableAttribute | list[QueryableAttribute]] | None = None) -> None:
        self.preload = preload

    async def __call__(
        self,
        experiment_id: Annotated[str, Path(..., description="The ID of the experiment to fetch.")],
        api_key: Annotated[
            str,
            Depends(DatasourceApiKeyHeader()),
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
            options = []
            for f in self.preload:
                # If a list is provided, treat it as loading a nested relationship.
                if isinstance(f, list) and f:
                    nested_load = selectinload(f[0])
                    for attr in f[1:]:
                        nested_load = nested_load.selectinload(attr)
                    options.append(nested_load)
                elif isinstance(f, QueryableAttribute):
                    options.append(selectinload(f))
            query = query.options(*options)
        experiment = (await xngin_session.scalars(query)).unique().one_or_none()

        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or not authorized.",
            )

        return experiment


# Default dependency for experiments that only need arms joined in.
experiment_dependency = ExperimentDependency()

# Use this when you need an experiment with info on the fields it uses.
# TODO: remove the datasource dependency as part of the participant type cleanup.
experiment_and_datasource_dependency = ExperimentDependency(
    preload=[
        tables.Experiment.datasource,
        [tables.Experiment.experiment_fields, tables.ExperimentField.experiment_filters],
    ]
)

# Use this version when you also want contexts for assignment responses.
experiment_with_contexts_dependency = ExperimentDependency(preload=[tables.Experiment.contexts])

# Use this version when a full GetExperimentResponse is needed.
experiment_response_dependency = ExperimentDependency(
    preload=[
        tables.Experiment.webhooks,
        tables.Experiment.contexts,
        [tables.Experiment.experiment_fields, tables.ExperimentField.experiment_filters],
    ]
)

# This version is used with processing assignments, e.g. exporting them.
experiment_with_assignments_dependency = ExperimentDependency(
    preload=[tables.Experiment.arm_assignments, tables.Experiment.draws],
)
