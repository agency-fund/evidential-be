"""FastAPI dependencies that resolve integrator API parameters to the resources they name.

The counterpart to admin_dependencies for the routes integrators call, and read the same way at the call
site: `edeps.experiment` resolves the resource its route names and rejects a caller who may not have it.
What differs is the principal. These callers present a datasource API key rather than a session, and the
datasource itself is named by a header rather than by the path.

A `_with_x` suffix names an added relationship that any route may want; a `_for_x` suffix names the
particular set one route needs. Per-request caching is keyed on the dependency object, so two instances of
_Experiment are two separate lookups.
"""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Path
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.orm import QueryableAttribute, Session, joinedload
from starlette import status

from xngin.apiserver import apikeys, constants
from xngin.apiserver.apikeys import hash_key_or_raise, require_valid_api_key
from xngin.apiserver.dependencies import CannotFindDatasourceError, xngin_sync_db_session
from xngin.apiserver.routers.preloads import (
    EXPERIMENT_FIELDS_WITH_FILTERS,
    PreloadChain,
    build_preload_options,
)
from xngin.apiserver.settings import (
    Datasource,
)
from xngin.apiserver.sqla import tables


class _DatasourceApiKeyHeader(APIKeyHeader):
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


def datasource(
    datasource_id: Annotated[
        str,
        Header(
            examples=["testing"],
            alias=constants.HEADER_CONFIG_ID,
            description="The ID of the datasource to operate on.",
        ),
    ],
    xngin_session: Annotated[Session, Depends(xngin_sync_db_session)],
    api_key: Annotated[
        str,
        Depends(_DatasourceApiKeyHeader()),
    ],
):
    """Returns the configuration for the current request, as determined by the Datasource-ID HTTP request header."""
    if not datasource_id:
        raise CannotFindDatasourceError(f"{constants.HEADER_CONFIG_ID} is required.")

    if from_db := xngin_session.get(tables.Datasource, datasource_id):
        require_valid_api_key(xngin_session, api_key, datasource_id)
        dsconfig = from_db.get_config()
        return Datasource(id=datasource_id, config=dsconfig)

    raise CannotFindDatasourceError("Datasource not found.")


class _Experiment:
    """
    Parameterizable db Experiment dependency (instances are callable) for endpoints that require API keys.

    When constructing the dependency, you can provide a list of experiment attributes to preload to
    avoid N+1 queries.

    You can alternatively provide a list of preload chains, each of which walks through one
    relationship to reach another, using the preloading method named for each level.

    See __call__ for additional injected parameters when called as a dependency.
    """

    def __init__(
        self,
        preload: list[QueryableAttribute] | None = None,
        nested_preload: list[PreloadChain] | None = None,
    ) -> None:
        self.preload = preload
        self.nested_preload = nested_preload

    def __call__(
        self,
        experiment_id: Annotated[str, Path(..., description="The ID of the experiment to fetch.")],
        api_key: Annotated[
            str,
            Depends(_DatasourceApiKeyHeader()),
        ],
        xngin_session: Annotated[Session, Depends(xngin_sync_db_session)],
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
        options = build_preload_options(self.preload, self.nested_preload)
        if options:
            query = query.options(*options)
        experiment = xngin_session.scalars(query).unique().one_or_none()

        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or not authorized.",
            )

        return experiment


experiment = _Experiment()

# TODO: remove the datasource dependency as part of the participant type cleanup.
experiment_with_datasource_and_fields = _Experiment(
    preload=[
        tables.Experiment.datasource,
    ],
    nested_preload=[EXPERIMENT_FIELDS_WITH_FILTERS],
)

experiment_with_contexts = _Experiment(preload=[tables.Experiment.contexts])

experiment_for_full_response = _Experiment(
    preload=[
        tables.Experiment.webhooks,
        tables.Experiment.contexts,
    ],
    nested_preload=[EXPERIMENT_FIELDS_WITH_FILTERS],
)
