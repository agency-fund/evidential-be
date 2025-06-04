from typing import Annotated

import httpx
from fastapi import Depends, Header, HTTPException, Path, status
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from xngin.apiserver import constants
from xngin.apiserver.apikeys import hash_key_or_raise, require_valid_api_key
from xngin.apiserver.database import SessionLocal
from xngin.apiserver.gsheet_cache import GSheetCache
from xngin.apiserver.models import tables
from xngin.apiserver.settings import (
    Datasource,
    DatasourceConfig,
    XnginSettings,
    get_settings_for_server,
)


class CannotFindDatasourceError(Exception):
    """Error raised when an invalid Datasource-ID is provided in a request."""


def random_seed_dependency():
    """Returns None; to be overridden by tests."""
    return


def settings_dependency():
    """Provides the settings for the server.

    This may be overridden by tests using the FastAPI dependency override features.
    """
    return get_settings_for_server()


def xngin_db_session():
    """Returns a database connection to the xngin app database (not customer data warehouse)."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def datasource_dependency(
    settings: Annotated[XnginSettings, Depends(settings_dependency)],
    datasource_id: Annotated[
        str,
        Header(
            example="testing",
            alias=constants.HEADER_CONFIG_ID,
            description="The ID of the datasource to operate on.",
        ),
    ],
    xngin_db: Annotated[Session, Depends(xngin_db_session)],
    api_key: Annotated[
        str | None,
        Depends(APIKeyHeader(name=constants.HEADER_API_KEY, auto_error=False)),
    ],
):
    """Returns the configuration for the current request, as determined by the Datasource-ID HTTP request header."""
    if not datasource_id:
        return None

    # Datasource configs can be in the static JSON settings or in the database.
    from_json = settings.get_datasource(datasource_id)

    # Datasources from the database always require an API key.
    if from_json is None and (
        from_db := xngin_db.get(tables.Datasource, datasource_id)
    ):
        require_valid_api_key(xngin_db, api_key, datasource_id)
        dsconfig = from_db.get_config()
        return Datasource(id=datasource_id, config=dsconfig)

    # Datasources from the static JSON settings optionally require an API key.
    if from_json and from_json.require_api_key:
        require_valid_api_key(xngin_db, api_key, datasource_id)

    if from_json is None:
        raise CannotFindDatasourceError("Invalid datasource.")

    return from_json


def datasource_config_required(
    ds: Annotated[Datasource | None, Depends(datasource_dependency)],
) -> DatasourceConfig:
    """Returns the connection-specific implementation for this datasource."""
    if ds is None:
        raise CannotFindDatasourceError("Invalid datasource.")
    return ds.config


def gsheet_cache(xngin_db: Annotated[Session, Depends(xngin_db_session)]):
    return GSheetCache(xngin_db)


async def httpx_dependency():
    """Returns a new httpx client with default configuration, to be used with each request"""
    async with httpx.AsyncClient(timeout=15.0) as client:
        yield client


def experiment_dependency(
    experiment_id: Annotated[
        str, Path(..., description="The ID of the experiment to fetch.")
    ],
    xngin_db: Annotated[Session, Depends(xngin_db_session)],
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
    experiment = xngin_db.scalars(query).unique().one_or_none()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found or not authorized.",
        )

    return experiment
