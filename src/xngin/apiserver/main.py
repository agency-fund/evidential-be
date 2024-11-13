from contextlib import asynccontextmanager
import datetime
from typing import List, Dict, Any, Annotated
import logging
import uuid

import httpx
import sqlalchemy
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi import Request
from sqlalchemy import distinct
from starlette.responses import JSONResponse

from xngin.apiserver import database
from xngin.apiserver.api_types import (
    DataTypeClass,
    AudienceSpec,
    DesignSpec,
    UnimplementedResponse,
    GetStrataResponseElement,
    GetFiltersResponseElement,
    GetMetricsResponseElement,
)
from xngin.apiserver.dependencies import (
    settings_dependency,
    config_dependency,
    gsheet_cache,
)
from xngin.apiserver.gsheet_cache import GSheetCache
from xngin.apiserver.settings import (
    get_settings_for_server,
    XnginSettings,
    ClientConfig,
    CannotFindTableException,
    get_sqlalchemy_table_from_engine,
    CannotFindUnitException,
)
from xngin.apiserver.utils import safe_for_headers
from xngin.sheets.config_sheet import (
    fetch_and_parse_sheet,
    create_sheetconfig_from_table,
)
import warnings

# Workaround for: https://github.com/fastapi/fastapi/discussions/10537
warnings.filterwarnings(
    "ignore",
    message="`example`",
    category=DeprecationWarning,
    module="xngin.apiserver.main",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_settings_for_server()
    database.setup()
    yield


app = FastAPI(lifespan=lifespan)
logger = logging.getLogger(__name__)


# TODO: unify exception handling
@app.exception_handler(CannotFindTableException)
async def exception_handler_cannotfindthetableexception(
    _request: Request, exc: CannotFindTableException
):
    return JSONResponse(status_code=404, content={"message": exc.message})


@app.exception_handler(CannotFindUnitException)
async def exception_handler_cannotfindtheunitexception(
    _request: Request, exc: CannotFindUnitException
):
    return JSONResponse(status_code=404, content={"message": exc.message})


class CommonQueryParams:
    """Describes query parameters common to the /strata, /filters, and /metrics APIs."""

    def __init__(
        self,
        unit_type: Annotated[
            str,
            Query(
                description="Unit of analysis for experiment.", example="test_unit_type"
            ),
        ],
        refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
    ):
        self.unit_type = unit_type
        self.refresh = refresh


# API Endpoints
@app.get(
    "/strata", summary="Get possible strata covariates.", tags=["Experiment Design"]
)
def get_strata(
    commons: Annotated[CommonQueryParams, Depends()],
    gsheets: Annotated[GSheetCache, Depends(gsheet_cache)],
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
) -> list[GetStrataResponseElement]:
    """
    Get possible strata covariates for a given unit type.

    This reimplements dwh.R get_strata().
    """
    config = require_config(client)
    with config.dbsession(commons.unit_type) as session:
        sa_table = get_sqlalchemy_table_from_engine(
            session.get_bind(), commons.unit_type
        )
        db_schema = generate_column_descriptors(sa_table)
        config_sheet = fetch_worksheet(commons, config, gsheets)
        strata_cols = {c.column_name: c for c in config_sheet.columns if c.is_strata}

    return sorted(
        [
            GetStrataResponseElement(
                data_type=db_schema.get(col_name).data_type,
                column_name=col_name,
                description=col_descriptor.description,
                # TODO: work on naming for strata_group/column_group
                strata_group=col_descriptor.column_group,
            )
            for col_name, col_descriptor in strata_cols.items()
            if db_schema.get(col_name)
        ],
        key=lambda item: item.column_name,
    )


@app.get(
    "/filters",
    summary="Get possible filters covariates for a given unit type.",
    tags=["Experiment Design"],
)
def get_filters(
    commons: Annotated[CommonQueryParams, Depends()],
    gsheets: Annotated[GSheetCache, Depends(gsheet_cache)],
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
) -> list[GetFiltersResponseElement]:
    config = require_config(client)
    with config.dbsession(commons.unit_type) as session:
        sa_table = get_sqlalchemy_table_from_engine(
            session.get_bind(), commons.unit_type
        )
        db_schema = generate_column_descriptors(sa_table)
        config_sheet = fetch_worksheet(commons, config, gsheets)
        filter_cols = {c.column_name: c for c in config_sheet.columns if c.is_filter}

        # TODO: implement caching, respecting commons.refresh
        def mapper(col_name, column_descriptor):
            db_col = db_schema.get(col_name)
            filter_class = db_col.data_type.filter_class(col_name)

            # Collect metadata on the values in the database.
            sa_col = sa_table.columns[col_name]
            distinct_values, min_, max_ = None, None, None
            match filter_class:
                case DataTypeClass.DISCRETE:
                    distinct_values = [
                        str(v)
                        for v in session.execute(
                            sqlalchemy.select(distinct(sa_col))
                            .where(sa_col.is_not(None))
                            .order_by(sa_col)
                        ).scalars()
                    ]
                case DataTypeClass.NUMERIC:
                    min_, max_ = session.execute(
                        sqlalchemy.select(
                            sqlalchemy.func.min(sa_col), sqlalchemy.func.max(sa_col)
                        ).where(sa_col.is_not(None))
                    ).first()
                case _:
                    raise RuntimeError("unexpected filter class")

            return GetFiltersResponseElement(
                filter_name=col_name,
                data_type=db_col.data_type,
                relations=filter_class.valid_relations(),
                description=column_descriptor.description,
                distinct_values=distinct_values,
                min=min_,
                max=max_,
            )

        return sorted(
            [
                mapper(col_name, col_descriptor)
                for col_name, col_descriptor in filter_cols.items()
                if db_schema.get(col_name)
            ],
            key=lambda item: item.filter_name,
        )


@app.get(
    "/metrics",
    summary="Get possible metric covariates for a given unit type.",
    tags=["Experiment Design"],
)
def get_metrics(
    commons: Annotated[CommonQueryParams, Depends()],
    gsheets: Annotated[GSheetCache, Depends(gsheet_cache)],
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
) -> list[GetMetricsResponseElement]:
    """
    Get possible metrics for a given unit type.

    This reimplements dwh.R get_metrics().
    """
    config = require_config(client)
    with config.dbsession(commons.unit_type) as session:
        sa_table = get_sqlalchemy_table_from_engine(
            session.get_bind(), commons.unit_type
        )
        db_schema = generate_column_descriptors(sa_table)
        config_sheet = fetch_worksheet(commons, config, gsheets)
        metric_cols = {c.column_name: c for c in config_sheet.columns if c.is_metric}

    return sorted(
        [
            GetMetricsResponseElement(
                data_type=db_schema.get(col_name).data_type,
                column_name=col_name,
                description=col_descriptor.description,
            )
            for col_name, col_descriptor in metric_cols.items()
            if db_schema.get(col_name)
        ],
        key=lambda item: item.column_name,
    )


@app.post(
    "/power",
    summary="Check power given an experiment and audience specification.",
    response_model=UnimplementedResponse,
    tags=["Experiment Design"],
)
def check_power(
    design_spec: DesignSpec,
    audience_spec: AudienceSpec,
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
):
    # Implement power calculation logic
    return UnimplementedResponse()


@app.post(
    "/assign",
    summary="Assign treatment given experiment and audience specification.",
    response_model=UnimplementedResponse,
    tags=["Manage Experiments"],
)
def assign_treatment(
    design_spec: DesignSpec,
    audience_spec: AudienceSpec,
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
    chosen_n: int = 1000,
):
    # Implement treatment assignment logic
    return UnimplementedResponse()


@app.post(
    "/assignment-file",
    summary="TODO",
    response_model=UnimplementedResponse,
    tags=["Manage Experiments"],
)
def assignment_file(
    design_spec: DesignSpec,
    audience_spec: AudienceSpec,
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
    chosen_n: int = 1000,
):
    # Implement treatment assignment logic
    return UnimplementedResponse()


@app.post(
    "/commit",
    summary="Commit an experiment to the database.",
    response_model=Any,  # Any since we're forwarding the webhook response
    tags=["Manage Experiments"],
)
async def commit_experiment(
    design_spec: DesignSpec,
    audience_spec: AudienceSpec,
    # TODO: convert to a proper api_types.py model
    experiment_assignment: Dict[str, Any],
    user_id: str = "testuser",
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
):
    config = require_config(client).webhook_config
    action = config.actions.commit
    if action is None:
        # TODO: write to internal storage if webhooks are not defined.
        raise HTTPException(501, "Action 'commit' not configured.")

    # Expect user webhook to take a POST payload:
    async with httpx.AsyncClient() as http_client:
        headers = {}
        auth_header_value = config.common_headers.authorization
        if auth_header_value is not None:
            headers["Authorization"] = auth_header_value

        # TODO: convert to a proper api_types.py model
        data = {
            "experiment_commit_datetime": datetime.datetime.now().isoformat(),
            "experiment_commit_id": str(uuid.uuid4()),
            "creator_user_id": user_id,
            "experiment_assignment": experiment_assignment,
            "design_spec": design_spec.model_dump_json(),
            "audience_spec": audience_spec.model_dump_json(),
        }

        try:
            # dynamically call method based on action
            method = action.method
            dispatcher = {
                "get": httpx.AsyncClient.get,
                "post": httpx.AsyncClient.post,
                "put": httpx.AsyncClient.put,
                "patch": httpx.AsyncClient.patch,
            }
            response = await dispatcher[method](
                http_client, url=action.url, headers=headers, json=data
            )
            # Stricter than response.raise_for_status(), we require HTTP 200:
            if response.status_code != 200:
                logger.error(
                    "ERROR response %s requesting webhook: %s",
                    response.status_code,
                    action.url,
                )
                raise HTTPException(
                    status_code=502,  # Would a 421 be better?
                    detail=f"webhook request failed with status {response.status_code}",
                )
        except httpx.RequestError as e:
            logger.error("ERROR requesting webhook: %s (%s)", e.request.url, str(e))
            raise HTTPException(status_code=500, detail="server error") from e

    # TODO: embed response in our own custom return type for better extensibility
    return response.json()


@app.post(
    "/update-commit",
    summary="TODO",
    response_model=UnimplementedResponse,
    tags=["Manage Experiments"],
)
def update_experiment(
    design_spec: DesignSpec,
    audience_spec: AudienceSpec,
    experiment_assignment: Dict[str, Any],
    user_id: str = "testuser",
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
):
    # Implement experiment commit logic
    return UnimplementedResponse()


@app.get("/_settings", include_in_schema=False)
def debug_settings(
    request: Request,
    settings: Annotated[XnginSettings, Depends(settings_dependency)],
    config: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
):
    """Endpoint for testing purposes. Returns the current server configuration and optionally the config ID."""
    # Secrets will not be returned because they are stored as SecretStrs, but nonetheless this method
    # should only be invoked from trusted IP addresses.
    if request.client.host not in settings.trusted_ips:
        raise HTTPException(403)

    config_id = None
    if config:
        config_id = config.id
    return {"settings": settings, "config_id": config_id}


# Main experiment assignment function
def experiment_assignment(
    design_spec: DesignSpec, audience_spec: AudienceSpec, chosen_n: int
):
    # Implement experiment assignment logic
    pass


# Helper functions for database operations
def get_dwh_participants(audience_spec: AudienceSpec, chosen_n: int):
    # Implement logic to get participants from the data warehouse
    pass


def get_metric_meta(metrics: List[str], audience_spec: AudienceSpec):
    # Implement logic to get metric metadata
    pass


# MongoDB interaction function
def experiments_reg_request(
    settings: XnginSettings, endpoint: str, json_data: Dict[str, Any] | None = None
):
    url = f"https://{settings.api_host}/dev/api/v1/experiment-commit/{endpoint}"

    api_token = safe_for_headers(
        settings.get_client_config("customer").config.api_token.get_secret_value()
    )
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_token}",
    }

    if (
        endpoint.startswith("get-file-name-by-experiment-id")
        or endpoint == "get-all-experiments"
    ):
        response = httpx.get(url, headers=headers)
    else:
        if endpoint.startswith("update"):
            response = httpx.put(url, headers=headers, json=json_data)
        else:
            response = httpx.post(url, headers=headers, json=json_data)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Request failed")

    return response.json()


def require_config(client: ClientConfig | None):
    """Raises an exception unless we have a usable ClientConfig available."""
    if not client:
        raise HTTPException(
            404, "Configuration for the requested client was not found."
        )
    return client.config


def fetch_worksheet(commons: CommonQueryParams, config, gsheets: GSheetCache):
    """Fetches a worksheet from the cache, reading it from the source if refresh or if the cache doesn't have it."""
    sheet = config.find_unit(commons.unit_type).sheet
    return gsheets.get(
        sheet,
        lambda: fetch_and_parse_sheet(sheet),
        refresh=commons.refresh,
    )


def generate_column_descriptors(table: sqlalchemy.Table):
    """Fetches a map of column name to SheetConfig column metadata.

    Raises 500 if the table does not exist.
    """
    try:
        return {c.column_name: c for c in create_sheetconfig_from_table(table).columns}
    except CannotFindTableException as cfte:
        raise HTTPException(status_code=500, detail=cfte.message) from cfte


def main():
    database.setup()

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
