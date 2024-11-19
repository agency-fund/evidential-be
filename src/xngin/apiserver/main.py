from contextlib import asynccontextmanager
from typing import List, Dict, Any, Annotated, Literal, Tuple, Union
import logging
import warnings

import httpx
from pydantic import BaseModel
import sqlalchemy
from fastapi import FastAPI, HTTPException, Depends, Path, Query, Response
from fastapi import Request
from sqlalchemy import distinct
from starlette.responses import JSONResponse

from xngin.apiserver import database
from xngin.apiserver.api_types import (
    DataTypeClass,
    AudienceSpec,
    DesignSpec,
    ExperimentAssignment,
    UnimplementedResponse,
    GetStrataResponseElement,
    GetFiltersResponseElement,
    GetMetricsResponseElement,
)
from xngin.apiserver.dependencies import (
    httpx_dependency,
    settings_dependency,
    config_dependency,
    gsheet_cache,
)
from xngin.apiserver.gsheet_cache import GSheetCache
from xngin.apiserver.settings import (
    WebhookConfig,
    WebhookUrl,
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
from xngin.apiserver.webhook_types import (
    STANDARD_WEBHOOK_RESPONSES,
    UpdateExperimentDescriptionsRequest,
    UpdateExperimentStartEndRequest,
    WebhookRequestCommit,
    WebhookRequestUpdate,
    WebhookResponse,
)

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
    responses=STANDARD_WEBHOOK_RESPONSES,
    tags=["Manage Experiments"],
)
async def commit_experiment(
    response: Response,
    design_spec: DesignSpec,
    audience_spec: AudienceSpec,
    experiment_assignment: ExperimentAssignment,
    user_id: str = "testuser",
    http_client: Annotated[httpx.AsyncClient, Depends(httpx_dependency)] = None,
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
) -> WebhookResponse:
    config = require_config(client).webhook_config
    action = config.actions.commit
    if action is None:
        # TODO: write to internal storage if webhooks are not defined.
        raise HTTPException(501, "Action 'commit' not configured.")

    commit_payload = WebhookRequestCommit(
        creator_user_id=user_id,
        experiment_assignment=experiment_assignment,
        design_spec=design_spec,
        audience_spec=audience_spec,
    )

    response.status_code, payload = await make_webhook_request(
        http_client, config, action, commit_payload
    )
    return payload


@app.post(
    "/update-commit",
    summary="Update an existing experiment's timestamps or description (experiment and arms)",
    responses=STANDARD_WEBHOOK_RESPONSES,
    tags=["Manage Experiments"],
)
async def update_experiment(
    response: Response,
    request_payload: WebhookRequestUpdate,
    update_type: Annotated[
        Literal["timestamps", "description"],
        Query(description="The type of experiment metadata update to perform"),
    ],
    http_client: Annotated[httpx.AsyncClient, Depends(httpx_dependency)] = None,
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
) -> WebhookResponse:
    config = require_config(client).webhook_config
    action = None
    if update_type == "timestamps":
        action = config.actions.update_timestamps
    elif update_type == "description":
        action = config.actions.update_description
    if action is None:
        # TODO: write to internal storage if webhooks are not defined.
        raise HTTPException(501, f"Action '{update_type}' not configured.")
    # Need to pull out the upstream server payload:
    response.status_code, payload = await make_webhook_request(
        http_client, config, action, request_payload.update_json
    )
    return payload


@app.post(
    "/experiment/{experiment_id}",
    summary="Update an existing experiment. (limited update capabilities)",
    responses=STANDARD_WEBHOOK_RESPONSES,
    tags=["WIP New API"],
)
async def alt_update_experiment(
    response: Response,
    body: Union[UpdateExperimentStartEndRequest, UpdateExperimentDescriptionsRequest],
    experiment_id: str = Annotated[
        str, Path(description="The ID of the experiment to update.")
    ],
    http_client: Annotated[httpx.AsyncClient, Depends(httpx_dependency)] = None,
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
) -> WebhookResponse:
    config = require_config(client).webhook_config
    # TODO: write to internal storage if no webhook_config
    action = None
    if body.update_type == "timestamps":
        action = config.actions.update_timestamps
    elif body.update_type == "description":
        action = config.actions.update_description
    if action is None:
        raise HTTPException(501, f"Action for '{body.update_type}' not configured.")

    # TODO: how best to handle experiment_id?
    # Fill it in for the user as part of the request bodies only if missing?
    # (Wouldn't be needed in the body if we handled the request here saving internally.)
    if body.experiment_id is None:
        body.experiment_id = experiment_id
    response.status_code, payload = await make_webhook_request(
        http_client, config, action, body
    )
    return payload


async def make_webhook_request(
    http_client: httpx.AsyncClient,
    config: WebhookConfig,
    action: WebhookUrl,
    data: BaseModel,
) -> Tuple[int, WebhookResponse]:
    """Helper function to make webhook requests with common error handling.

    Returns: tuple of (status_code, WebhookResponse to use as body)
    """
    # TODO: use DI for a consistently configured shared client across endpoints
    headers = {}
    auth_header_value = config.common_headers.authorization
    if auth_header_value is not None:
        headers["Authorization"] = auth_header_value
    headers["Accept"] = "application/json"
    # headers["Content-Type"] is set by httpx

    try:
        # Explicitly convert to a dict via pydantic since we use custom serializers
        json_data = data.model_dump(mode="json")
        upstream_response = await http_client.request(
            method=action.method, url=action.url, headers=headers, json=json_data
        )
        webhook_response = WebhookResponse.from_httpx(upstream_response)
        # Stricter than response.raise_for_status(), we require HTTP 200:
        status_code = 200
        if upstream_response.status_code != 200:
            logger.error(
                "ERROR response %s requesting webhook: %s",
                upstream_response.status_code,
                action.url,
            )
            status_code = 502
        # Always return a WebhookResponse in the body on HTTPStatusError and non-200 response.
        return (status_code, webhook_response)
    except httpx.RequestError as e:
        logger.error(
            "ERROR %s requesting webhook: %s (%s)", type(e), e.request.url, str(e)
        )
        raise HTTPException(status_code=500, detail="server error") from e


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
def assign_units_to_arms(
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
