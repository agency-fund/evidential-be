import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import httpx
import sqlalchemy
from fastapi import FastAPI, HTTPException, Depends, Path, Query, Response
from fastapi import Request
from fastapi.openapi.utils import get_openapi
from pandas import DataFrame
from pydantic import BaseModel
from sqlalchemy import distinct

from xngin.apiserver import database, exceptionhandlers, middleware
from xngin.apiserver.api_types import (
    DataTypeClass,
    AssignResponse,
    GetFiltersResponseDiscrete,
    GetFiltersResponseNumeric,
    GetStrataResponseElement,
    GetMetricsResponseElement,
    PowerResponse,
    GetStrataResponse,
    GetFiltersResponse,
    GetMetricsResponse,
    AssignRequest,
    CommitRequest,
    PowerRequest,
)
from xngin.apiserver.dependencies import (
    httpx_dependency,
    settings_dependency,
    config_dependency,
    gsheet_cache,
)
from xngin.apiserver.dwh.queries import get_stats_on_metrics, query_for_participants
from xngin.apiserver.gsheet_cache import GSheetCache
from xngin.apiserver.routers import oidc, admin
from xngin.apiserver.settings import (
    ParticipantsMixin,
    WebhookConfig,
    WebhookUrl,
    get_settings_for_server,
    XnginSettings,
    ClientConfig,
    infer_table,
)
from xngin.apiserver.utils import substitute_url
from xngin.apiserver.webhook_types import (
    STANDARD_WEBHOOK_RESPONSES,
    UpdateExperimentDescriptionsRequest,
    UpdateExperimentStartEndRequest,
    WebhookRequestCommit,
    WebhookResponse,
    UpdateCommitRequest,
)
from xngin.sheets.config_sheet import (
    fetch_and_parse_sheet,
    create_configworksheet_from_table,
)
from xngin.stats.assignment import assign_treatment
from xngin.stats.power import check_power


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if sentry_dsn := os.environ.get("SENTRY_DSN"):
    import sentry_sdk

    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=os.environ.get("ENVIRONMENT", "local"),
        traces_sample_rate=1.0,
        _experiments={
            "continuous_profiling_auto_start": True,
        },
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_settings_for_server()
    database.setup()
    yield


# TODO: pass parameters to Swagger to support OIDC/PKCE
app = FastAPI(lifespan=lifespan)
exceptionhandlers.setup(app)
middleware.setup(app)
if oidc.is_enabled():
    app.include_router(oidc.router, tags=["Auth"], include_in_schema=False)
if oidc.is_enabled() and admin.is_enabled():
    app.include_router(admin.router, tags=["Admin"], include_in_schema=False)


def custom_openapi():
    """Customizes the generated OpenAPI schema."""
    if app.openapi_schema:  # cache
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="xngin: Experiments API",
        version="0.9.0",
        summary="",
        description="",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


class CommonQueryParams:
    """Describes query parameters common to the /strata, /filters, and /metrics APIs."""

    def __init__(
        self,
        participant_type: Annotated[
            str,
            Query(
                description="Unit of analysis for experiment.",
                example="test_participant_type",
            ),
        ],
        refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
    ):
        self.participant_type = participant_type
        self.refresh = refresh


# API Endpoints
@app.get(
    "/strata", summary="Get possible strata covariates.", tags=["Experiment Design"]
)
def get_strata(
    commons: Annotated[CommonQueryParams, Depends()],
    gsheets: Annotated[GSheetCache, Depends(gsheet_cache)],
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
) -> GetStrataResponse:
    """Get possible strata covariates for a given unit type."""
    config = require_config(client)
    participants = config.find_participants(commons.participant_type)
    config_sheet = fetch_worksheet(commons, config, gsheets)
    strata_cols = {c.column_name: c for c in config_sheet.columns if c.is_strata}

    with config.dbsession() as session:
        sa_table = infer_table(
            session.get_bind(), participants.table_name, config.supports_reflection()
        )
        db_schema = generate_column_descriptors(
            sa_table, config_sheet.get_unique_id_col()
        )

    return sorted(
        [
            GetStrataResponseElement(
                data_type=db_schema.get(col_name).data_type,
                column_name=col_name,
                description=col_descriptor.description,
                # For strata columns, we will echo back any extra annotations
                extra=col_descriptor.extra,
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
) -> GetFiltersResponse:
    config = require_config(client)
    participants = config.find_participants(commons.participant_type)
    config_sheet = fetch_worksheet(commons, config, gsheets)
    filter_cols = {c.column_name: c for c in config_sheet.columns if c.is_filter}

    with config.dbsession() as session:
        sa_table = infer_table(
            session.get_bind(), participants.table_name, config.supports_reflection()
        )
        db_schema = generate_column_descriptors(
            sa_table, config_sheet.get_unique_id_col()
        )

        # TODO: implement caching, respecting commons.refresh
        def mapper(col_name, column_descriptor):
            db_col = db_schema.get(col_name)
            filter_class = db_col.data_type.filter_class(col_name)

            # Collect metadata on the values in the database.
            sa_col = sa_table.columns[col_name]
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
                    return GetFiltersResponseDiscrete(
                        filter_name=col_name,
                        data_type=db_col.data_type,
                        relations=filter_class.valid_relations(),
                        description=column_descriptor.description,
                        distinct_values=distinct_values,
                    )
                case DataTypeClass.NUMERIC:
                    min_, max_ = session.execute(
                        sqlalchemy.select(
                            sqlalchemy.func.min(sa_col), sqlalchemy.func.max(sa_col)
                        ).where(sa_col.is_not(None))
                    ).first()
                    return GetFiltersResponseNumeric(
                        filter_name=col_name,
                        data_type=db_col.data_type,
                        relations=filter_class.valid_relations(),
                        description=column_descriptor.description,
                        min=min_,
                        max=max_,
                    )
                case _:
                    raise RuntimeError("unexpected filter class")

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
) -> GetMetricsResponse:
    """Get possible metrics for a given unit type."""
    config = require_config(client)
    participants = config.find_participants(commons.participant_type)
    config_sheet = fetch_worksheet(commons, config, gsheets)
    metric_cols = {c.column_name: c for c in config_sheet.columns if c.is_metric}

    with config.dbsession() as session:
        sa_table = infer_table(
            session.get_bind(), participants.table_name, config.supports_reflection()
        )
        db_schema = generate_column_descriptors(
            sa_table, config_sheet.get_unique_id_col()
        )

    # Merge data type info above with the columns to be used as metrics:
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
    tags=["Experiment Design"],
)
def check_power_api(
    body: PowerRequest,
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
) -> PowerResponse:
    """
    Calculates statistical power given an AudienceSpec and a DesignSpec
    """
    config = require_config(client)
    participant = config.find_participants(body.audience_spec.participant_type)

    with config.dbsession() as session:
        sa_table = infer_table(
            session.get_bind(), participant.table_name, config.supports_reflection()
        )

        metric_stats = get_stats_on_metrics(
            session,
            sa_table,
            body.design_spec.metrics,
            body.audience_spec,
        )

        return check_power(
            metrics=metric_stats,
            n_arms=len(body.design_spec.arms),
            power=body.design_spec.power,
            alpha=body.design_spec.alpha,
        )


@app.post(
    "/assign",
    summary="Assign treatment given experiment and audience specification.",
    tags=["Experiment Management"],
)
def assign_treatment_api(
    body: AssignRequest,
    chosen_n: Annotated[
        int, Query(..., description="Number of participants to assign.")
    ],
    gsheets: Annotated[GSheetCache, Depends(gsheet_cache)],
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
    random_state: Annotated[
        int | None,
        Query(
            description="Specify a random seed for reproducibility.",
            include_in_schema=False,
        ),
    ] = None,
) -> AssignResponse:
    config = require_config(client)
    participant = config.find_participants(body.audience_spec.participant_type)
    config_sheet = fetch_worksheet(
        CommonQueryParams(
            participant_type=participant.participant_type, refresh=refresh
        ),
        config,
        gsheets,
    )
    unique_id_col = config_sheet.get_unique_id_col()

    with config.dbsession() as session:
        sa_table = infer_table(
            session.get_bind(), participant.table_name, config.supports_reflection()
        )
        participants = query_for_participants(
            session, sa_table, body.audience_spec, chosen_n
        )

    arm_names = [arm.arm_name for arm in body.design_spec.arms]
    metric_names = [m.metric_name for m in body.design_spec.metrics]
    return assign_treatment(
        data=DataFrame(participants),
        stratum_cols=body.design_spec.strata_cols + metric_names,
        id_col=unique_id_col,
        arm_names=arm_names,
        experiment_id=str(body.design_spec.experiment_id),
        description=body.design_spec.description,
        fstat_thresh=body.design_spec.fstat_thresh,
        random_state=random_state,
    )


@app.get(
    "/assignment-file",
    summary="Retrieve all participant assignments for the given experiment_id.",
    responses=STANDARD_WEBHOOK_RESPONSES,
    tags=["Experiment Management"],
)
async def assignment_file(
    response: Response,
    experiment_id: Annotated[
        str,
        Query(description="ID of the experiment whose assignments we wish to fetch."),
    ],
    http_client: Annotated[httpx.AsyncClient, Depends(httpx_dependency)],
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
) -> WebhookResponse:
    config = require_config(client).webhook_config
    action = config.actions.assignment_file
    if action is None:
        # TODO: read from internal storage if webhooks are not defined.
        raise HTTPException(501, "Action 'assignment_file' not configured.")

    url = substitute_url(action.url, {"experiment_id": experiment_id})
    response.status_code, payload = await make_webhook_request_base(
        http_client, config, method=action.method, url=url
    )
    return payload


@app.post(
    "/commit",
    summary="Commit an experiment to the database.",
    responses=STANDARD_WEBHOOK_RESPONSES,
    tags=["Experiment Management"],
)
async def commit_experiment(
    response: Response,
    body: CommitRequest,
    user_id: Annotated[str, Query(...)],
    http_client: Annotated[httpx.AsyncClient, Depends(httpx_dependency)],
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
) -> WebhookResponse:
    config = require_config(client).webhook_config
    action = config.actions.commit
    if action is None:
        # TODO: write to internal storage if webhooks are not defined.
        raise HTTPException(501, "Action 'commit' not configured.")

    commit_payload = WebhookRequestCommit(
        creator_user_id=user_id,
        experiment_assignment=body.experiment_assignment,
        design_spec=body.design_spec,
        audience_spec=body.audience_spec,
    )

    response.status_code, payload = await make_webhook_request(
        http_client, config, action, commit_payload
    )
    return payload


@app.post(
    "/update-commit",
    summary="Update an existing experiment's timestamps or description (experiment and arms)",
    responses=STANDARD_WEBHOOK_RESPONSES,
    tags=["Experiment Management"],
)
async def update_experiment(
    response: Response,
    body: UpdateCommitRequest,
    update_type: Annotated[
        Literal["timestamps", "description"],
        Query(description="The type of experiment metadata update to perform"),
    ],
    http_client: Annotated[httpx.AsyncClient, Depends(httpx_dependency)],
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
        http_client, config, action, body.update_json
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
    body: UpdateExperimentStartEndRequest | UpdateExperimentDescriptionsRequest,
    _experiment_id: Annotated[
        str,
        Path(description="The ID of the experiment to update.", alias="experiment_id"),
    ],
    http_client: Annotated[httpx.AsyncClient, Depends(httpx_dependency)],
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
    # TODO: use the experiment_id in an upstream url
    response.status_code, payload = await make_webhook_request(
        http_client, config, action, body
    )
    return payload


async def make_webhook_request(
    http_client: httpx.AsyncClient,
    config: WebhookConfig,
    action: WebhookUrl,
    data: BaseModel,
) -> tuple[int, WebhookResponse]:
    """Helper function to make webhook requests with common error handling.

    Returns: tuple of (status_code, WebhookResponse to use as body)
    """
    return await make_webhook_request_base(
        http_client, config, action.method, action.url, data
    )


async def make_webhook_request_base(
    http_client: httpx.AsyncClient,
    config: WebhookConfig,
    method: Literal["get", "post", "put", "patch", "delete"],
    url: str,
    data: BaseModel = None,
) -> tuple[int, WebhookResponse]:
    """Like make_webhook_request() but can directly take an http method and url.

    Returns: tuple of (status_code, WebhookResponse to use as body)
    """
    headers = {}
    auth_header_value = config.common_headers.authorization
    if auth_header_value is not None:
        headers["Authorization"] = auth_header_value.get_secret_value()
    headers["Accept"] = "application/json"
    # headers["Content-Type"] is set by httpx

    try:
        # Explicitly convert to a dict via pydantic since we use custom serializers
        json_data = data.model_dump(mode="json") if data else None
        upstream_response = await http_client.request(
            method=method, url=url, headers=headers, json=json_data
        )
        webhook_response = WebhookResponse.from_httpx(upstream_response)
        status_code = 200
        # Stricter than response.raise_for_status(), we require HTTP 200:
        if upstream_response.status_code != 200:
            logger.error(
                "ERROR response %s requesting webhook: %s",
                upstream_response.status_code,
                url,
            )
            status_code = 502
    except httpx.ConnectError as e:
        logger.exception("ERROR requesting webhook (ConnectError): %s", e.request.url)
        raise HTTPException(
            status_code=502, detail=f"Error connecting to {e.request.url}: {e}"
        ) from e
    except httpx.RequestError as e:
        logger.exception("ERROR requesting webhook: %s", e.request.url)
        raise HTTPException(status_code=500, detail="server error") from e
    else:
        # Always return a WebhookResponse in the body, even on non-200 responses.
        return status_code, webhook_response


@app.get("/_settings", include_in_schema=False)
def debug_settings(
    request: Request,
    settings: Annotated[XnginSettings, Depends(settings_dependency)],
):
    """Endpoint for testing purposes. Returns the current server configuration and optionally the config ID."""
    # Secrets will not be returned because they are stored as SecretStrs, but nonetheless this method
    # should only be invoked from trusted IP addresses.
    if request.client.host not in settings.trusted_ips:
        raise HTTPException(403)
    response = {"settings": settings}
    if config_id := request.headers.get("config-id"):
        response["config_id"] = config_id
    return response


def require_config(client: ClientConfig | None):
    """Raises an exception unless we have a usable ClientConfig available."""
    if not client:
        raise HTTPException(
            404, "Configuration for the requested client was not found."
        )
    return client.config


def fetch_worksheet(
    commons: CommonQueryParams, config: ParticipantsMixin, gsheets: GSheetCache
):
    """Fetches a worksheet from the cache, reading it from the source if refresh or if the cache doesn't have it."""
    sheet = config.find_participants(commons.participant_type).sheet
    return gsheets.get(
        sheet,
        lambda: fetch_and_parse_sheet(sheet),
        refresh=commons.refresh,
    )


def generate_column_descriptors(table: sqlalchemy.Table, unique_id_col: str):
    """Fetches a map of column name to ConfigWorksheet column metadata.

    Uniqueness of the values in the column unique_id_col is assumed, not verified!
    """
    return {
        c.column_name: c
        for c in create_configworksheet_from_table(table, unique_id_col).columns
    }


def main():
    database.setup()

    import uvicorn

    # Handy for debugging in your IDE
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("UVICORN_PORT", 8000)))


if __name__ == "__main__":
    main()
