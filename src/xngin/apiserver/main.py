from contextlib import asynccontextmanager
import os
from typing import Annotated, Literal
import logging
import warnings

import httpx
from pydantic import BaseModel
import sqlalchemy
from fastapi import FastAPI, HTTPException, Depends, Path, Query, Response
from fastapi import Request
from pandas import DataFrame
from sqlalchemy import distinct
from xngin.apiserver import database, exceptionhandlers
from xngin.apiserver.api_types import (
    DataTypeClass,
    AudienceSpec,
    DesignSpec,
    ExperimentAssignment,
    GetStrataResponseElement,
    GetFiltersResponseElement,
    GetMetricsResponseElement,
    PowerAnalysis,
)
from xngin.apiserver.dependencies import (
    httpx_dependency,
    settings_dependency,
    config_dependency,
    gsheet_cache,
)
from xngin.apiserver.dwh.queries import get_stats_on_metrics, query_for_participants
from xngin.apiserver.gsheet_cache import GSheetCache
from xngin.apiserver.settings import (
    ParticipantsMixin,
    WebhookConfig,
    WebhookUrl,
    get_settings_for_server,
    XnginSettings,
    ClientConfig,
    infer_table,
)
from xngin.stats.power import check_power
from xngin.stats.assignment import assign_treatment
from xngin.apiserver.utils import substitute_url
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
exceptionhandlers.setup(app)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
) -> list[GetStrataResponseElement]:
    """
    Get possible strata covariates for a given unit type.

    This reimplements dwh.R get_strata().
    """
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
    design_spec: DesignSpec,
    audience_spec: AudienceSpec,
    gsheets: Annotated[GSheetCache, Depends(gsheet_cache)],
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
) -> PowerAnalysis:
    """
    Calculates statistical power given an AudienceSpec and a DesignSpec
    """
    config = require_config(client)
    participant = config.find_participants(audience_spec.participant_type)

    with config.dbsession() as session:
        sa_table = infer_table(
            session.get_bind(), participant.table_name, config.supports_reflection()
        )

        metric_stats = get_stats_on_metrics(
            session,
            sa_table,
            design_spec.metrics,
            audience_spec,
        )

        return check_power(
            metrics=metric_stats,
            n_arms=len(design_spec.arms),
            power=design_spec.power,
            alpha=design_spec.alpha,
        )


@app.post(
    "/assign",
    summary="Assign treatment given experiment and audience specification.",
    tags=["Manage Experiments"],
)
def assign_treatment_api(
    design_spec: DesignSpec,
    audience_spec: AudienceSpec,
    chosen_n: int,
    random_state: int,
    gsheets: Annotated[GSheetCache, Depends(gsheet_cache)],
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
) -> ExperimentAssignment:
    config = require_config(client)
    participant = config.find_participants(audience_spec.participant_type)
    config_sheet = fetch_worksheet(
        CommonQueryParams(
            participant_type=participant.participant_type, refresh=refresh
        ),
        config,
        gsheets,
    )
    _unique_id_col = config_sheet.get_unique_id_col()

    with config.dbsession() as session:
        sa_table = infer_table(
            session.get_bind(), participant.table_name, config.supports_reflection()
        )
        participants = query_for_participants(
            session, sa_table, audience_spec, chosen_n
        )

    metric_names = [metric.metric_name for metric in design_spec.metrics]
    arm_names = [arm.arm_name for arm in design_spec.arms]
    return assign_treatment(
        data=DataFrame(participants),
        stratum_cols=design_spec.strata_cols,
        metric_cols=metric_names,
        id_col=_unique_id_col,
        arm_names=arm_names,
        experiment_id=str(design_spec.experiment_id),
        description=design_spec.description,
        fstat_thresh=design_spec.fstat_thresh,
        random_state=random_state,
    )


@app.get(
    "/assignment-file",
    summary="Retrieve all participant assignments for the given experiment_id.",
    responses=STANDARD_WEBHOOK_RESPONSES,
    tags=["Manage Experiments"],
)
async def assignment_file(
    response: Response,
    experiment_id: str = Annotated[
        str,
        Query(description="ID of the experiment whose assignments we wish to fetch."),
    ],
    http_client: Annotated[httpx.AsyncClient, Depends(httpx_dependency)] = None,
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
    body: UpdateExperimentStartEndRequest | UpdateExperimentDescriptionsRequest,
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
    """Fetches a map of column name to SheetConfig column metadata.

    Uniqueness of the values in the column unique_id_col is assumed, not verified!
    """
    return {
        c.column_name: c
        for c in create_sheetconfig_from_table(table, unique_id_col).columns
    }


def main():
    database.setup()

    import uvicorn

    # Handy for debugging in your IDE
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("UVICORN_PORT", 8000)))


if __name__ == "__main__":
    main()
