from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query
from typing import List, Dict, Any, Annotated
import requests
from sqlalchemy.exc import NoSuchTableError

from xngin.apiserver import database
from xngin.apiserver.api_types import (
    DataType,
    DataTypeClass,
    Relation,
    AudienceSpec,
    DesignSpec,
    UnimplementedResponse,
    GetStrataResponseElement,
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
    get_sqlalchemy_table,
    CannotFindTheTableException,
)
from fastapi import Request
from xngin.apiserver.utils import safe_for_headers
from xngin.sheets.config_sheet import (
    fetch_and_parse_sheet,
    create_sheetconfig_from_table,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_settings_for_server()
    database.setup()
    yield


app = FastAPI(lifespan=lifespan)

DISCRETE_TYPES = [DataType.BOOLEAN, DataType.CHARACTER_VARYING]
NUMERIC_TYPES = [
    DataType.DATE,
    DataType.INTEGER,
    DataType.DOUBLE_PRECISION,
    DataType.NUMERIC,
    DataType.TIMESTAMP_WITHOUT_TIMEZONE,
    DataType.BIGINT,
]


# Helper functions
def classify_data_type(filter_name: str, data_type: str):
    filter_name = filter_name.lower()
    data_type = data_type.lower()

    if data_type in DISCRETE_TYPES or filter_name.endswith("_id"):
        return DataTypeClass.DISCRETE
    elif data_type in NUMERIC_TYPES:
        return DataTypeClass.NUMERIC
    else:
        return DataTypeClass.UNKNOWN


def get_relations(data_class: DataTypeClass):
    match data_class:
        case DataTypeClass.DISCRETE:
            return [Relation.INCLUDES, Relation.EXCLUDES]
        case DataTypeClass.NUMERIC:
            return [Relation.BETWEEN]
        case _:
            raise ValueError(f"Unsupported data class: {data_class}")


class CommonQueryParams:
    """Describes query parameters common to the /strata, /filters, and /metrics APIs."""

    def __init__(
        self,
        group: Annotated[str, Query(description="Column group to derive strata from.")],
        refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
    ):
        self.group = group
        self.refresh = refresh


# API Endpoints
@app.get(
    "/strata",
    summary="Get possible strata covariates.",
)
def get_strata(
    commons: Annotated[CommonQueryParams, Depends()],
    gsheet_cache: Annotated[GSheetCache, Depends(gsheet_cache)],
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
):
    """
    Get possible strata covariates for a given unit type.

    This reimplements dwh.R get_strata().
    """
    if not client:
        raise HTTPException(
            404, "Configuration for the requested client was not found."
        )
    config = client.config

    # TODO: determine if the RL behavior should be ported
    if commons.group != config.table_name:
        raise HTTPException(400, "group parameter must match configured table name.")

    try:
        table = get_sqlalchemy_table(config.to_sqlalchemy_url_and_table())
    except NoSuchTableError as nste:
        raise HTTPException(
            status_code=500,
            detail=f"The configured table '{config.table_name}' does not exist.",
        ) from nste
    try:
        db_schema = {
            c.column_name: c for c in create_sheetconfig_from_table(table).rows
        }
    except CannotFindTheTableException as cfte:
        raise HTTPException(status_code=500, detail=cfte.message) from cfte

    fetched = gsheet_cache.get(
        config.sheet,
        lambda: fetch_and_parse_sheet(config.sheet),
        refresh=commons.refresh,
    )
    config_schema = {
        c.column_name: c
        for c in fetched.rows
        if c.table == config.table_name and c.is_strata
    }
    return sorted(
        [
            GetStrataResponseElement(
                table_name=config.table_name,
                data_type=db_schema.get(col_name).data_type,
                column_name=col_name,
                description=db_schema.get(col_name).description,
                strata_group=config_col.column_group,
            )
            for col_name, config_col in config_schema.items()
            if db_schema.get(col_name)
        ],
        key=lambda item: item.column_name,
    )


def rl_get_col_names(type):
    if type == "groups":
        return ["final_groups"]
    else:
        return [f"trs_{type}", f"olap_{type}"]


def get_strata_impl(type: str, refresh: bool):
    # TODO
    return []
    # table_names = rl_get_col_names(type)


@app.get(
    "/filters",
    summary="Get possible filters covariates for a given unit type.",
    response_model=UnimplementedResponse,
)
def get_filters(
    commons: Annotated[CommonQueryParams, Depends()],
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
):
    # Implement get_filters logic
    return UnimplementedResponse()


@app.get(
    "/metrics",
    summary="Get possible metric covariates for a given unit type.",
    response_model=UnimplementedResponse,
)
def get_metrics(
    commons: Annotated[CommonQueryParams, Depends()],
    client: Annotated[ClientConfig | None, Depends(config_dependency)] = None,
):
    # Implement get_metrics logic
    return UnimplementedResponse()


@app.post(
    "/power",
    summary="Check power given an experiment and audience specification.",
    response_model=UnimplementedResponse,
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
    "/commit",
    summary="Commit an experiment to the database.",
    response_model=UnimplementedResponse,
)
def commit_experiment(
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
        response = requests.get(url, headers=headers)
    else:
        if endpoint.startswith("update"):
            response = requests.put(url, headers=headers, json=json_data)
        else:
            response = requests.post(url, headers=headers, json=json_data)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Request failed")

    return response.json()


def main():
    database.setup()

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
