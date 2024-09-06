from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Annotated, Optional, Literal
import requests

from xngin.apiserver.api_types import (
    DataType,
    DataTypeClass,
    Relation,
    AudienceSpec,
    DesignSpec,
    UnimplementedResponse,
)
from xngin.apiserver.dependencies import settings_dependency, dwh_dependency, Dwh
from xngin.apiserver.settings import get_settings_for_server, XnginSettings
from fastapi import Request
from xngin.apiserver.utils import safe_for_headers


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_settings_for_server()
    yield


app = FastAPI(lifespan=lifespan)


class DwhFieldConfig(BaseModel):
    created: Optional[str] = None
    id: str
    olap: Optional[str] = None
    org_id: Optional[str] = None
    trs: str


# TODO: This appears to be customer-specific; move to config?
DWH_FIELD_MAP = {
    "groups": DwhFieldConfig(
        created="groups_created_at",
        id="groups_id",
        olap="olap_groups",
        org_id="organization_id",
        trs="trs_groups",
    ),
    "organizations": DwhFieldConfig(
        created="org_created_at",
        id="organization_id",
        olap="olap_organizations",
        org_id="organization_id",
        trs="organizations",
    ),
    "phones": DwhFieldConfig(
        created="guardians_created_at",
        id="guardian_id",
        olap="olap_phone",
        trs="trs_phones",
    ),
    "schools": DwhFieldConfig(
        created="schools_created_at",
        id="school_id",
        olap="olap_school",
        org_id="organizations_id",
        trs="trs_schools",
    ),
    "moderators": DwhFieldConfig(
        created="mod_created_at",
        id="moderator_id",
        org_id="organization_id",
        trs="trs_moderators",
    ),
    "kids": DwhFieldConfig(
        id="kids",
        trs="trs_kids",
    ),
    "guardians": DwhFieldConfig(
        created="guardians_created_at",
        id="guardian_id",
        trs="trs_guardians",
    ),
}

type UnitType = Literal[tuple(sorted(DWH_FIELD_MAP.keys()))]

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


async def common_parameters(
    type: Annotated[
        UnitType,
        Query(description="Type of unit to derive strata from."),
    ] = "groups",
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
):
    """Defines parameters common to the GET methods."""
    return {"type": type, "refresh": refresh}


# API Endpoints
@app.get(
    "/strata",
    summary="Get possible strata covariates.",
    response_model=UnimplementedResponse,
)
def get_strata(
    commons: Annotated[dict, Depends(common_parameters)],
    dwh: Annotated[Dwh, Depends(dwh_dependency)] = None,
):
    """
    Get possible strata covariates for a given unit type.
    """
    # Implement get_strata logic
    return UnimplementedResponse()


@app.get(
    "/filters",
    summary="Get possible filters covariates for a given unit type.",
    response_model=UnimplementedResponse,
)
def get_filters(
    dwh: Annotated[Dwh, Depends(dwh_dependency)],
    commons: Annotated[dict, Depends(common_parameters)],
):
    # Implement get_filters logic
    return UnimplementedResponse()


@app.get(
    "/metrics",
    summary="Get possible metric covariates for a given unit type.",
    response_model=UnimplementedResponse,
)
def get_metrics(
    dwh: Annotated[Dwh, Depends(dwh_dependency)],
    commons: Annotated[dict, Depends(common_parameters)],
):
    # Implement get_metrics logic
    return UnimplementedResponse()


@app.post(
    "/power",
    summary="Check power given an experiment and audience specification.",
    response_model=UnimplementedResponse,
)
def check_power(
    dwh: Annotated[Dwh, Depends(dwh_dependency)],
    design_spec: DesignSpec,
    audience_spec: AudienceSpec,
):
    # Implement power calculation logic
    return UnimplementedResponse()


@app.post(
    "/assign",
    summary="Assign treatment given experiment and audience specification.",
    response_model=UnimplementedResponse,
)
def assign_treatment(
    dwh: Annotated[Dwh, Depends(dwh_dependency)],
    design_spec: DesignSpec,
    audience_spec: AudienceSpec,
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
    dwh: Annotated[Dwh, Depends(dwh_dependency)],
    design_spec: DesignSpec,
    audience_spec: AudienceSpec,
    experiment_assignment: Dict[str, Any],
    user_id: str = "testuser",
):
    # Implement experiment commit logic
    return UnimplementedResponse()


@app.get("/_settings", include_in_schema=False)
def debug_settings(
    request: Request,
    settings: Annotated[XnginSettings, Depends(settings_dependency)],
):
    if request.client.host in settings.trusted_ips:
        return {"settings": settings}
    raise HTTPException(403)


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

    api_token = safe_for_headers(settings.customer.api_token.get_secret_value())
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
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
