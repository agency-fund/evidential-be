import enum
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Annotated, Optional
import requests
from datetime import datetime

from app.dependencies import settings_dependency, dwh_dependency, Dwh
from app.settings import get_settings_for_server, XnginSettings
from fastapi import Request
from app.utils import safe_for_headers


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


class DataType(enum.StrEnum):
    BOOLEAN = "boolean"
    CHARACTER_VARYING = "character varying"
    DATE = "date"
    INTEGER = "integer"
    DOUBLE_PRECISION = "double precision"
    NUMERIC = "numeric"
    TIMESTAMP_WITHOUT_TIMEZONE = "timestamp without time zone"
    BIGINT = "bigint"


class DataTypeClass(enum.StrEnum):
    DISCRETE = "discrete"
    NUMERIC = "numeric"
    UNKNOWN = "unknown"


DISCRETE_TYPES = [DataType.BOOLEAN, DataType.CHARACTER_VARYING]
NUMERIC_TYPES = [
    DataType.DATE,
    DataType.INTEGER,
    DataType.DOUBLE_PRECISION,
    DataType.NUMERIC,
    DataType.TIMESTAMP_WITHOUT_TIMEZONE,
    DataType.BIGINT,
]


class Relation(enum.StrEnum):
    INCLUDES = "includes"
    EXCLUDES = "excludes"
    BETWEEN = "between"


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


# API Models
class AudienceSpec(BaseModel):
    type: str
    filters: List[Dict[str, Any]]


class DesignSpec(BaseModel):
    experiment_id: str
    experiment_name: str
    description: str
    arms: List[Dict[str, str]]
    start_date: datetime
    end_date: datetime
    strata_cols: List[str]
    power: float
    alpha: float
    fstat_thresh: float
    metrics: List[Dict[str, Any]]


# API Endpoints
@app.get("/strata")
def get_strata(
    dwh: Annotated[Dwh, Depends(dwh_dependency)],
    type: str = "groups",
    refresh: bool = False,
):
    # Implement get_strata logic
    pass


@app.get("/filters")
def get_filters(
    dwh: Annotated[Dwh, Depends(dwh_dependency)],
    type: str = "groups",
    refresh: bool = False,
):
    # Implement get_filters logic
    pass


@app.get("/metrics")
def get_metrics(
    dwh: Annotated[Dwh, Depends(dwh_dependency)],
    type: str = "groups",
    refresh: bool = False,
):
    # Implement get_metrics logic
    pass


@app.post("/power")
def check_power(
    dwh: Annotated[Dwh, Depends(dwh_dependency)],
    design_spec: DesignSpec,
    audience_spec: AudienceSpec,
):
    # Implement power calculation logic
    pass


@app.post("/assign")
def assign_treatment(
    dwh: Annotated[Dwh, Depends(dwh_dependency)],
    design_spec: DesignSpec,
    audience_spec: AudienceSpec,
    chosen_n: int = 1000,
):
    # Implement treatment assignment logic
    pass


@app.get("/_settings")
def debug_settings(
    request: Request,
    settings: Annotated[XnginSettings, Depends(settings_dependency)],
):
    if request.client.host in settings.trusted_ips:
        return {"settings": settings}
    raise HTTPException(403)


@app.post("/commit")
def commit_experiment(
    dwh: Annotated[Dwh, Depends(dwh_dependency)],
    design_spec: DesignSpec,
    audience_spec: AudienceSpec,
    experiment_assignment: Dict[str, Any],
    user_id: str = "testuser",
):
    # Implement experiment commit logic
    pass


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
    settings: XnginSettings, endpoint: str, json_data: Dict[str, Any] = None
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
