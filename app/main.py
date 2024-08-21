# ruff: noqa: F401

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
from scipy import stats
import psycopg2
from pymongo import MongoClient
import requests
import json
from datetime import datetime, timedelta

app = FastAPI()

# Global variables
type_map = {
    "groups": {
        "trs": "trs_groups",
        "olap": "olap_groups",
        "id": "groups_id",
        "created": "groups_created_at",
        "org_id": "organization_id",
    },
    # ... (other type mappings)
}


# Helper functions
def classify_data_type(filter_name: str, data_type: str) -> str:
    filter_name = filter_name.lower()
    data_type = data_type.lower()

    discrete_types = ["boolean", "character varying"]
    numeric_types = [
        "date",
        "integer",
        "double precision",
        "numeric",
        "timestamp without time zone",
        "bigint",
    ]

    if data_type in discrete_types or filter_name.endswith("_id"):
        return "discrete"
    elif data_type in numeric_types:
        return "numeric"
    else:
        return "unknown"


def get_relations(data_class: str) -> List[str]:
    if data_class == "discrete":
        return ["includes", "excludes"]
    elif data_class == "numeric":
        return ["between"]
    else:
        raise ValueError(f"Unsupported data class: {data_class}")


# Database connection functions
def get_dwh_con():
    return psycopg2.connect(
        user="agency",
        port=5439,
        host="instance.redshift.amazonaws.com",
        password=open("redshift_prod_pw.txt").read().strip(),
        dbname="customer_dwh_prod",
        sslmode="require",
    )


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
def get_strata(type: str = "groups", refresh: bool = False):
    # Implement get_strata logic
    pass


@app.get("/filters")
def get_filters(type: str = "groups", refresh: bool = False):
    # Implement get_filters logic
    pass


@app.get("/metrics")
def get_metrics(type: str = "groups", refresh: bool = False):
    # Implement get_metrics logic
    pass


@app.post("/power")
def check_power(design_spec: DesignSpec, audience_spec: AudienceSpec):
    # Implement power calculation logic
    pass


@app.post("/assign")
def assign_treatment(
    design_spec: DesignSpec, audience_spec: AudienceSpec, chosen_n: int = 1000
):
    # Implement treatment assignment logic
    pass


@app.post("/commit")
def commit_experiment(
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
def experiments_reg_request(endpoint: str, json_data: Dict[str, Any] = None):
    api_host = "api.example.com"
    url = f"https://{api_host}/dev/api/v1/experiment-commit/{endpoint}"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {open(f'{api_host}.token').read().strip()}",
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
