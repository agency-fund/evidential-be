"""Checks that admin routes add the caller and the resources named in the path to the request's log context."""

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.testclient import TestClient

from xngin.apiserver.main import app
from xngin.apiserver.routers.auth.auth_dependencies import PRIVILEGED_EMAIL, PRIVILEGED_TOKEN_FOR_TESTING
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.logs_like_uvicorn import LogsLikeUvicorn


@pytest.fixture
def logging_client(xngin_session):
    headers = {"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"}
    with TestClient(LogsLikeUvicorn(app), headers=headers) as client:
        yield client


def _access_fields(records):
    (access,) = [r for r in records if r["message"] == "access"]
    return access["extra"]


def _caller_id(xngin_session: Session) -> str:
    return xngin_session.scalars(select(tables.User.id).where(tables.User.email == PRIVILEGED_EMAIL)).one()


def test_admin_routes_log_the_caller_and_the_resources_named_in_the_path(
    testing_datasource, logging_client, records, xngin_session: Session
):
    response = logging_client.get(f"/v1/m/datasources/{testing_datasource.datasource_id}")

    assert response.status_code == 200
    access = _access_fields(records)
    assert access["user_id"] == _caller_id(xngin_session)
    assert access["datasource_id"] == testing_datasource.datasource_id
    assert "organization_id" not in access
    assert "experiment_id" not in access


def test_admin_routes_log_path_ids_even_when_the_lookup_fails(
    testing_datasource, logging_client, records, xngin_session: Session
):
    org_id = testing_datasource.organization_id
    ds_id = testing_datasource.datasource_id
    bogus_experiment_id = "x" * 1000

    response = logging_client.get(
        f"/v1/m/organizations/{org_id}/datasources/{ds_id}/experiments/{bogus_experiment_id}/snapshots"
    )

    assert response.status_code == 404
    access = _access_fields(records)
    assert access["user_id"] == _caller_id(xngin_session)
    assert access["organization_id"] == org_id
    assert access["datasource_id"] == ds_id
    assert access["experiment_id"] == "x" * 64
