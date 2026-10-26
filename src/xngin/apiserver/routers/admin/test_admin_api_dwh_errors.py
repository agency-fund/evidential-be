"""Tests of how the admin API reports data warehouse failures.

These create datasources against the test warehouse with credentials that are wrong or under-privileged, so the
errors are the real driver's rather than synthetic ones.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from xngin.apiserver import flags
from xngin.apiserver.routers.admin.admin_api import HTTPExceptionError
from xngin.apiserver.routers.admin.admin_api_types import CreateDatasourceRequest, CreateOrganizationRequest
from xngin.apiserver.routers.common_api_types import (
    Arm,
    CreateExperimentRequest,
    DesignSpecMetricRequest,
    OnlineFrequentistExperimentSpec,
    PowerRequest,
)
from xngin.apiserver.routers.common_enums import ExperimentsType
from xngin.apiserver.settings import Dsn
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.assertions import expect_status_code
from xngin.apiserver.testing.helpers import convert_dwh_to_create_api_dsn
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_TABLE_NAME


def _create_datasource(aclient: AdminAPIClient, dsn: Dsn, name: str, *, connectivity_check: bool = False) -> str:
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name=name)).data.id
    return aclient.create_datasource(
        body=CreateDatasourceRequest(organization_id=org_id, name=name, dsn=convert_dwh_to_create_api_dsn(dsn)),
        connectivity_check=connectivity_check,
    ).data.id


def _power_request() -> PowerRequest:
    return PowerRequest(
        table_name=TESTING_DWH_TABLE_NAME,
        metrics=[DesignSpecMetricRequest(field_name="current_income", metric_pct_change=0.1)],
        n_arms=2,
    )


def test_power_check_reports_a_role_without_select(restricted_dsn: Dsn, aclient: AdminAPIClient):
    ds_id = _create_datasource(aclient, restricted_dsn, "power check without select")

    with expect_status_code(424, code_eq="dwh_permission_denied") as match:
        aclient.power_check(datasource_id=ds_id, body=_power_request())
    body = match.http_response().json()
    assert body["table_name"] == TESTING_DWH_TABLE_NAME
    assert body["hint_command"] == f"GRANT SELECT ON <schema>.{TESTING_DWH_TABLE_NAME} TO {restricted_dsn.user};"
    assert "SELECT " not in body["message"]
    assert "[SQL" not in body["message"]


@pytest.mark.parametrize(
    "request_the_table",
    [
        pytest.param(
            lambda aclient, ds_id: aclient.inspect_table_in_datasource(
                datasource_id=ds_id, table_name=TESTING_DWH_TABLE_NAME
            ),
            id="inspect_table",
        ),
        pytest.param(
            lambda aclient, ds_id: aclient.power_check(datasource_id=ds_id, body=_power_request()), id="power"
        ),
    ],
)
def test_a_wrong_password_is_reported_as_an_authentication_error(
    aclient: AdminAPIClient, request_the_table: Callable[[AdminAPIClient, str], object]
):
    """These routes used to answer 500 through the application database's OperationalError handler."""
    dsn = Dsn.from_url(flags.XNGIN_DEVDWH_DSN).model_copy(update={"password": "definitely-wrong"})
    ds_id = _create_datasource(aclient, dsn, "wrong password")

    with expect_status_code(424, message_contains="password authentication failed", code_eq="dwh_authentication"):
        request_the_table(aclient, ds_id)


def _create_experiment_request() -> CreateExperimentRequest:
    return CreateExperimentRequest(
        design_spec=OnlineFrequentistExperimentSpec(
            experiment_type=ExperimentsType.FREQ_ONLINE,
            experiment_name="missing resource",
            description="missing resource",
            table_name=TESTING_DWH_TABLE_NAME,
            primary_key="id",
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC) + timedelta(days=1),
            arms=[Arm(arm_name="C", arm_description="C"), Arm(arm_name="T", arm_description="T")],
            metrics=[DesignSpecMetricRequest(field_name="is_engaged", metric_pct_change=0.1)],
            strata=[],
            filters=[],
        )
    )


@pytest.mark.parametrize(
    "request_it",
    [
        pytest.param(lambda aclient, ds_id: aclient.inspect_datasource(datasource_id=ds_id), id="inspect_datasource"),
        pytest.param(
            lambda aclient, ds_id: aclient.inspect_table_in_datasource(
                datasource_id=ds_id, table_name=TESTING_DWH_TABLE_NAME
            ),
            id="inspect_table",
        ),
        pytest.param(
            lambda aclient, ds_id: aclient.power_check(datasource_id=ds_id, body=_power_request()), id="power_check"
        ),
        pytest.param(
            lambda aclient, ds_id: aclient.create_experiment(datasource_id=ds_id, body=_create_experiment_request()),
            id="create_experiment",
        ),
        pytest.param(
            lambda aclient, ds_id: aclient.analyze_experiment(datasource_id=ds_id, experiment_id="exp_missing"),
            id="analyze_experiment",
        ),
    ],
)
def test_a_missing_datasource_is_still_an_ordinary_404(
    aclient: AdminAPIClient, request_it: Callable[[AdminAPIClient, str], object]
):
    """Routes that declare DwhErrorResponse for 404 still answer their dependencies' HTTPException 404s.

    The typed client validates the body against the declared 404 models, so this fails if HTTPExceptionError
    is dropped from the union.
    """
    with expect_status_code(404, text="Datasource not found.") as match:
        request_it(aclient, "ds_missing")
    assert match.exc is not None
    assert isinstance(match.exc.result.data, HTTPExceptionError)


def test_a_missing_experiment_is_still_an_ordinary_404(aclient: AdminAPIClient):
    ds_id = _create_datasource(aclient, Dsn.from_url(flags.XNGIN_DEVDWH_DSN), "missing experiment")

    with expect_status_code(404, text="Experiment not found.") as match:
        aclient.analyze_experiment(datasource_id=ds_id, experiment_id="exp_missing")
    assert match.exc is not None
    assert isinstance(match.exc.result.data, HTTPExceptionError)


def test_a_search_path_data_source_inspects_the_tables_in_its_schema(dwh_schema: str, aclient: AdminAPIClient):
    dsn = Dsn.from_url(flags.XNGIN_DEVDWH_DSN).model_copy(update={"search_path": dwh_schema})
    ds_id = _create_datasource(aclient, dsn, "search_path", connectivity_check=True)

    assert aclient.inspect_datasource(datasource_id=ds_id).data.tables == ["participants"]
    fields = aclient.inspect_table_in_datasource(datasource_id=ds_id, table_name="participants").data.fields
    assert {field.field_name for field in fields} == {"id", "is_engaged"}
