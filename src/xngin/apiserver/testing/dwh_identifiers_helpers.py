"""Tables, columns and checks shared by the tests of the table, schema and column names data sources can use.

See xngin/apiserver/dwh/test_dwh_identifiers*.py.
"""

import datetime as dt
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import astuple, dataclass

import pytest
import sqlalchemy
from sqlalchemy import text

from xngin.apiserver import flags
from xngin.apiserver.dwh import dwh_utils
from xngin.apiserver.routers.admin.admin_api_types import CreateDatasourceRequest, CreateOrganizationRequest
from xngin.apiserver.routers.common_api_types import (
    Arm,
    CreateExperimentRequest,
    DesignSpecMetricRequest,
    Filter,
    FreqExperimentAnalysisResponse,
    PowerRequest,
    PreassignedFrequentistExperimentSpec,
    Stratum,
)
from xngin.apiserver.routers.common_enums import Relation
from xngin.apiserver.settings import BqDsn, Dsn
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.assertions import expect_status_code
from xngin.apiserver.testing.helpers import convert_dwh_to_create_api_dsn
from xngin.apiserver.testing.pg_helpers import with_name_suffix

# Names that need quoting in SQL, beside one that does not. Postgres and Redshift store unquoted names in lowercase,
# so MixedCase only resolves when it is quoted.
TABLE_NAMES = ["plain_t", "MixedCase", "has space", "dash-name"]
# Postgres and Redshift also allow a dot in a quoted table name. BigQuery does not; there, a dot names a dataset.
PG_TABLE_NAMES = [*TABLE_NAMES, "dot.ted"]


@dataclass(frozen=True)
class Columns:
    """The names of a test table's columns: a participant id, a metric, a segment to filter on, and a cluster."""

    id: str = "id"
    metric: str = "metric"
    segment: str = "segment"
    cluster: str = "cluster"


PLAIN_COLUMNS = Columns()
# Every TABLE_NAMES table has PLAIN_COLUMNS. One more table, QUOTED_COLUMNS_TABLE, has column names that need quoting:
# dots, spaces, dashes and mixed case. BigQuery does not allow a dot in a column name; there, a dot reaches into a
# RECORD (see NESTED_TABLE).
QUOTED_COLUMNS_TABLE = "quoted_columns"
PG_QUOTED_COLUMNS = Columns(id="participant.id", metric="My Metric", segment="seg-ment", cluster="Clu.ster")
BQ_QUOTED_COLUMNS = Columns(id="Participant ID", metric="my-metric", segment="Seg Ment", cluster="clu-ster")
# (id, metric, segment, cluster) for every test table: enough rows for an experiment's balance check, in four clusters
# of ten.
ROWS = [(i, float(i % 7), "ab"[i % 2], f"c{i % 4}") for i in range(1, 41)]


def create_datasource(aclient: AdminAPIClient, dwh_config: Dsn | BqDsn) -> str:
    """Creates an organization with a data source for dwh_config, returning the data source's ID."""
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="identifiers")).data.id
    return aclient.create_datasource(
        body=CreateDatasourceRequest(
            organization_id=org_id, name="identifiers", dsn=convert_dwh_to_create_api_dsn(dwh_config)
        )
    ).data.id


def _experiment_request(**design: object) -> CreateExperimentRequest:
    """A preassigned experiment with two arms and the given design."""
    now = dt.datetime.now(dt.UTC)
    return CreateExperimentRequest(
        design_spec=PreassignedFrequentistExperimentSpec(
            experiment_name="identifiers",
            description="Names that need quoting",
            start_date=now,
            end_date=now + dt.timedelta(days=7),
            arms=[Arm(arm_name="control", arm_description="Control"), Arm(arm_name="treatment", arm_description="T")],
            **design,
        )
    )


def exercise_through_api(
    aclient: AdminAPIClient,
    dwh_config: Dsn | BqDsn,
    table_name: str,
    columns: Columns = PLAIN_COLUMNS,
    *,
    listed_as: str | None = None,
    exact_fields: bool = False,
) -> None:
    """Runs every read of table_name a customer can cause, through the admin API, using columns for every field.

    Covers listing and inspecting the table, power checks with filters and with clusters, an individual- and a
    cluster-randomized experiment, analysis of each, and the assignments CSV. listed_as is the name the warehouse
    lists the table under, when that differs from table_name. With exact_fields, inspection must offer exactly columns.
    """
    datasource_id = create_datasource(aclient, dwh_config)
    assert (listed_as or table_name) in aclient.inspect_datasource(datasource_id=datasource_id).data.tables

    inspected = aclient.inspect_table_in_datasource(datasource_id=datasource_id, table_name=table_name).data
    offered = {f.field_name for f in inspected.fields}
    if exact_fields:
        assert offered == set(astuple(columns))
    else:
        assert set(astuple(columns)) <= offered

    metric = DesignSpecMetricRequest(field_name=columns.metric, metric_pct_change=0.5)
    filters = [
        Filter(field_name=columns.segment, relation=Relation.INCLUDES, value=["a"]),
        Filter(field_name=columns.metric, relation=Relation.BETWEEN, value=[1, 100]),
    ]
    matching = [row for row in ROWS if row[2] == "a" and row[1] >= 1]
    power = aclient.power_check(
        datasource_id=datasource_id,
        body=PowerRequest(table_name=table_name, metrics=[metric], filters=filters, n_arms=2),
    ).data
    assert power.analyses[0].metric_spec.available_n == len(matching)
    clustered_power = aclient.power_check(
        datasource_id=datasource_id,
        body=PowerRequest(table_name=table_name, metrics=[metric], filters=[], cluster_key=columns.cluster, n_arms=2),
    ).data
    assert clustered_power.analyses[0].metric_spec.avg_cluster_size == pytest.approx(len(ROWS) / 4)

    created = aclient.create_experiment(
        datasource_id=datasource_id,
        body=_experiment_request(
            table_name=table_name,
            primary_key=columns.id,
            metrics=[metric],
            filters=filters,
            strata=[Stratum(field_name=columns.cluster)],
            desired_n=len(matching),
        ),
        random_state=42,
    ).data
    assert created.assign_summary is not None
    assert created.assign_summary.sample_size == len(matching)

    analysis = aclient.analyze_experiment(datasource_id=datasource_id, experiment_id=created.experiment_id).data
    assert isinstance(analysis, FreqExperimentAnalysisResponse)
    assert [m.metric_name for m in analysis.metric_analyses] == [columns.metric]
    assert analysis.num_participants == len(matching)

    # The generated client does not handle CSV responses.
    csv_response = aclient.client.get(
        f"/v1/m/datasources/{datasource_id}/experiments/{created.experiment_id}/assignments/csv"
    )
    assert csv_response.status_code == 200, csv_response.content
    assert columns.cluster in csv_response.text.splitlines()[0]

    # Every participant in two of the four clusters.
    clustered = aclient.create_experiment(
        datasource_id=datasource_id,
        body=_experiment_request(
            table_name=table_name,
            primary_key=columns.id,
            metrics=[metric],
            filters=[],
            strata=[],
            cluster_key=columns.cluster,
            desired_n_clusters=2,
        ),
        random_state=42,
    ).data
    assert clustered.assign_summary is not None
    assert clustered.assign_summary.sample_size == len(ROWS) // 2
    clustered_analysis = aclient.analyze_experiment(
        datasource_id=datasource_id, experiment_id=clustered.experiment_id
    ).data
    assert isinstance(clustered_analysis, FreqExperimentAnalysisResponse)
    assert [m.metric_name for m in clustered_analysis.metric_analyses] == [columns.metric]


def expect_unusable_table_name(aclient: AdminAPIClient, dwh_config: Dsn | BqDsn, table_name: str, message: str) -> None:
    """Checks that a power check on table_name fails with a 404 whose message contains message.

    The table name travels in the request body, which can carry any character, unlike a URL path.
    """
    datasource_id = create_datasource(aclient, dwh_config)
    metric = DesignSpecMetricRequest(field_name=PLAIN_COLUMNS.metric, metric_pct_change=0.5)
    with expect_status_code(404, message_contains=message):
        aclient.power_check(
            datasource_id=datasource_id,
            body=PowerRequest(table_name=table_name, metrics=[metric], filters=[], n_arms=2),
        )


# Tests are parametrized by these names, and use them with with_name_suffix() applied: the testing warehouse is shared
# by every pytest-xdist worker and test run, and test IDs must be the same on every worker.
PG_SCHEMAS = ["dwh_tn_plain", "Mixed Schema-Name"]
PG_OTHER_SCHEMA = "dwh_tn_other"


def pg_quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def pg_create_table(
    conn: sqlalchemy.Connection, schema: str, name: str, columns: Columns, rows: list[tuple] = ROWS
) -> None:
    qualified = f"{pg_quote(schema)}.{pg_quote(name)}"
    column_types = zip(astuple(columns), ("int", "float8", "text", "text"), strict=True)
    conn.execute(text(f"CREATE TABLE {qualified} ({', '.join(f'{pg_quote(c)} {t}' for c, t in column_types)})"))
    conn.execute(
        text(f"INSERT INTO {qualified} VALUES (:id, :metric, :segment, :cluster)"),  # noqa: S608
        [dict(zip(("id", "metric", "segment", "cluster"), row, strict=True)) for row in rows],
    )


def pg_dsn(driver: str, search_path: str | None) -> Dsn:
    dsn: Dsn = Dsn.from_url(flags.XNGIN_DEVDWH_DSN)
    return dsn.model_copy(update={"driver": driver, "search_path": search_path})


@contextmanager
def pg_identifier_schemas(dsn: Dsn | None = None) -> Generator[None]:
    """Each of PG_SCHEMAS holds a table for each of PG_TABLE_NAMES, and QUOTED_COLUMNS_TABLE with PG_QUOTED_COLUMNS.

    PG_OTHER_SCHEMA holds one table, elsewhere. The schemas are created in
    the local testing warehouse, or in the Postgres or Redshift database dsn names.
    """
    admin = sqlalchemy.create_engine((dsn or Dsn.from_url(flags.XNGIN_DEVDWH_DSN)).to_sqlalchemy_url())
    dwh_utils.extra_engine_setup(admin)
    schemas = [with_name_suffix(name) for name in (*PG_SCHEMAS, PG_OTHER_SCHEMA)]
    try:
        with admin.begin() as conn:
            for schema in schemas:
                conn.execute(text(f"DROP SCHEMA IF EXISTS {pg_quote(schema)} CASCADE"))
                conn.execute(text(f"CREATE SCHEMA {pg_quote(schema)}"))
            for schema in PG_SCHEMAS:
                for name in PG_TABLE_NAMES:
                    pg_create_table(conn, with_name_suffix(schema), name, PLAIN_COLUMNS)
                pg_create_table(conn, with_name_suffix(schema), QUOTED_COLUMNS_TABLE, PG_QUOTED_COLUMNS)
            pg_create_table(conn, with_name_suffix(PG_OTHER_SCHEMA), "elsewhere", PLAIN_COLUMNS)
        yield
    finally:
        with admin.begin() as conn:
            for schema in schemas:
                conn.execute(text(f"DROP SCHEMA IF EXISTS {pg_quote(schema)} CASCADE"))
        admin.dispose()
