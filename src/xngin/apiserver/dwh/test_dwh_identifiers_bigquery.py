"""Tests of the table, column and RECORD sub-field names a data source can use, through the admin API, against a
real BigQuery dataset.

They are marked integration and need XNGIN_DEVDWH_GCP_CREDENTIALS.
"""

import os
import uuid
from collections.abc import Iterator
from dataclasses import astuple

import pytest
from google.cloud import bigquery
from google.oauth2 import service_account

from xngin.apiserver.settings import BqDsn, GcpServiceAccountFile
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.dwh_identifiers_helpers import (
    BQ_QUOTED_COLUMNS,
    PLAIN_COLUMNS,
    QUOTED_COLUMNS_TABLE,
    ROWS,
    TABLE_NAMES,
    Columns,
    exercise_through_api,
    expect_unusable_table_name,
)
from xngin.apiserver.testing.helpers import GCP_CREDENTIALS_VAR, skip_unless_gcp_credentials

pytestmark = [pytest.mark.integration, skip_unless_gcp_credentials]

BQ_TABLE_NAMES = [*TABLE_NAMES, "tëst 名前"]
# A table whose metric, segment and cluster are sub-fields of a RECORD column, outer_rec. The BigQuery dialect
# reflects the RECORD as a STRUCT column followed by one column per sub-field, named outer_rec.metric and so on.
# It also has a REPEATED RECORD column, events, which is reflected as one ARRAY column and is not offered as fields.
NESTED_TABLE = "nested"
NESTED_COLUMNS = Columns(id="id", metric="outer_rec.metric", segment="outer_rec.segment", cluster="outer_rec.cluster")
BQ_LOCATION = "us-west1"


def _bq_create_table_sql(qualified: str, columns: Columns, rows: list[tuple] = ROWS) -> str:
    column_types = zip(astuple(columns), ("INT64", "FLOAT64", "STRING", "STRING"), strict=True)
    values = ", ".join(f"({i}, {m}, '{s}', '{c}')" for i, m, s, c in rows)
    return (
        f"CREATE TABLE {qualified} ({', '.join(f'`{c}` {t}' for c, t in column_types)}); "
        f"INSERT INTO {qualified} VALUES {values};"
    )


def _bq_create_nested_table_sql(qualified: str, rows: list[tuple]) -> str:
    values = ", ".join(f"STRUCT({i}, STRUCT({m}, '{s}', '{c}'), [STRUCT('event')])" for i, m, s, c in rows)
    return (
        f"CREATE TABLE {qualified} AS SELECT * FROM UNNEST(ARRAY<STRUCT<id INT64, "
        f"outer_rec STRUCT<metric FLOAT64, segment STRING, cluster STRING>, events ARRAY<STRUCT<name STRING>>>>"
        f"[{values}]);"
    )


@pytest.fixture(name="bq_datasets", scope="module")
def fixture_bq_datasets() -> Iterator[tuple[BqDsn, str]]:
    """A BqDsn for a new dataset, and a second dataset's name.

    The dataset holds a table for each of BQ_TABLE_NAMES, QUOTED_COLUMNS_TABLE with BQ_QUOTED_COLUMNS, and
    NESTED_TABLE. The second dataset holds one table, elsewhere,
    which a qualified table name must not reach.
    """
    path = os.environ[GCP_CREDENTIALS_VAR]
    credentials = service_account.Credentials.from_service_account_file(path)
    suffix = uuid.uuid4().hex[:12]
    configured, other = f"tablenames_{suffix}", f"tablenames_{suffix}_other"
    with bigquery.Client(project=credentials.project_id, credentials=credentials, location=BQ_LOCATION) as client:
        datasets = []
        try:
            for dataset_id in (configured, other):
                dataset = bigquery.Dataset(f"{client.project}.{dataset_id}")
                dataset.location = BQ_LOCATION
                # A safety net in case teardown never runs.
                dataset.default_table_expiration_ms = 24 * 60 * 60 * 1000
                datasets.append(client.create_dataset(dataset))
            tables = [(configured, name, PLAIN_COLUMNS) for name in BQ_TABLE_NAMES]
            tables += [(configured, QUOTED_COLUMNS_TABLE, BQ_QUOTED_COLUMNS), (other, "elsewhere", PLAIN_COLUMNS)]
            script = "".join(
                _bq_create_table_sql(f"`{client.project}.{dataset_id}.{name}`", columns)
                for dataset_id, name, columns in tables
            )
            script += _bq_create_nested_table_sql(f"`{client.project}.{configured}.{NESTED_TABLE}`", ROWS)
            client.query_and_wait(script)
            yield (
                BqDsn(
                    driver="bigquery",
                    project_id=client.project,
                    dataset_id=configured,
                    credentials=GcpServiceAccountFile(type="serviceaccountfile", path=path),
                ),
                other,
            )
        finally:
            for dataset in datasets:
                client.delete_dataset(dataset, delete_contents=True, not_found_ok=True)


# Each test creates its own data source, but they share the dataset, so that BigQuery's slow table creation happens
# once per module.


@pytest.mark.parametrize("table_name", BQ_TABLE_NAMES)
def test_bigquery_reads_tables_whose_names_need_quoting(
    aclient: AdminAPIClient, bq_datasets: tuple[BqDsn, str], table_name: str
):
    dsn, _ = bq_datasets
    exercise_through_api(aclient, dsn, table_name)


def test_bigquery_reads_columns_whose_names_need_quoting(aclient: AdminAPIClient, bq_datasets: tuple[BqDsn, str]):
    dsn, _ = bq_datasets
    exercise_through_api(aclient, dsn, QUOTED_COLUMNS_TABLE, BQ_QUOTED_COLUMNS, exact_fields=True)


def test_bigquery_refuses_dataset_qualified_table_names(aclient: AdminAPIClient, bq_datasets: tuple[BqDsn, str]):
    """Unrefused, the two-part name reflected and then read the other dataset's table."""
    dsn, other = bq_datasets
    for table_name in (f"{other}.elsewhere", f"{dsn.project_id}.{other}.elsewhere"):
        expect_unusable_table_name(aclient, dsn, table_name, "not a valid BigQuery table name")


@pytest.mark.parametrize(
    "table_name",
    [
        pytest.param("plain_t` UNION ALL SELECT 1 FROM `plain_t", id="backtick"),
        pytest.param("plain_t\\`", id="backslash"),
        pytest.param("plain_t\nx", id="newline"),
        pytest.param("plain_t\x00x", id="nul"),
        pytest.param("plain_t*", id="wildcard"),
    ],
)
def test_bigquery_refuses_names_bigquery_cannot_create(
    aclient: AdminAPIClient, bq_datasets: tuple[BqDsn, str], table_name: str
):
    dsn, _ = bq_datasets
    expect_unusable_table_name(aclient, dsn, table_name, "not a valid BigQuery table name")
