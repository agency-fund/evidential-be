"""Tests of create-testing-dwh against a real BigQuery dataset. They are marked integration and need
XNGIN_DEVDWH_GCP_CREDENTIALS.
"""

import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from google.cloud import bigquery
from google.oauth2 import service_account

from xngin.apiserver.testing import stress_dwh_def
from xngin.apiserver.testing.helpers import GCP_CREDENTIALS_VAR, skip_unless_gcp_credentials
from xngin.cli.commands.create_testing_dwh import (
    create_testing_dwh,
)

TESTDATA = Path(__file__).resolve().parent.parent.parent / "apiserver/testdata"


@pytest.fixture(name="bigquery_dataset")
def fixture_bigquery_dataset() -> Iterator[tuple[bigquery.Client, str]]:
    """A new, empty BigQuery dataset, deleted afterwards: the client and the dataset's ID."""
    path = os.environ[GCP_CREDENTIALS_VAR]
    credentials = service_account.Credentials.from_service_account_file(path)
    with bigquery.Client(project=credentials.project_id, credentials=credentials) as client:
        dataset = bigquery.Dataset(f"{credentials.project_id}.create_testing_dwh_{uuid.uuid4().hex[:12]}")
        dataset.location = "us-west1"
        client.create_dataset(dataset)
        try:
            yield client, dataset.dataset_id
        finally:
            client.delete_dataset(dataset.reference, delete_contents=True, not_found_ok=True)


@pytest.mark.integration
@skip_unless_gcp_credentials
def test_jsonl_populates_bigquery_records(bigquery_dataset: tuple[bigquery.Client, str]):
    client, dataset_id = bigquery_dataset
    path = os.environ[GCP_CREDENTIALS_VAR]
    dsn = f"bigquery://{client.project}/{dataset_id}?credentials_path={path}"

    create_testing_dwh(
        dsn=dsn, src=TESTDATA / "stress_dwh.jsonl.zst", table_name=stress_dwh_def.BIGQUERY.table, nrows=5
    )

    table = f"`{client.project}.{dataset_id}.{stress_dwh_def.BIGQUERY.table}`"
    rows = list(
        client.query_and_wait(
            f"SELECT `Participant ID` AS id, address.city AS city, ARRAY_LENGTH(events) AS events, "
            f"ARRAY_LENGTH(tags) AS tags FROM {table} ORDER BY 1"
        )
    )
    expected = stress_dwh_def.generate_rows()[:5]
    assert [(r["id"], r["city"], r["events"], r["tags"]) for r in rows] == [
        (row["id"], row["city"], len(cast(list, row["events"])), len(cast(list, row["tags"]))) for row in expected
    ]
