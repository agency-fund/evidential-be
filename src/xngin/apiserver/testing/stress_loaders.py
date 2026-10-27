"""Loads private copies of the stress test table into each kind of warehouse, for integration tests.

Each copy lives in its own schema or dataset, which is dropped afterwards, so tests always see the current
stress_dwh.* files rather than whatever task bootstrap-dwh-database or bootstrap-cloud-dwh loaded earlier.
"""

import os
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import sqlalchemy
from google.cloud import bigquery
from google.oauth2 import service_account

from xngin.apiserver import flags
from xngin.apiserver.dwh import dwh_utils
from xngin.apiserver.settings import BqDsn, Dsn, GcpServiceAccountFile
from xngin.apiserver.testing import stress_dwh_def
from xngin.apiserver.testing.helpers import GCP_CREDENTIALS_VAR
from xngin.apiserver.testing.pg_helpers import with_name_suffix
from xngin.cli.commands.create_testing_dwh import create_testing_dwh

TESTDATA = Path(__file__).resolve().parent.parent / "testdata"

REDSHIFT_DSN_VAR = "XNGIN_DEVDWH_AWS_REDSHIFT_DSN"
REDSHIFT_BUCKET_VAR = "XNGIN_DEVDWH_AWS_BUCKET"
REDSHIFT_IAM_ROLE_VAR = "XNGIN_DEVDWH_AWS_REDSHIFT_IAM_ROLE"


def pg_quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@contextmanager
def _dropped_afterwards(dsn: str, schema: str) -> Generator[None]:
    try:
        yield
    finally:
        engine = sqlalchemy.create_engine(Dsn.from_url(dsn).to_sqlalchemy_url())
        dwh_utils.extra_engine_setup(engine)
        try:
            with engine.begin() as conn:
                conn.execute(sqlalchemy.text(f"DROP SCHEMA IF EXISTS {pg_quote(schema)} CASCADE"))
        finally:
            engine.dispose()


@contextmanager
def postgres_stress_schema(name: str) -> Generator[str]:
    """Loads the Postgres stress test table into a new schema of the local testing warehouse, yielding the schema."""
    schema = with_name_suffix(name)
    with _dropped_afterwards(flags.XNGIN_DEVDWH_DSN, schema):
        create_testing_dwh(
            dsn=flags.XNGIN_DEVDWH_DSN,
            src=TESTDATA / "stress_dwh.csv.zst",
            schema_name=schema,
            table_name=stress_dwh_def.POSTGRES.table,
        )
        yield schema


@contextmanager
def redshift_stress_schema(name: str) -> Generator[Dsn]:
    """Loads the Redshift stress test table into a new schema, yielding a Dsn whose search_path names it.

    Needs the XNGIN_DEVDWH_AWS_* variables, and AWS credentials that may write to the bucket.
    """
    dsn = os.environ[REDSHIFT_DSN_VAR]
    schema = with_name_suffix(name)
    with _dropped_afterwards(dsn, schema):
        create_testing_dwh(
            dsn=dsn,
            src=TESTDATA / "stress_dwh.redshift.csv.zst",
            schema_name=schema,
            table_name=stress_dwh_def.REDSHIFT.table,
            bucket=os.environ[REDSHIFT_BUCKET_VAR],
            iam_role=os.environ[REDSHIFT_IAM_ROLE_VAR],
        )
        # Redshift stores the schema's name in lowercase.
        yield Dsn.from_url(dsn).model_copy(update={"search_path": pg_quote(schema.lower())})


@contextmanager
def bigquery_stress_dataset() -> Generator[BqDsn]:
    """Loads the BigQuery stress test table into a new dataset, yielding a BqDsn for it."""
    path = os.environ[GCP_CREDENTIALS_VAR]
    credentials = service_account.Credentials.from_service_account_file(path)
    with bigquery.Client(project=credentials.project_id, credentials=credentials) as client:
        dataset = bigquery.Dataset(f"{credentials.project_id}.stress_{uuid.uuid4().hex[:12]}")
        dataset.location = "us-west1"
        # A safety net in case teardown never runs.
        dataset.default_table_expiration_ms = 24 * 60 * 60 * 1000
        client.create_dataset(dataset)
        try:
            create_testing_dwh(
                dsn=f"bigquery://{credentials.project_id}/{dataset.dataset_id}?credentials_path={path}",
                src=TESTDATA / "stress_dwh.jsonl.zst",
                table_name=stress_dwh_def.BIGQUERY.table,
            )
            yield BqDsn(
                driver="bigquery",
                project_id=credentials.project_id,
                dataset_id=dataset.dataset_id,
                credentials=GcpServiceAccountFile(type="serviceaccountfile", path=path),
            )
        finally:
            client.delete_dataset(dataset.reference, delete_contents=True, not_found_ok=True)
