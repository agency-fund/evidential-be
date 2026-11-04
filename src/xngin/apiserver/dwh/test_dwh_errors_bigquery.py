"""Integration tests that provoke real BigQuery errors and check how DwhSession reports them.

Tests only run with ``-m integration`` and when these two variables are set:

- XNGIN_DEVDWH_GCP_CREDENTIALS: a key for a service account in xngin-development-dc that can create and delete
  datasets. Each module run creates its own dataset and deletes it afterwards.
- XNGIN_DEVDWH_GCP_CREDENTIALS_RESTRICTED: a key for a second service account that holds only the BigQuery Job
  User role in the project, so it can run jobs but cannot read any dataset. Only the permission tests need it.

The jobs-denied case points the first account at a project it holds no role in, so it needs no further credentials.
"""

import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import google.api_core.exceptions
import google.auth.exceptions
import pytest
import sqlalchemy.exc
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.cloud import bigquery
from google.oauth2 import service_account
from sqlalchemy import text

from xngin.apiserver import exceptionhandlers
from xngin.apiserver.dwh.dwh_errors import classify_dwh_error
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.exceptions_common import (
    CannotFindTableError,
    DwhAuthenticationError,
    DwhColumnNotFoundError,
    DwhDatabaseDoesNotExistError,
    DwhErrorCode,
    DwhPermissionDeniedError,
)
from xngin.apiserver.settings import BqDsn, GcpServiceAccountFile

# Skips an integration test when no service account key is available.
skip_unless_bq_available = pytest.mark.skipif(
    not os.environ.get("XNGIN_DEVDWH_GCP_CREDENTIALS"),
    reason="XNGIN_DEVDWH_GCP_CREDENTIALS is required for BigQuery integration tests",
)

# Skips a permission test when the restricted account's key is not available.
skip_unless_bq_restricted_available = pytest.mark.skipif(
    not os.environ.get("XNGIN_DEVDWH_GCP_CREDENTIALS_RESTRICTED"),
    reason="XNGIN_DEVDWH_GCP_CREDENTIALS_RESTRICTED is required for BigQuery permission tests",
)

# A public project the test accounts hold no role in, so creating a query job there is denied.
PROJECT_WITHOUT_JOB_ACCESS = "bigquery-public-data"
DATASET_WITHOUT_JOB_ACCESS = "samples"

# Matches the location CI creates its BigQuery test datasets in.
LOCATION = "us-west1"
TABLE = "participants"


@pytest.fixture(name="credentials_path", scope="module")
def fixture_credentials_path() -> str:
    return os.environ["XNGIN_DEVDWH_GCP_CREDENTIALS"]


@pytest.fixture(name="bq_client", scope="module")
def fixture_bq_client(credentials_path: str) -> Iterator[bigquery.Client]:
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    with bigquery.Client(project=credentials.project_id, credentials=credentials, location=LOCATION) as client:
        yield client


@pytest.fixture(name="bq_dsn", scope="module")
def fixture_bq_dsn(credentials_path: str, bq_client: bigquery.Client) -> Iterator[BqDsn]:
    """A BigQuery DSN for a dataset created for this module, holding one empty table."""
    client = bq_client
    dataset_id = f"dwherrors_{uuid.uuid4().hex[:12]}"
    dataset = bigquery.Dataset(f"{client.project}.{dataset_id}")
    dataset.location = LOCATION
    # A safety net in case teardown never runs.
    dataset.default_table_expiration_ms = 24 * 60 * 60 * 1000
    client.create_dataset(dataset)
    try:
        client.query_and_wait(f"CREATE TABLE `{client.project}.{dataset_id}.{TABLE}` (id INT64, is_engaged BOOL)")
        yield BqDsn(
            driver="bigquery",
            project_id=client.project,
            dataset_id=dataset_id,
            credentials=GcpServiceAccountFile(type="serviceaccountfile", path=credentials_path),
        )
    finally:
        client.delete_dataset(dataset, delete_contents=True, not_found_ok=True)


@pytest.fixture(name="restricted_bq_dsn", scope="module")
def fixture_restricted_bq_dsn(bq_dsn: BqDsn) -> BqDsn:
    """The module's dataset as seen by the account that may run jobs but may not read any dataset."""
    return bq_dsn.model_copy(
        update={
            "credentials": GcpServiceAccountFile(
                type="serviceaccountfile", path=os.environ["XNGIN_DEVDWH_GCP_CREDENTIALS_RESTRICTED"]
            )
        }
    )


def _assert_permission_denied(error: DwhPermissionDeniedError, dsn: BqDsn) -> None:
    assert error.code == DwhErrorCode.PERMISSION_DENIED
    assert error.http_status == 424
    assert error.table_name == TABLE
    assert error.message == (
        f'The configured BigQuery credentials do not have permission to read the table "{TABLE}" '
        f"in {dsn.project_id}.{dsn.dataset_id}."
    )
    assert error.hint == (
        f"Grant the service account the BigQuery Data Viewer role on the dataset {dsn.dataset_id} "
        f"in the project {dsn.project_id}."
    )
    assert error.hint_command is None
    _assert_undecorated(error.dwh_message)
    assert error.dwh_message is not None
    assert f"{dsn.project_id}:{dsn.dataset_id}.{TABLE}" in error.dwh_message


def _assert_undecorated(dwh_message: str | None) -> None:
    """The server's own text, without the method and URL that google.api_core prefixes or the job footer."""
    assert dwh_message is not None
    assert dwh_message.strip()
    assert "https://" not in dwh_message
    assert "Job ID" not in dwh_message
    assert "Location:" not in dwh_message
    assert "[SQL:" not in dwh_message
    assert "[parameters:" not in dwh_message


@pytest.mark.integration
@skip_unless_bq_available
def test_an_existing_table_inspects_and_is_readable(bq_dsn: BqDsn):
    with DwhSession.open(bq_dsn) as dwh:
        sa_table = dwh.inspect_table(TABLE)
        assert {c.name for c in sa_table.columns} == {"id", "is_engaged"}
        dwh.check_table_readable(TABLE)


@pytest.mark.integration
@skip_unless_bq_available
def test_reflecting_a_missing_table_names_it(bq_dsn: BqDsn):
    with DwhSession.open(bq_dsn) as dwh, pytest.raises(CannotFindTableError) as excinfo:
        dwh.inspect_table("no_such_table")
    assert excinfo.value.table_name == "no_such_table"
    assert excinfo.value.code == DwhErrorCode.TABLE_NOT_FOUND
    assert excinfo.value.message == (
        "The table 'no_such_table' does not exist. Check the table name, and the tables in the database, and try again."
    )


@pytest.mark.integration
@skip_unless_bq_available
def test_the_readability_probe_reports_a_missing_table(bq_dsn: BqDsn):
    """The dry run answers with a NotFound whose message is prefixed with the request method and URL."""
    with DwhSession.open(bq_dsn) as dwh, pytest.raises(CannotFindTableError) as excinfo:
        dwh.check_table_readable("no_such_table")
    assert excinfo.value.table_name == "no_such_table"
    assert excinfo.value.code == DwhErrorCode.TABLE_NOT_FOUND
    _assert_undecorated(excinfo.value.dwh_message)
    assert excinfo.value.dwh_message is not None
    assert f"{bq_dsn.project_id}:{bq_dsn.dataset_id}.no_such_table" in excinfo.value.dwh_message


@pytest.mark.integration
@skip_unless_bq_available
def test_a_query_against_a_missing_table_names_it(bq_dsn: BqDsn):
    """Through the DB-API, where the Google exception is wrapped in a DatabaseError without a cause."""
    with DwhSession.open(bq_dsn) as dwh, pytest.raises(CannotFindTableError) as excinfo:
        dwh._on_worker(
            "query",
            lambda: dwh._session.execute(
                text(f"SELECT 1 FROM `{bq_dsn.project_id}.{bq_dsn.dataset_id}.no_such_table`")
            ),
        )
    assert excinfo.value.table_name == "no_such_table"
    assert excinfo.value.code == DwhErrorCode.TABLE_NOT_FOUND
    _assert_undecorated(excinfo.value.dwh_message)
    assert excinfo.value.dwh_message is not None
    assert f"{bq_dsn.project_id}:{bq_dsn.dataset_id}.no_such_table" in excinfo.value.dwh_message


@pytest.mark.integration
@skip_unless_bq_available
def test_a_query_against_a_missing_column_names_it(bq_dsn: BqDsn):
    with DwhSession.open(bq_dsn) as dwh, pytest.raises(DwhColumnNotFoundError) as excinfo:
        dwh._on_worker(
            "query",
            lambda: dwh._session.execute(
                text(f"SELECT no_such_column FROM `{bq_dsn.project_id}.{bq_dsn.dataset_id}.{TABLE}`")
            ),
        )
    assert excinfo.value.message == 'The column "no_such_column" does not exist.'
    assert excinfo.value.code == DwhErrorCode.COLUMN_NOT_FOUND
    assert excinfo.value.hint == "Check the column name, and the columns in the table, and try again."
    _assert_undecorated(excinfo.value.dwh_message)
    assert excinfo.value.dwh_message is not None
    assert excinfo.value.dwh_message.startswith("Unrecognized name: no_such_column")


@pytest.mark.integration
@skip_unless_bq_available
@pytest.mark.parametrize("failure", ["missing-table", "column-suggestion", "type-mismatch"])
def test_failed_jobs_use_real_sdk_messages(bq_dsn: BqDsn, bq_client: bigquery.Client, failure: str):
    """Submit jobs so the SDK adds its real location/job footer, rather than fabricating the message format."""
    table = f"`{bq_dsn.project_id}.{bq_dsn.dataset_id}.{TABLE}`"
    statements = {
        "missing-table": f"SELECT 1 FROM `{bq_dsn.project_id}.{bq_dsn.dataset_id}.no_such_table`",
        "column-suggestion": f"SELECT is_engage FROM {table}",
        "type-mismatch": f"SELECT id = is_engaged FROM {table}",
    }
    expected_types = {
        "missing-table": CannotFindTableError,
        "column-suggestion": DwhColumnNotFoundError,
        "type-mismatch": google.api_core.exceptions.BadRequest,
    }
    with DwhSession.open(bq_dsn) as dwh, pytest.raises(expected_types[failure]) as excinfo:
        dwh._on_worker("query job", lambda: bq_client.query(statements[failure]).result(timeout=30))

    error = excinfo.value
    google_error = error if isinstance(error, google.api_core.exceptions.BadRequest) else error.__cause__
    assert isinstance(google_error, google.api_core.exceptions.GoogleAPICallError)
    assert "Location:" in google_error.message
    assert "Job ID:" in google_error.message
    if isinstance(error, CannotFindTableError):
        assert error.code == DwhErrorCode.TABLE_NOT_FOUND
        assert error.table_name == "no_such_table"
        _assert_undecorated(error.dwh_message)
        assert error.dwh_message is not None
        assert f"{bq_dsn.project_id}:{bq_dsn.dataset_id}.no_such_table" in error.dwh_message
    elif isinstance(error, DwhColumnNotFoundError):
        assert error.code == DwhErrorCode.COLUMN_NOT_FOUND
        assert error.message == 'The column "is_engage" does not exist.'
        _assert_undecorated(error.dwh_message)
        assert error.dwh_message is not None
        assert "Did you mean is_engaged?" in error.dwh_message
    else:
        assert google_error.errors[0]["reason"] == "invalidQuery"
        assert "No matching signature" in google_error.message
        assert classify_dwh_error(error, bq_dsn) is None


@pytest.mark.integration
@skip_unless_bq_available
@skip_unless_bq_restricted_available
def test_permission_hint_uses_the_denied_table_instead_of_the_configured_dataset(restricted_bq_dsn: BqDsn):
    """A fully qualified table can name a different project/dataset from the data source's defaults."""
    credentials = service_account.Credentials.from_service_account_file(
        os.environ["XNGIN_DEVDWH_GCP_CREDENTIALS_RESTRICTED"]
    )
    with (
        bigquery.Client(project=restricted_bq_dsn.project_id, credentials=credentials) as client,
        pytest.raises(google.api_core.exceptions.Forbidden) as excinfo,
    ):
        client.get_table(f"{restricted_bq_dsn.project_id}.{restricted_bq_dsn.dataset_id}.{TABLE}")
    config = restricted_bq_dsn.model_copy(update={"project_id": "other-project", "dataset_id": "other_dataset"})
    error = classify_dwh_error(excinfo.value, config)
    assert isinstance(error, DwhPermissionDeniedError)
    _assert_permission_denied(error, restricted_bq_dsn)


@pytest.mark.integration
@skip_unless_bq_available
def test_a_missing_dataset_is_reported_as_a_missing_database(bq_dsn: BqDsn):
    missing = bq_dsn.model_copy(update={"dataset_id": f"{bq_dsn.dataset_id}_missing"})
    with DwhSession.open(missing) as dwh:
        # Not an error: sqlalchemy-bigquery's get_table_names() swallows NotFound and reports no tables, so a
        # misspelled dataset only surfaces once a table is involved.
        assert dwh.list_tables() == []
        with pytest.raises(DwhDatabaseDoesNotExistError) as excinfo:
            dwh.check_table_readable(TABLE)
    assert excinfo.value.code == DwhErrorCode.DATABASE_NOT_FOUND
    assert excinfo.value.message == f"The BigQuery dataset {missing.project_id}.{missing.dataset_id} does not exist."
    assert excinfo.value.hint == "Check the project and dataset in the data source configuration."
    _assert_undecorated(excinfo.value.dwh_message)
    assert excinfo.value.dwh_message is not None
    assert f"{missing.project_id}:{missing.dataset_id}" in excinfo.value.dwh_message


@pytest.mark.integration
@skip_unless_bq_available
def test_a_key_that_google_rejects_is_an_authentication_error(bq_dsn: BqDsn, credentials_path: str, tmp_path: Path):
    """A well-formed key file whose private key does not belong to the service account fails at token refresh."""
    key = json.loads(Path(credentials_path).read_text())
    key["private_key"] = (
        rsa
        .generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        .decode()
    )
    wrong_key_path = tmp_path / "wrong_key.json"
    wrong_key_path.write_text(json.dumps(key))
    wrong = bq_dsn.model_copy(
        update={"credentials": GcpServiceAccountFile(type="serviceaccountfile", path=str(wrong_key_path))}
    )

    with DwhSession.open(wrong) as dwh, pytest.raises(DwhAuthenticationError) as excinfo:
        dwh.list_tables()
    assert excinfo.value.code == DwhErrorCode.AUTHENTICATION
    assert excinfo.value.message == "Google rejected the configured BigQuery credentials."
    assert excinfo.value.hint == "Check the service account credentials in the data source configuration."
    assert excinfo.value.dwh_message is not None
    assert "invalid_grant" in excinfo.value.dwh_message
    assert "PRIVATE KEY" not in excinfo.value.dwh_message
    cause = excinfo.value.__cause__
    assert isinstance(cause, google.auth.exceptions.RefreshError)
    assert excinfo.value.dwh_message == cause.args[0].strip().splitlines()[0]


@pytest.mark.integration
@skip_unless_bq_available
@skip_unless_bq_restricted_available
def test_an_account_without_dataset_access_cannot_reflect_the_table(restricted_bq_dsn: BqDsn):
    """Reflection asks for the table directly, so the denial arrives raw rather than through the DB-API."""
    with DwhSession.open(restricted_bq_dsn) as dwh, pytest.raises(DwhPermissionDeniedError) as excinfo:
        dwh.inspect_table(TABLE)
    _assert_permission_denied(excinfo.value, restricted_bq_dsn)


@pytest.mark.integration
@skip_unless_bq_available
@skip_unless_bq_restricted_available
def test_an_account_without_dataset_access_fails_the_readability_probe(restricted_bq_dsn: BqDsn):
    with DwhSession.open(restricted_bq_dsn) as dwh, pytest.raises(DwhPermissionDeniedError) as excinfo:
        dwh.check_table_readable(TABLE)
    _assert_permission_denied(excinfo.value, restricted_bq_dsn)


@pytest.mark.integration
@skip_unless_bq_available
@skip_unless_bq_restricted_available
def test_an_account_without_dataset_access_cannot_query_the_table(restricted_bq_dsn: BqDsn):
    with DwhSession.open(restricted_bq_dsn) as dwh, pytest.raises(DwhPermissionDeniedError) as excinfo:
        dwh._on_worker(
            "query",
            lambda: dwh._session.execute(
                text(f"SELECT 1 FROM `{restricted_bq_dsn.project_id}.{restricted_bq_dsn.dataset_id}.{TABLE}` LIMIT 0")
            ),
        )
    _assert_permission_denied(excinfo.value, restricted_bq_dsn)


@pytest.mark.integration
@skip_unless_bq_available
@skip_unless_bq_restricted_available
def test_an_account_without_dataset_access_sees_no_tables(restricted_bq_dsn: BqDsn):
    """Not an error: sqlalchemy-bigquery lists only the datasets the account can see, so, as with a misspelled
    dataset, a misconfigured account reports no tables and the denial surfaces once a table is involved."""
    with DwhSession.open(restricted_bq_dsn) as dwh:
        assert dwh.list_tables() == []


@pytest.mark.integration
@skip_unless_bq_available
def test_an_account_that_cannot_run_jobs_in_the_project_is_told_which_role_it_needs(bq_dsn: BqDsn):
    """The public dataset is readable by anyone, but query jobs are created in the configured project."""
    elsewhere = bq_dsn.model_copy(
        update={"project_id": PROJECT_WITHOUT_JOB_ACCESS, "dataset_id": DATASET_WITHOUT_JOB_ACCESS}
    )
    with DwhSession.open(elsewhere) as dwh, pytest.raises(DwhPermissionDeniedError) as excinfo:
        dwh.check_table_readable("shakespeare")
    assert excinfo.value.code == DwhErrorCode.PERMISSION_DENIED
    assert excinfo.value.table_name is None
    assert excinfo.value.message == (
        f"The configured BigQuery credentials cannot run queries in the project {PROJECT_WITHOUT_JOB_ACCESS}."
    )
    assert excinfo.value.hint == (
        f"Grant the service account the BigQuery Job User role on the project {PROJECT_WITHOUT_JOB_ACCESS}."
    )
    _assert_undecorated(excinfo.value.dwh_message)
    assert excinfo.value.dwh_message is not None
    assert "bigquery.jobs.create" in excinfo.value.dwh_message
    assert PROJECT_WITHOUT_JOB_ACCESS in excinfo.value.dwh_message


@pytest.mark.integration
@skip_unless_bq_available
@pytest.mark.parametrize(
    ("statement", "parameters", "sensitive_value"),
    [
        pytest.param("SELECT 1 'SYNTAX_SENTINEL'", {}, "SYNTAX_SENTINEL", id="syntax-error"),
        pytest.param(
            "SELECT CAST(:value AS INT64)", {"value": "BIND_SENTINEL"}, "BIND_SENTINEL", id="invalid-bound-value"
        ),
    ],
)
def test_invalid_queries_propagate_but_do_not_expose_values_in_http_responses(
    bq_dsn: BqDsn, monkeypatch: pytest.MonkeyPatch, statement: str, parameters: dict[str, str], sensitive_value: str
):
    """Use real BigQuery errors with the production handlers; SQL bugs stay 500s with a generic response body."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    app = FastAPI()
    exceptionhandlers.setup(app)
    failures: list[sqlalchemy.exc.DBAPIError] = []

    @app.get("/query")
    def query() -> None:
        try:
            with DwhSession.open(bq_dsn) as dwh:
                dwh._on_worker("query", lambda: dwh._session.execute(text(statement), parameters))
        except sqlalchemy.exc.DBAPIError as exc:
            failures.append(exc)
            raise

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/query")

    assert len(failures) == 1
    error = failures[0]
    assert error.orig is not None
    google_error = error.orig.args[0]
    assert isinstance(google_error, google.api_core.exceptions.BadRequest)
    assert google_error.errors[0]["reason"] == "invalidQuery"
    # Prove the server echoed the value: a generic response is necessary even with SQLAlchemy parameters hidden.
    assert sensitive_value in google_error.message
    assert error.hide_parameters
    assert classify_dwh_error(error, bq_dsn) is None
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert sensitive_value not in response.text
    assert statement not in response.text
