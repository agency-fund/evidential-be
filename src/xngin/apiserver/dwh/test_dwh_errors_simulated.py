"""Isolated DWH error fallbacks and failures that are impractical to induce reliably.

Prefer real server coverage in test_dwh_errors_postgres.py and test_dwh_errors_bigquery.py. Keep simulated
exceptions here only for behavior those integration tests cannot exercise.
"""

import google.api_core.exceptions
import google.auth.exceptions
import psycopg
import psycopg.errors
import psycopg2
import psycopg2.errors
import pytest
import requests.exceptions
import sqlalchemy
import sqlalchemy.exc
from google.cloud.bigquery.dbapi import exceptions as dbapi_exceptions

from xngin.apiserver import flags
from xngin.apiserver.dns.safe_resolve import DnsLookupError, DnsLookupUnsafeError
from xngin.apiserver.dwh.dwh_errors import (
    HINT_CHECK_HOSTNAME,
    HINT_UNSAFE_HOSTNAME,
    classify_dwh_error,
)
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.exceptions_common import (
    CannotFindTableError,
    DwhAuthenticationError,
    DwhConfigurationError,
    DwhConnectionError,
    DwhDatabaseDoesNotExistError,
    DwhErrorCode,
    DwhPermissionDeniedError,
    DwhQueryError,
    DwhTimeoutError,
)
from xngin.apiserver.settings import BqDsn, Dsn, GcpServiceAccountFile

STATEMENT = "SELECT secret FROM t WHERE a = %(a)s"
PARAMS = {"a": "BIND_VALUE"}

PG_DSN = Dsn(
    driver="postgresql+psycopg",
    host="db.example.com",
    user="readonly",
    password="pw",
    dbname="warehouse",
    sslmode="require",
)
BQ_DSN = BqDsn(
    driver="bigquery",
    project_id="my-project",
    dataset_id="my_dataset",
    credentials=GcpServiceAccountFile(type="serviceaccountfile", path="/dev/null"),
)

PG3_CONNECT = 'connection failed: connection to server at "127.0.0.1", port 5499 failed: '
PG2_CONNECT = 'connection to server at "127.0.0.1", port 5499 failed: '


def _pg2(code: str, message: str) -> psycopg2.Error:
    """Builds a psycopg2 error that reports a SQLSTATE.

    psycopg2 only fills the read-only pgcode from a server result, so a subclass shadows it for the synthetic case.
    test_dwh_errors_postgres.py covers errors the server actually produced.
    """
    base = psycopg2.errors.lookup(code)
    error: psycopg2.Error = type(base.__name__, (base,), {"pgcode": code})(message)
    return error


def _wrapped(orig: BaseException) -> sqlalchemy.exc.DBAPIError:
    """Wraps a driver error the way SQLAlchemy does, with the statement and parameters attached."""
    return sqlalchemy.exc.DBAPIError.instance(STATEMENT, PARAMS, orig, Exception)  # type: ignore[return-value]


def _bq_wrapped(google_exc: BaseException) -> sqlalchemy.exc.DatabaseError:
    """Mirrors the BigQuery DB-API cursor, which raises DatabaseError(exc) with no `from`."""
    return sqlalchemy.exc.DatabaseError("stmt", {}, dbapi_exceptions.DatabaseError(google_exc))


def _with_cause(exc: BaseException, cause: BaseException) -> BaseException:
    exc.__cause__ = cause
    return exc


MAPPING = [
    pytest.param(
        _wrapped(_pg2("42501", "permission denied for relation t")),
        PG_DSN,
        DwhPermissionDeniedError,
        DwhErrorCode.PERMISSION_DENIED,
        "t",
        id="pg2-permission-denied-relation",
    ),
    pytest.param(
        _wrapped(psycopg.errors.UndefinedTable("unparseable")),
        PG_DSN,
        CannotFindTableError,
        DwhErrorCode.TABLE_NOT_FOUND,
        "unknown",
        id="pg3-table-not-found-unparseable",
    ),
    pytest.param(
        _wrapped(psycopg.errors.InvalidCatalogName('database "warehouse" does not exist')),
        PG_DSN,
        DwhDatabaseDoesNotExistError,
        DwhErrorCode.DATABASE_NOT_FOUND,
        None,
        id="pg3-database-not-found-sqlstate",
    ),
    pytest.param(
        _wrapped(psycopg.errors.InvalidPassword('password authentication failed for user "readonly"')),
        PG_DSN,
        DwhAuthenticationError,
        DwhErrorCode.AUTHENTICATION,
        None,
        id="pg3-authentication-sqlstate",
    ),
    pytest.param(
        _wrapped(_pg2("28000", 'role "readonly" is not permitted to log in')),
        PG_DSN,
        DwhAuthenticationError,
        DwhErrorCode.AUTHENTICATION,
        None,
        id="pg2-authentication-class-28",
    ),
    pytest.param(
        _wrapped(psycopg.errors.ConnectionFailure("server closed the connection unexpectedly")),
        PG_DSN,
        DwhConnectionError,
        DwhErrorCode.CONNECTION,
        None,
        id="pg3-connection-class-08",
    ),
    pytest.param(
        _wrapped(_pg2("08006", "server closed the connection unexpectedly")),
        PG_DSN,
        DwhConnectionError,
        DwhErrorCode.CONNECTION,
        None,
        id="pg2-connection-class-08",
    ),
    pytest.param(
        _wrapped(_pg2("53100", "could not extend file: No space left on device")),
        PG_DSN,
        DwhQueryError,
        DwhErrorCode.QUERY_FAILED,
        None,
        id="pg2-insufficient-resources-class-53",
    ),
    pytest.param(
        _wrapped(
            psycopg.OperationalError(
                PG3_CONNECT
                + 'FATAL:  no pg_hba.conf entry for host "203.0.113.5", user "readonly", database "warehouse"'
            )
        ),
        PG_DSN,
        DwhAuthenticationError,
        DwhErrorCode.AUTHENTICATION,
        None,
        id="pg3-connect-pg-hba",
    ),
    pytest.param(
        _wrapped(psycopg.errors.ConnectionTimeout("connection timeout expired")),
        PG_DSN,
        DwhTimeoutError,
        DwhErrorCode.TIMEOUT,
        None,
        id="pg3-connect-timeout",
    ),
    pytest.param(
        _wrapped(psycopg2.OperationalError(PG2_CONNECT + "timeout expired\n")),
        PG_DSN,
        DwhTimeoutError,
        DwhErrorCode.TIMEOUT,
        None,
        id="pg2-connect-timeout",
    ),
    pytest.param(
        google.api_core.exceptions.Unauthorized("Request had invalid authentication credentials."),
        BQ_DSN,
        DwhAuthenticationError,
        DwhErrorCode.AUTHENTICATION,
        None,
        id="bq-unauthorized-raw",
    ),
    pytest.param(
        _bq_wrapped(google.api_core.exceptions.Unauthorized("Request had invalid authentication credentials.")),
        BQ_DSN,
        DwhAuthenticationError,
        DwhErrorCode.AUTHENTICATION,
        None,
        id="bq-unauthorized-wrapped",
    ),
    pytest.param(
        _bq_wrapped(
            google.api_core.exceptions.BadRequest(
                "Resources exceeded during query execution.",
                errors=[{"reason": "resourcesExceeded", "message": "Resources exceeded during query execution."}],
            )
        ),
        BQ_DSN,
        DwhQueryError,
        DwhErrorCode.QUERY_FAILED,
        None,
        id="bq-badrequest-resources-exceeded",
    ),
    pytest.param(
        _bq_wrapped(google.api_core.exceptions.DeadlineExceeded("Deadline Exceeded")),
        BQ_DSN,
        DwhTimeoutError,
        DwhErrorCode.TIMEOUT,
        None,
        id="bq-deadline-exceeded",
    ),
    pytest.param(
        google.api_core.exceptions.ServiceUnavailable("The service is currently unavailable."),
        BQ_DSN,
        DwhConnectionError,
        DwhErrorCode.CONNECTION,
        None,
        id="bq-service-unavailable",
    ),
    pytest.param(
        _bq_wrapped(google.api_core.exceptions.InternalServerError("Internal error encountered.")),
        BQ_DSN,
        DwhQueryError,
        DwhErrorCode.QUERY_FAILED,
        None,
        id="bq-other-google-api-error",
    ),
    pytest.param(
        google.auth.exceptions.DefaultCredentialsError("Your default credentials were not found."),
        BQ_DSN,
        DwhAuthenticationError,
        DwhErrorCode.AUTHENTICATION,
        None,
        id="bq-default-credentials-error",
    ),
    pytest.param(
        google.auth.exceptions.TransportError("Failed to retrieve a token."),
        BQ_DSN,
        DwhConnectionError,
        DwhErrorCode.CONNECTION,
        None,
        id="bq-transport-error",
    ),
    pytest.param(
        requests.exceptions.ConnectionError("HTTPSConnectionPool(host='bigquery.googleapis.com', port=443)"),
        BQ_DSN,
        DwhConnectionError,
        DwhErrorCode.CONNECTION,
        None,
        id="bq-requests-connection-error",
    ),
    pytest.param(
        google.api_core.exceptions.RetryError(
            "retry failed", cause=google.api_core.exceptions.DeadlineExceeded("Deadline Exceeded")
        ),
        BQ_DSN,
        DwhTimeoutError,
        DwhErrorCode.TIMEOUT,
        None,
        id="bq-retry-error-deadline",
    ),
    pytest.param(
        google.api_core.exceptions.RetryError("retry failed", cause=ConnectionResetError("reset")),
        BQ_DSN,
        DwhConnectionError,
        DwhErrorCode.CONNECTION,
        None,
        id="bq-retry-error-other",
    ),
    pytest.param(
        _with_cause(ValueError("reflection failed"), sqlalchemy.exc.NoSuchTableError("no_such_table_xyz")),
        PG_DSN,
        CannotFindTableError,
        DwhErrorCode.TABLE_NOT_FOUND,
        "no_such_table_xyz",
        id="walks-cause",
    ),
]


@pytest.mark.parametrize(("exc", "dwh_config", "expected_cls", "code", "table_name"), MAPPING)
def test_classify_dwh_error_mapping(exc, dwh_config, expected_cls, code, table_name):
    result = classify_dwh_error(exc, dwh_config)
    assert result is not None
    assert type(result) is expected_cls
    assert result.code == code
    assert result.http_status == 424
    assert result.table_name == table_name


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(TypeError("our own bug"), id="type-error"),
        pytest.param(_wrapped(Exception("some driver error")), id="unknown-dbapi-error"),
        pytest.param(_wrapped(psycopg.InterfaceError("the connection is closed")), id="pg3-client-side-wrapped"),
        pytest.param(
            _bq_wrapped(google.api_core.exceptions.BadRequest("Syntax error: Unexpected keyword SELEC at [1:1]")),
            id="bq-badrequest-without-reason",
        ),
        pytest.param(
            google.api_core.exceptions.BadRequest(
                "No matching signature for operator = for argument types: INT64, STRING",
                errors=[{"reason": "invalidQuery", "message": "No matching signature for operator ="}],
            ),
            id="bq-badrequest-invalid-query-raw",
        ),
        pytest.param(
            _bq_wrapped(
                google.api_core.exceptions.BadRequest(
                    "No matching signature for operator = for argument types: INT64, STRING",
                    errors=[{"reason": "invalidQuery", "message": "No matching signature for operator ="}],
                )
            ),
            id="bq-badrequest-invalid-query-wrapped",
        ),
        pytest.param(TimeoutError("deadline"), id="builtin-timeout"),
        pytest.param(DwhQueryError("already classified"), id="dwh-error"),
        pytest.param(DwhTimeoutError("already classified"), id="dwh-timeout-error"),
        pytest.param(sqlalchemy.exc.InvalidRequestError("not a dbapi error"), id="sqlalchemy-non-dbapi"),
        pytest.param(psycopg.ProgrammingError("the last operation didn't produce a result"), id="pg3-client-side-raw"),
    ],
)
def test_classify_dwh_error_returns_none_for_errors_that_may_be_our_own(exc):
    """Unclassified errors propagate as 500s, so a bug in the SQL we generate reaches Sentry."""
    assert classify_dwh_error(exc, PG_DSN) is None


@pytest.mark.parametrize(
    "orig",
    [
        pytest.param(psycopg.errors.OutOfMemory("out of memory"), id="insufficient-resources"),
    ],
)
def test_statement_and_bind_values_never_reach_the_user(orig):
    result = classify_dwh_error(_wrapped(orig), PG_DSN)
    assert result is not None
    for field in (result.message, result.hint, result.dwh_message):
        assert field is None or not any(leak in field for leak in ("SELECT secret", "[SQL:", "BIND_VALUE"))


def test_database_not_found_falls_back_to_the_configured_name():
    result = classify_dwh_error(psycopg.errors.InvalidCatalogName("unparseable"), PG_DSN)
    assert result is not None
    assert result.message == 'The database "warehouse" does not exist.'


def test_dns_errors_carry_the_host_and_a_hint():
    plain = classify_dwh_error(DnsLookupError("db.example.com"), PG_DSN)
    unsafe = classify_dwh_error(DnsLookupUnsafeError("db.example.com"), PG_DSN)
    assert plain is not None and unsafe is not None
    assert plain.message == unsafe.message == "DNS issue with host: db.example.com"
    assert plain.dwh_message == unsafe.dwh_message == "db.example.com"
    assert plain.hint == HINT_CHECK_HOSTNAME
    assert unsafe.hint == HINT_UNSAFE_HOSTNAME


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    ("reason", "message"),
    [
        ("quotaExceeded", "Quota exceeded: Your project exceeded quota for free query bytes scanned."),
        ("billingNotEnabled", "Billing has not been enabled for this project."),
        ("rateLimitExceeded", "Exceeded rate limits: too many api requests per user per method."),
    ],
)
def test_bigquery_forbidden_non_permission_reasons_pass_through(reason, message, wrapped):
    exc = google.api_core.exceptions.Forbidden(
        f"POST https://bigquery.googleapis.com/query: {message}",
        errors=[{"reason": reason, "message": message}],
    )
    result = classify_dwh_error(_bq_wrapped(exc) if wrapped else exc, BQ_DSN)
    assert isinstance(result, DwhQueryError)
    assert result.message == f"BigQuery rejected the request ({reason})."
    assert result.dwh_message == message
    assert result.hint is None
    assert result.hint_command is None
    assert result.table_name is None


def test_bigquery_forbidden_from_grpc_without_structured_errors():
    """gRPC-origin errors carry the RPC error in errors rather than a dict, which must not break classification."""
    exc = google.api_core.exceptions.PermissionDenied("Access Denied", errors=(RuntimeError("rpc error"),))
    result = classify_dwh_error(exc, BQ_DSN)
    assert isinstance(result, DwhPermissionDeniedError)
    assert result.dwh_message == "Access Denied"


@pytest.mark.parametrize(
    ("config", "where", "hint_target"),
    [
        (BQ_DSN, "my-project.my_dataset", "the dataset my_dataset in the project my-project"),
        (PG_DSN, "the configured dataset", "the dataset"),
    ],
)
def test_bigquery_forbidden_without_a_table_uses_configured_dataset(config, where, hint_target):
    result = classify_dwh_error(google.api_core.exceptions.Forbidden("Access Denied"), config)
    assert isinstance(result, DwhPermissionDeniedError)
    assert result.message == f"The configured BigQuery credentials do not have permission to read {where}."
    assert result.hint == f"Grant the service account the BigQuery Data Viewer role on {hint_target}."
    assert result.table_name is None


def test_google_message_decoration_is_removed_without_structured_errors():
    """Keep fallback coverage for errors without the structured message returned by ordinary REST failures."""
    server_text = "Not found: Table my-project:my_dataset.t was not found in location US"
    exc = google.api_core.exceptions.NotFound(
        f"POST https://bigquery.googleapis.com/query: {server_text}\n\nLocation: US\nJob ID: job_abc123\n"
    )
    for candidate in (exc, _bq_wrapped(exc)):
        result = classify_dwh_error(candidate, BQ_DSN)
        assert isinstance(result, CannotFindTableError)
        assert result.table_name == "t"
        assert result.dwh_message == server_text


# Redshift refuses to apply a search_path that names a schema it cannot find, with SQLSTATE 3F000, where Postgres
# skips the entry. With enable_case_sensitive_identifier off (the default), that includes a schema created as
# "QB Space", which Redshift stores as qb space. These are the code and message a real Redshift Serverless workgroup
# returned (QUOTING-BUGS.md, bug 4); Postgres cannot produce them, so they are simulated.
REDSHIFT_MISSING_SCHEMA = 'schema "QB Space" does not exist'


@pytest.mark.parametrize(
    "orig",
    [
        pytest.param(_pg2("3F000", REDSHIFT_MISSING_SCHEMA), id="psycopg2"),
        pytest.param(psycopg.errors.InvalidSchemaName(REDSHIFT_MISSING_SCHEMA), id="psycopg"),
    ],
)
def test_a_search_path_schema_the_warehouse_cannot_find_is_a_configuration_error(orig):
    result = classify_dwh_error(_wrapped(orig), PG_DSN)
    assert isinstance(result, DwhConfigurationError)
    assert result.code == DwhErrorCode.CONFIGURATION
    assert result.message == 'The configured search_path names a schema the data warehouse could not find: "QB Space".'
    assert result.hint is not None and "stores schema names in lowercase" in result.hint
    assert result.dwh_message == REDSHIFT_MISSING_SCHEMA


def test_a_search_path_redshift_refuses_while_connecting_is_reported_as_a_configuration_error(monkeypatch):
    """Redshift raises the error while DwhSession applies the search_path to a new connection, before any query."""

    def refuse_search_path(_dbapi_connection, _connection_record):
        raise _pg2("3F000", REDSHIFT_MISSING_SCHEMA)

    original = DwhSession._extra_engine_setup

    def extra_engine_setup(self, engine: sqlalchemy.Engine):
        original(self, engine)
        sqlalchemy.event.listen(engine, "connect", refuse_search_path, insert=True)

    monkeypatch.setattr(DwhSession, "_extra_engine_setup", extra_engine_setup)
    # Redshift is always reached through psycopg2.
    dsn: Dsn = Dsn.from_url(flags.XNGIN_DEVDWH_DSN).model_copy(
        update={"driver": "postgresql+psycopg2", "search_path": '"QB Space"'}
    )
    with DwhSession.open(dsn) as dwh, pytest.raises(DwhConfigurationError) as excinfo:
        dwh.connectivity_check()
    assert excinfo.value.dwh_message == REDSHIFT_MISSING_SCHEMA
    assert isinstance(excinfo.value.__cause__, sqlalchemy.exc.ProgrammingError)


# libpq tries every address a host resolves to. These are the texts each driver reported when localhost resolved to
# ::1, where nothing listened, before 127.0.0.1, where the server refused the connection.
PG2_REFUSED_THEN = (
    'connection to server at "::1", port 5499 failed: Connection refused\n'
    "\tIs the server running on that host and accepting TCP/IP connections?\n"
    'connection to server at "127.0.0.1", port 5499 failed: FATAL:  {}\n'
)
PG3_REFUSED_THEN = (
    'connection failed: connection to server at "127.0.0.1", port 5499 failed: FATAL:  {0}\n'
    "Multiple connection attempts failed. All failures were:\n"
    "- host: '::1', port: '5499', hostaddr: None: connection failed: connection to server at \"::1\", port 5499 "
    "failed: Connection refused\n"
    "\tIs the server running on that host and accepting TCP/IP connections?\n"
    "- host: '127.0.0.1', port: '5499', hostaddr: '127.0.0.1': connection failed: connection to server at "
    '"127.0.0.1", port 5499 failed: FATAL:  {0}'
)


@pytest.mark.parametrize(
    "orig",
    [
        psycopg2.OperationalError(PG2_REFUSED_THEN.format('password authentication failed for user "postgres"')),
        psycopg.OperationalError(PG3_REFUSED_THEN.format('password authentication failed for user "postgres"')),
    ],
    ids=["psycopg2", "psycopg"],
)
def test_a_refused_address_before_a_rejected_password_is_an_authentication_error(orig):
    result = classify_dwh_error(_wrapped(orig), PG_DSN)
    assert isinstance(result, DwhAuthenticationError)
    assert result.dwh_message is not None
    assert result.dwh_message.endswith('FATAL:  password authentication failed for user "postgres"')


@pytest.mark.parametrize(
    "orig",
    [
        psycopg2.OperationalError(PG2_REFUSED_THEN.format('database "nonexistent_db" does not exist')),
        psycopg.OperationalError(PG3_REFUSED_THEN.format('database "nonexistent_db" does not exist')),
    ],
    ids=["psycopg2", "psycopg"],
)
def test_a_refused_address_before_a_missing_database_reports_the_servers_message(orig):
    result = classify_dwh_error(_wrapped(orig), PG_DSN)
    assert isinstance(result, DwhDatabaseDoesNotExistError)
    assert result.dwh_message is not None
    assert 'database "nonexistent_db" does not exist' in result.dwh_message
    assert "::1" not in result.dwh_message
