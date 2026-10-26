"""Isolated DWH error fallbacks and failures that are impractical to induce reliably.

Prefer real server coverage in test_dwh_errors_postgres.py. Keep simulated exceptions here only for behavior
those integration tests cannot exercise.
"""

import psycopg
import psycopg.errors
import psycopg2
import psycopg2.errors
import pytest
import sqlalchemy.exc

from xngin.apiserver.dns.safe_resolve import DnsLookupError, DnsLookupUnsafeError
from xngin.apiserver.dwh.dwh_errors import (
    HINT_CHECK_HOSTNAME,
    HINT_UNSAFE_HOSTNAME,
    classify_dwh_error,
)
from xngin.apiserver.exceptions_common import (
    CannotFindTableError,
    DwhAuthenticationError,
    DwhConnectionError,
    DwhDatabaseDoesNotExistError,
    DwhErrorCode,
    DwhPermissionDeniedError,
    DwhQueryError,
    DwhTimeoutError,
)
from xngin.apiserver.settings import Dsn

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
