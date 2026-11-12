"""Integration tests of actual local Postgres failures with psycopg 3 and psycopg2.

These run in the normal test suite: the integration marker is reserved for tests requiring remote services.
Prefer extending this module to simulating errors in test_dwh_errors_simulated.py.
"""

import socket

import psycopg
import psycopg2
import pytest
import sqlalchemy
import sqlalchemy.exc
from sqlalchemy import text

from xngin.apiserver import flags
from xngin.apiserver.dwh.dwh_errors import classify_dwh_error
from xngin.apiserver.exceptions_common import (
    CannotFindTableError,
    DwhAuthenticationError,
    DwhColumnNotFoundError,
    DwhConfigurationError,
    DwhConnectionError,
    DwhDatabaseDoesNotExistError,
    DwhError,
    DwhErrorCode,
    DwhPermissionDeniedError,
    DwhQueryError,
    DwhTimeoutError,
)
from xngin.apiserver.settings import Dsn
from xngin.apiserver.testing.pg_helpers import with_name_suffix
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_TABLE_NAME

PARAMS = {"a": "BIND_VALUE"}


@pytest.fixture(name="real_dsn", params=["postgresql+psycopg", "postgresql+psycopg2"])
def fixture_real_dsn(request) -> Dsn:
    """The test warehouse's DSN under each Postgres driver we ship: psycopg 3 and psycopg2 (which Redshift uses)."""
    dsn: Dsn = Dsn.from_url(flags.XNGIN_DEVDWH_DSN)
    return dsn.model_copy(update={"driver": request.param})


def _connect_error(dsn: Dsn) -> sqlalchemy.exc.OperationalError:
    engine = sqlalchemy.create_engine(dsn.to_sqlalchemy_url())
    try:
        with pytest.raises(sqlalchemy.exc.OperationalError) as excinfo, engine.connect():
            pass
    finally:
        engine.dispose()
    return excinfo.value


def _query_error(dsn: Dsn, sql: str, params: dict | None = None) -> sqlalchemy.exc.DBAPIError:
    engine = sqlalchemy.create_engine(dsn.to_sqlalchemy_url())
    try:
        with pytest.raises(sqlalchemy.exc.DBAPIError) as excinfo, engine.connect() as conn:
            conn.execute(text(sql), params or {})
    finally:
        engine.dispose()
    return excinfo.value


def _classify_real_error(error: sqlalchemy.exc.DBAPIError, dsn: Dsn) -> DwhError:
    """Check both the SQLAlchemy wrapper and the raw driver's error from the same server response."""
    result = classify_dwh_error(error, dsn)
    assert isinstance(result, DwhError)
    assert result.http_status == 424
    assert error.orig is not None
    raw = classify_dwh_error(error.orig, dsn)
    assert type(raw) is type(result)
    assert raw is not None
    assert vars(raw) == vars(result)
    for field in (result.message, result.hint, result.dwh_message):
        assert field is None or not any(leak in field for leak in ("SELECT ", "[SQL:", "[parameters:", "BIND_VALUE"))
    return result


@pytest.mark.parametrize("table", ["no_such_table_xyz", "public.no_such_table_xyz"])
def test_real_driver_table_not_found(real_dsn: Dsn, table: str):
    error = _query_error(real_dsn, f"SELECT :a FROM {table}", PARAMS)
    result = _classify_real_error(error, real_dsn)
    assert isinstance(result, CannotFindTableError)
    assert result.code == DwhErrorCode.TABLE_NOT_FOUND
    assert result.table_name == table
    assert result.dwh_message == f'relation "{table}" does not exist'


@pytest.mark.parametrize(
    ("column", "primary"),
    [
        ("no_such_column", 'column "no_such_column" does not exist'),
        ("dwh_alias.no_such_column", "column dwh_alias.no_such_column does not exist"),
        ("current_incom", 'column "current_incom" does not exist'),
    ],
)
def test_real_driver_column_not_found(real_dsn: Dsn, column: str, primary: str):
    error = _query_error(real_dsn, f"SELECT :a, {column} FROM {TESTING_DWH_TABLE_NAME} AS dwh_alias", PARAMS)
    result = _classify_real_error(error, real_dsn)
    assert isinstance(result, DwhColumnNotFoundError)
    assert result.code == DwhErrorCode.COLUMN_NOT_FOUND
    assert result.message == f'The column "{column}" does not exist.'
    assert result.dwh_message == primary
    if column == "current_incom":
        assert result.hint is not None and "current_income" in result.hint


def test_real_driver_missing_insert_column_names_the_relation(real_dsn: Dsn):
    error = _query_error(real_dsn, f"INSERT INTO {TESTING_DWH_TABLE_NAME} (no_such_column) VALUES (:a)", PARAMS)
    result = _classify_real_error(error, real_dsn)
    assert isinstance(result, DwhColumnNotFoundError)
    assert result.code == DwhErrorCode.COLUMN_NOT_FOUND
    assert result.table_name == TESTING_DWH_TABLE_NAME
    assert result.message == 'The column "no_such_column" does not exist.'
    assert result.dwh_message == f'column "no_such_column" of relation "{TESTING_DWH_TABLE_NAME}" does not exist'


def test_real_driver_permission_denied(real_dsn, restricted_dsn):
    restricted = restricted_dsn.model_copy(update={"driver": real_dsn.driver})
    error = _query_error(restricted, f"SELECT :a FROM {TESTING_DWH_TABLE_NAME} LIMIT 0", PARAMS)
    result = _classify_real_error(error, restricted)
    assert isinstance(result, DwhPermissionDeniedError)
    assert result.code == DwhErrorCode.PERMISSION_DENIED
    assert result.message == (
        f'User "{restricted.user}" does not have permission to read the table "{TESTING_DWH_TABLE_NAME}".'
    )
    assert result.table_name == TESTING_DWH_TABLE_NAME
    assert result.hint_command == f"GRANT SELECT ON <schema>.{TESTING_DWH_TABLE_NAME} TO {restricted.user};"
    assert result.dwh_message == f"permission denied for table {TESTING_DWH_TABLE_NAME}"


@pytest.mark.parametrize(
    ("name", "search_path_kind"),
    [
        ("My Schema", "single"),
        ("MixedCase", "single"),
        ('a"quote', "single"),
        ("a.dot", "single"),
        ("select", "single"),
        ("My Schema", "default"),
        ("My Schema", "multiple"),
        ("My Schema", "user-alias"),
        ("My Schema", "existing-user-alias"),
    ],
)
def test_real_driver_grant_hints_restore_access(real_dsn: Dsn, restricted_dsn: Dsn, name: str, search_path_kind: str):
    admin = sqlalchemy.create_engine(real_dsn.to_sqlalchemy_url())
    # The schema is suffixed because the testing warehouse is shared by every pytest-xdist worker and test run; the
    # table inside it keeps the name exactly, so its hints still quote it.
    schema = restricted_dsn.user if search_path_kind == "existing-user-alias" else with_name_suffix(name)
    # quote() quotes only when needed, as the hints do; the suffix makes reserved words such as select ordinary.
    quoted_schema = admin.dialect.identifier_preparer.quote(schema)
    quoted = admin.dialect.identifier_preparer.quote_identifier(name)
    search_paths = {
        "single": quoted_schema,
        "default": None,
        "multiple": f"{quoted_schema}, public",
        "user-alias": f'"$user", {quoted_schema}',
        "existing-user-alias": '"$user", public',
    }
    restricted = restricted_dsn.model_copy(
        update={"driver": real_dsn.driver, "search_path": search_paths[search_path_kind]}
    )
    query = f"SELECT * FROM {quoted_schema}.{quoted}"
    try:
        with admin.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA {quoted_schema}"))
            conn.execute(text(f"CREATE TABLE {quoted_schema}.{quoted} (id int)"))
        try:
            for grant_kind in ("GRANT USAGE ON SCHEMA", "GRANT SELECT ON"):
                error = _classify_real_error(_query_error(restricted, query), restricted)
                assert isinstance(error, DwhPermissionDeniedError)
                assert error.code == DwhErrorCode.PERMISSION_DENIED
                if grant_kind == "GRANT USAGE ON SCHEMA":
                    assert (
                        error.message
                        == f'User "{restricted.user}" does not have permission to use the schema "{schema}".'
                    )
                    assert error.dwh_message == f"permission denied for schema {schema}"
                    assert error.hint_command == f"GRANT USAGE ON SCHEMA {quoted_schema} TO {restricted.user};"
                else:
                    assert error.hint_command == f"GRANT SELECT ON <schema>.{quoted} TO {restricted.user};"
                    assert error.hint == (
                        "Ask your database administrator to replace <schema> with the schema containing this table "
                        "and run the following statement."
                    )
                with admin.begin() as conn:
                    # The administrator supplies the schema rather than relying on their session's search_path.
                    conn.execute(text(error.hint_command.replace("<schema>", quoted_schema, 1)))
            reader = sqlalchemy.create_engine(restricted.to_sqlalchemy_url())
            try:
                with reader.connect() as conn:
                    assert conn.execute(text(query)).all() == []
            finally:
                reader.dispose()
        finally:
            with admin.begin() as conn:
                conn.execute(text(f"DROP SCHEMA {quoted_schema} CASCADE"))
    finally:
        admin.dispose()


def test_real_driver_grant_hint_quotes_the_role(real_dsn: Dsn, restricted_dsn: Dsn):
    admin = sqlalchemy.create_engine(real_dsn.to_sqlalchemy_url())
    original = admin.dialect.identifier_preparer.quote_identifier(restricted_dsn.user)
    username = f"{restricted_dsn.user}-Quoted"
    quoted = admin.dialect.identifier_preparer.quote_identifier(username)
    restricted = restricted_dsn.model_copy(update={"driver": real_dsn.driver, "user": username})
    try:
        with admin.begin() as conn:
            conn.execute(text(f"ALTER ROLE {original} RENAME TO {quoted}"))
        try:
            error = _classify_real_error(
                _query_error(restricted, f"SELECT 1 FROM {TESTING_DWH_TABLE_NAME} LIMIT 0"), restricted
            )
            assert isinstance(error, DwhPermissionDeniedError)
            assert error.hint_command == f"GRANT SELECT ON <schema>.{TESTING_DWH_TABLE_NAME} TO {quoted};"
            with admin.begin() as conn:
                conn.execute(text(error.hint_command.replace("<schema>", "public", 1)))
            reader = sqlalchemy.create_engine(restricted.to_sqlalchemy_url())
            try:
                with reader.connect() as conn:
                    assert conn.execute(text(f"SELECT 1 FROM {TESTING_DWH_TABLE_NAME} LIMIT 0")).all() == []
            finally:
                reader.dispose()
        finally:
            with admin.begin() as conn:
                conn.execute(text(f"ALTER ROLE {quoted} RENAME TO {original}"))
    finally:
        admin.dispose()


def test_real_driver_wrong_password(real_dsn):
    dsn = real_dsn.model_copy(update={"password": "definitely-wrong"})
    result = _classify_real_error(_connect_error(dsn), dsn)
    assert isinstance(result, DwhAuthenticationError)
    assert result.code == DwhErrorCode.AUTHENTICATION
    assert "password authentication failed" in result.message
    assert result.message.startswith("The data warehouse rejected the configured credentials: ")
    assert result.dwh_message is not None and "password authentication failed" in result.dwh_message


def test_real_driver_nonexistent_database(real_dsn):
    dsn = real_dsn.model_copy(update={"dbname": "nonexistent_db"})
    result = _classify_real_error(_connect_error(dsn), dsn)
    assert isinstance(result, DwhDatabaseDoesNotExistError)
    assert result.code == DwhErrorCode.DATABASE_NOT_FOUND
    assert result.message == 'The database "nonexistent_db" does not exist.'
    assert result.dwh_message is not None and 'database "nonexistent_db" does not exist' in result.dwh_message


@pytest.mark.parametrize(
    ("statement", "params", "sqlstate"),
    [
        pytest.param("SELECT CAST(:a AS integer)", PARAMS, "22P02", id="invalid-value"),
        pytest.param("SELECT 1 = CAST(:a AS text)", PARAMS, "42883", id="undefined-operator"),
        pytest.param("SELEC 1", {}, "42601", id="syntax-error"),
        pytest.param("SELECT 1 / 0", {}, "22012", id="division-by-zero"),
        pytest.param(
            "SELECT set_config('statement_timeout', :a, false)", {"a": "soon"}, "22023", id="invalid-other-setting"
        ),
    ],
)
def test_real_driver_sql_errors_are_left_unclassified(real_dsn: Dsn, statement: str, params: dict, sqlstate: str):
    """SQL/type errors may be bugs in generated SQL, so the actual driver exceptions must propagate."""
    error = _query_error(real_dsn, statement, params)
    assert isinstance(error.orig, psycopg.Error | psycopg2.Error)
    code = error.orig.sqlstate if isinstance(error.orig, psycopg.Error) else error.orig.pgcode
    assert code == sqlstate
    assert classify_dwh_error(error, real_dsn) is None
    assert classify_dwh_error(error.orig, real_dsn) is None


def test_real_driver_malformed_search_path(real_dsn: Dsn):
    error = _query_error(real_dsn, "SELECT set_config('search_path', :a, false)", {"a": 'public"; DROP'})
    result = _classify_real_error(error, real_dsn)
    assert isinstance(result, DwhConfigurationError)
    assert result.code == DwhErrorCode.CONFIGURATION
    assert result.message == "The configured search_path is not valid."
    assert result.hint == "Check the search_path in the data source configuration."
    assert result.dwh_message is not None and 'parameter "search_path"' in result.dwh_message


def test_real_driver_statement_timeout(real_dsn: Dsn):
    engine = sqlalchemy.create_engine(real_dsn.to_sqlalchemy_url())
    try:
        with engine.connect() as conn:
            conn.execute(text("SET LOCAL statement_timeout = '20ms'"))
            with pytest.raises(sqlalchemy.exc.DBAPIError) as excinfo:
                conn.execute(text("SELECT pg_sleep(10)"))
        result = _classify_real_error(excinfo.value, real_dsn)
        assert isinstance(result, DwhTimeoutError)
        assert result.code == DwhErrorCode.TIMEOUT
        assert result.message == "The data warehouse canceled the query before it finished."
        assert result.hint == "Try again, or ask your database administrator to raise the statement timeout."
    finally:
        engine.dispose()


def test_real_driver_connection_refused(real_dsn: Dsn):
    # Reserve a local port without listening so no other service can take it during the failed connection.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
        dsn = real_dsn.model_copy(update={"host": "127.0.0.1", "port": port})
        result = _classify_real_error(_connect_error(dsn), dsn)
    assert isinstance(result, DwhConnectionError)
    assert result.code == DwhErrorCode.CONNECTION
    # psycopg 3 prefixes libpq's text with "connection failed: ", which the message drops; the address stays so the
    # reader can see what was dialed.
    assert result.message == (
        f'The connection to the data warehouse failed: connection to server at "127.0.0.1", port {port} failed: '
        "Connection refused."
    )
    assert result.hint == "Check the host, port, and network settings in the data source configuration."
    assert result.dwh_message is not None and "Connection refused" in result.dwh_message


def test_real_driver_transaction_conflict(real_dsn: Dsn, dwh_schema: str):
    engine = sqlalchemy.create_engine(real_dsn.to_sqlalchemy_url())
    table = f"{dwh_schema}.participants"
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as first:
            first.execute(text(f"SELECT * FROM {table}")).all()
            with engine.begin() as second:
                second.execute(text(f"UPDATE {table} SET is_engaged = false WHERE id = 1"))
            with pytest.raises(sqlalchemy.exc.DBAPIError) as excinfo:
                first.execute(text(f"UPDATE {table} SET is_engaged = true WHERE id = 1"))
        result = _classify_real_error(excinfo.value, real_dsn)
        assert isinstance(result, DwhQueryError)
        assert result.code == DwhErrorCode.QUERY_FAILED
        assert result.message == "The data warehouse could not complete the query (SQLSTATE 40001)."
        assert result.hint == "Try again later."
        assert result.dwh_message == "could not serialize access due to concurrent update"
    finally:
        engine.dispose()


def test_real_driver_terminated_session(real_dsn: Dsn):
    engine = sqlalchemy.create_engine(real_dsn.to_sqlalchemy_url())
    try:
        with engine.connect() as victim, engine.connect() as admin:
            pid = victim.execute(text("SELECT pg_backend_pid()")).scalar_one()
            assert admin.execute(text("SELECT pg_terminate_backend(:pid, 5000)"), {"pid": pid}).scalar_one()
            with pytest.raises(sqlalchemy.exc.DBAPIError) as excinfo:
                victim.execute(text("SELECT 1"))
        result = _classify_real_error(excinfo.value, real_dsn)
        assert isinstance(result, DwhConnectionError)
        assert result.code == DwhErrorCode.CONNECTION
        assert result.dwh_message is not None
        assert "terminating connection" in result.dwh_message or "server closed the connection" in result.dwh_message
    finally:
        engine.dispose()
