"""Maps the exceptions raised while talking to a customer data warehouse onto the DwhError taxonomy.

classify_dwh_error() is a pure function: it inspects an exception and the data source configuration and returns a new
DwhError, or None when the exception may be our own doing. It never raises.
"""

import re
from collections.abc import Iterator

import google.api_core.exceptions
import google.auth.exceptions
import psycopg
import psycopg.errors
import psycopg2
import requests.exceptions
import sqlalchemy.dialects.postgresql
import sqlalchemy.exc

from xngin.apiserver.dns.safe_resolve import DnsLookupError, DnsLookupUnsafeError
from xngin.apiserver.exceptions_common import (
    CannotFindTableError,
    DwhAuthenticationError,
    DwhColumnNotFoundError,
    DwhConfigurationError,
    DwhConnectionError,
    DwhDatabaseDoesNotExistError,
    DwhDnsError,
    DwhError,
    DwhPermissionDeniedError,
    DwhQueryError,
    DwhTimeoutError,
)
from xngin.apiserver.settings import BqDsn, Dsn, Dwh

HINT_RUN_STATEMENT = "Ask your database administrator to run the following statement."
HINT_CHECK_SEARCH_PATH = "Check the search_path in the data source configuration."
HINT_CHECK_HOSTNAME = "Check the hostname in the data source configuration."
HINT_UNSAFE_HOSTNAME = "The hostname resolves to a private or reserved address, which Evidential does not connect to."
HINT_CHECK_CONNECTION = "Check the host, port, and network settings in the data source configuration."
HINT_CHECK_CREDENTIALS = "Check the username and password in the data source configuration."
HINT_CHECK_DATABASE = "Check the database name in the data source configuration."
HINT_CHECK_COLUMN = "Check the column name, and the columns in the table, and try again."
HINT_CHECK_BQ_CREDENTIALS = "Check the service account credentials in the data source configuration."
HINT_CHECK_BQ_DATASET = "Check the project and dataset in the data source configuration."
HINT_TRY_AGAIN_LATER = "Try again later."
HINT_STATEMENT_TIMEOUT = "Try again, or ask your database administrator to raise the statement timeout."

# Postgres primary messages name the object the query referred to. These parse that name back out.
_PG_RELATION_RE = re.compile(r'relation "(?P<name>[^"]+)" does not exist')
_PG_COLUMN_RE = re.compile(
    r'column (?:"(?P<quoted>[^"]+)"|(?P<bare>\S+)) (?:of relation "(?P<relation>[^"]+)" )?does not exist'
)
_PG_DENIED_RE = re.compile(r"permission denied for (?P<kind>table|relation|view|materialized view|schema) (?P<name>.+)")
_PG_DATABASE_RE = re.compile(r'database "(?P<name>[^"]+)" does not exist')
_PG_IDENTIFIER_PREPARER = sqlalchemy.dialects.postgresql.dialect().identifier_preparer
# Connect-time failures carry no SQLSTATE, only libpq's text. These phrases identify the rejected-credentials ones.
_PG_AUTH_PHRASES = ("password authentication failed", "no pg_hba.conf entry", "authentication failed")
# libpq's connect_timeout reports this phrase; psycopg 3 additionally raises its own ConnectionTimeout type.
_PG_CONNECT_TIMEOUT_PHRASE = "timeout expired"
# psycopg 3 prefixes libpq's connection error text with this; psycopg2 does not.
_PG_CONNECT_PREFIX = "connection failed: "
# SQLSTATE classes the warehouse raises for reasons of its own, whatever the query: transaction rollback (a
# serialization failure or a conflict with recovery on a replica), insufficient resources, and system errors. Their
# primary messages describe the server's state, not the query's data.
_PG_WAREHOUSE_SIDE_CLASSES = ("40", "53", "58")

# BigQuery messages for a missing or unreadable table name it as project:dataset.table.
_BQ_TABLE_RE = re.compile(r"Table (?P<project>[^\s:]+):(?P<dataset>[^\s.]+)\.(?P<table>[^\s:]+)")
_BQ_PROJECT_RE = re.compile(r"Project (?P<project>[^\s:]+):")
# "Unrecognized name: col at [1:8]" or "Unrecognized name: col; Did you mean cola? at [1:8]".
_BQ_UNRECOGNIZED_NAME_RE = re.compile(r"Unrecognized name: (?P<name>[^\s;]+)")
# google.api_core builds REST error messages as "{METHOD} {url}: {server message}".
_BQ_HTTP_PREFIX_RE = re.compile(r"^(?:GET|POST|PUT|PATCH|DELETE) https?://\S+: ")
# QueryJob appends "\n\nLocation: {location}\nJob ID: {job_id}\n" to the message of a failed job.
_BQ_JOB_FOOTER = "\n\nLocation: "


def classify_dwh_error(exc: BaseException, dwh_config: Dwh) -> DwhError | None:
    """Returns the DwhError that describes exc, or None when the caller should re-raise exc unchanged.

    None is returned for exceptions that may be our own doing, and for an exc that is already a DwhError or the
    builtin TimeoutError that DwhSession already converts. That includes warehouse errors that a query we generated
    could have caused, such as a type mismatch or a syntax error: only errors the data source's owner can act on,
    or that the warehouse raises for reasons of its own, are classified. When in doubt an error stays unclassified,
    so the caller re-raises it as a 500 that reaches Sentry, and classified errors are never re-wrapped.

    The driver's error is found by looking at exc itself, then at the DBAPI exception SQLAlchemy wrapped
    (exc.orig), then at what that wrapped in turn (orig.args[0]), then at exc.__cause__. The args[0] step matters
    for BigQuery, whose DB-API raises DatabaseError(google_exc) without `from`, so the Google exception is not a
    __cause__.

    The returned message never contains SQL text or bind parameters; dwh_message carries the driver's primary
    message only for categories whose messages name schema objects, hosts, or endpoints rather than data.
    """
    if isinstance(exc, DwhError | TimeoutError):
        return None
    seen: list[BaseException] = []
    for candidate in _candidates(exc):
        if any(candidate is s for s in seen):
            continue
        seen.append(candidate)
        classified = _classify_one(candidate, dwh_config)
        if classified is not None:
            return classified
    return None


def _is_postgres_database_not_found_error(exc: BaseException) -> bool:
    """Returns true when the exception's text indicates a Postgres database does not exist.

    Postgres reports this at connect time without a SQLSTATE, so only the text identifies it. Works on the raw
    psycopg error and on the sqlalchemy.exc.OperationalError that wraps it, since both carry the text in args[0].
    """
    return (
        len(exc.args) > 0
        and isinstance(exc.args[0], str)
        and "FATAL:  database" in exc.args[0]
        and "does not exist" in exc.args[0]
    )


def _candidates(exc: BaseException) -> Iterator[BaseException]:
    """Yields exc and the exceptions it wraps, nearest first."""
    yield exc
    if isinstance(exc, sqlalchemy.exc.DBAPIError) and exc.orig is not None:
        orig = exc.orig
        yield orig
        if orig.args and isinstance(orig.args[0], BaseException):
            yield orig.args[0]
    if exc.__cause__ is not None:
        yield exc.__cause__


def _classify_one(exc: BaseException, dwh_config: Dwh) -> DwhError | None:
    if isinstance(exc, DwhError):
        return None
    if isinstance(exc, DnsLookupError):
        return _classify_dns(exc)
    if isinstance(exc, sqlalchemy.exc.NoSuchTableError):
        return CannotFindTableError(str(exc))
    if isinstance(exc, psycopg.Error | psycopg2.Error):
        return _classify_postgres(exc, dwh_config)
    if isinstance(exc, google.api_core.exceptions.GoogleAPIError):
        return _classify_google_api(exc, dwh_config)
    # TransportError is a GoogleAuthError, so it must be recognized before the credential failures.
    if isinstance(exc, google.auth.exceptions.TransportError):
        return DwhConnectionError(
            "Could not reach Google to authenticate the BigQuery credentials.",
            hint=HINT_TRY_AGAIN_LATER,
            dwh_message=_first_line(exc),
        )
    if isinstance(exc, google.auth.exceptions.GoogleAuthError):
        return DwhAuthenticationError(
            "Google rejected the configured BigQuery credentials.",
            hint=HINT_CHECK_BQ_CREDENTIALS,
            dwh_message=_first_line(exc),
        )
    if isinstance(exc, requests.exceptions.RequestException):
        return DwhConnectionError("Could not reach BigQuery.", hint=HINT_TRY_AGAIN_LATER, dwh_message=_first_line(exc))
    return None


def _classify_dns(exc: DnsLookupError) -> DwhDnsError:
    hint = HINT_UNSAFE_HOSTNAME if isinstance(exc, DnsLookupUnsafeError) else HINT_CHECK_HOSTNAME
    return DwhDnsError(str(exc), hint=hint, dwh_message=exc.host)


def _classify_postgres(exc: psycopg.Error | psycopg2.Error, dwh_config: Dwh) -> DwhError | None:
    """Classifies a psycopg 3 or psycopg2 error, which carry the same facts under different attribute names."""
    sqlstate = exc.sqlstate if isinstance(exc, psycopg.Error) else exc.pgcode
    # An error built without a server result (as in tests) has an empty diag; the first line of its text is the
    # primary message. Later lines quote the statement, which must never reach a user.
    primary = exc.diag.message_primary or _first_line(exc)
    if sqlstate is None:
        if isinstance(exc, psycopg.OperationalError | psycopg2.OperationalError):
            return _classify_postgres_connect_failure(exc, primary, dwh_config)
        # A client-side driver error (a closed connection, a misuse of the cursor) is not the warehouse's answer.
        return None
    if sqlstate == "42501":
        return _permission_denied(primary, dwh_config)
    if sqlstate == "42P01":
        match = _PG_RELATION_RE.search(primary)
        error = CannotFindTableError(match["name"] if match else "unknown")
        error.dwh_message = primary
        return error
    if sqlstate == "42703":
        return _column_not_found(primary, exc.diag.message_hint)
    if sqlstate == "3D000":
        return _database_not_found(primary, dwh_config)
    if sqlstate.startswith("28"):
        return _authentication_failed(primary)
    if sqlstate == "57014":
        return DwhTimeoutError("The data warehouse canceled the query before it finished.", hint=HINT_STATEMENT_TIMEOUT)
    if sqlstate.startswith("08"):
        return _connection_failed(primary)
    if sqlstate.startswith("57P"):
        # Operator intervention: the server is shutting down, starting up, or ended the session.
        return DwhConnectionError(
            f"The data warehouse ended the connection (SQLSTATE {sqlstate}).",
            hint=HINT_TRY_AGAIN_LATER,
            dwh_message=primary,
        )
    if sqlstate.startswith(_PG_WAREHOUSE_SIDE_CLASSES):
        return DwhQueryError(
            f"The data warehouse could not complete the query (SQLSTATE {sqlstate}).",
            hint=HINT_TRY_AGAIN_LATER,
            dwh_message=primary,
        )
    if sqlstate == "22023" and 'parameter "search_path"' in primary:
        # Raised while applying the data source's search_path to a new connection; the value is the customer's
        # own setting, so the server's message, which echoes it, is safe to show.
        return DwhConfigurationError(
            "The configured search_path is not valid.", hint=HINT_CHECK_SEARCH_PATH, dwh_message=primary
        )
    # Anything else, such as a syntax error, an undefined function, or a value of the wrong type, may be a bug in
    # the SQL we generated.
    return None


def _classify_postgres_connect_failure(
    exc: psycopg.OperationalError | psycopg2.OperationalError, text: str, dwh_config: Dwh
) -> DwhError:
    """Classifies an OperationalError without a SQLSTATE: libpq reports connection failures as text only."""
    lowered = text.lower()
    if isinstance(exc, psycopg.errors.ConnectionTimeout) or _PG_CONNECT_TIMEOUT_PHRASE in lowered:
        return DwhTimeoutError("Connecting to the data warehouse timed out.", hint=HINT_CHECK_CONNECTION)
    if _is_postgres_database_not_found_error(exc):
        return _database_not_found(text, dwh_config)
    if any(phrase in lowered for phrase in _PG_AUTH_PHRASES):
        return _authentication_failed(text)
    return _connection_failed(text)


def _permission_denied(primary: str, dwh_config: Dwh) -> DwhPermissionDeniedError:
    user = dwh_config.user if isinstance(dwh_config, Dsn) else None
    who = f'User "{user}"' if user else "The configured user"
    match = _PG_DENIED_RE.search(primary)
    if match and match["kind"] == "schema":
        return _schema_usage_denied(match["name"], user, who, dwh_message=primary)
    if match:
        table = match["name"]
        # Postgres reports the raw relation name, without its schema; a dot belongs to the name itself.
        return DwhPermissionDeniedError(
            f'{who} does not have permission to read the table "{table}".',
            hint="Ask your database administrator to replace <schema> with the schema containing this table "
            "and run the following statement.",
            hint_command=_grant_statement(f"GRANT SELECT ON <schema>.{_sql_identifier(table)}", user),
            dwh_message=primary,
            table_name=table,
        )
    return DwhPermissionDeniedError(
        f"{who} does not have permission to run this query.",
        hint=f"Ask your database administrator to grant {_describe_role(user)} read access to the tables this "
        "data source uses.",
        dwh_message=primary,
    )


def _schema_usage_denied(schema: str, user: str | None, who: str, *, dwh_message: str) -> DwhPermissionDeniedError:
    return DwhPermissionDeniedError(
        f'{who} does not have permission to use the schema "{schema}".',
        hint=HINT_RUN_STATEMENT,
        hint_command=_grant_statement(f"GRANT USAGE ON SCHEMA {_sql_identifier(schema)}", user),
        dwh_message=dwh_message,
    )


def _grant_statement(grant: str, user: str | None) -> str:
    return f"{grant} TO {_sql_identifier(user) if user else "<the data source's user>"};"


def _describe_role(user: str | None) -> str:
    return f"the user {_sql_identifier(user)}" if user else "the data source's user"


def _sql_identifier(name: str) -> str:
    """Quotes names that need it, including reserved words, so the GRANT in a hint is valid as written."""
    return _PG_IDENTIFIER_PREPARER.quote(name)


def _column_not_found(primary: str, pg_hint: str | None) -> DwhColumnNotFoundError:
    match = _PG_COLUMN_RE.search(primary)
    column = (match["quoted"] or match["bare"]) if match else None
    message = f'The column "{column}" does not exist.' if column else "The column does not exist."
    # Postgres sometimes suggests the column the query probably meant; that names schema objects only.
    return DwhColumnNotFoundError(
        message,
        hint=pg_hint or HINT_CHECK_COLUMN,
        dwh_message=primary,
        table_name=match["relation"] if match else None,
    )


def _database_not_found(text: str, dwh_config: Dwh) -> DwhDatabaseDoesNotExistError:
    match = _PG_DATABASE_RE.search(text)
    if match:
        name = match["name"]
    else:
        name = dwh_config.dbname if isinstance(dwh_config, Dsn) else None
    message = f'The database "{name}" does not exist.' if name else "The database does not exist."
    return DwhDatabaseDoesNotExistError(message, hint=HINT_CHECK_DATABASE, dwh_message=text)


def _authentication_failed(text: str) -> DwhAuthenticationError:
    return DwhAuthenticationError(
        f"The data warehouse rejected the configured credentials: {_libpq_reason(text)}.",
        hint=HINT_CHECK_CREDENTIALS,
        dwh_message=text,
    )


def _connection_failed(text: str) -> DwhConnectionError:
    return DwhConnectionError(
        f"The connection to the data warehouse failed: {_libpq_reason(text)}.",
        hint=HINT_CHECK_CONNECTION,
        dwh_message=text,
    )


def _libpq_reason(text: str) -> str:
    """Reduces libpq's connection error text to the reason the server or the OS gave.

    'connection failed: connection to server at "h", port p failed: FATAL:  password authentication failed for
    user "u"' becomes 'password authentication failed for user "u"'. Text without a server FATAL keeps the
    address, which tells the reader what was dialed.
    """
    _, fatal, reason = text.rpartition("FATAL:")
    if not fatal:
        reason = text.removeprefix(_PG_CONNECT_PREFIX)
    return reason.strip().rstrip(".")


def _classify_google_api(exc: google.api_core.exceptions.GoogleAPIError, dwh_config: Dwh) -> DwhError | None:
    text = _google_text(exc)
    project = dwh_config.project_id if isinstance(dwh_config, BqDsn) else None
    dataset = dwh_config.dataset_id if isinstance(dwh_config, BqDsn) else None
    where = f"{project}.{dataset}" if project and dataset else "the configured dataset"
    reason = _google_reason(exc)
    if isinstance(exc, google.api_core.exceptions.Forbidden):
        # BigQuery also answers 403 for quota, rate limit, and billing problems; only accessDenied is about access.
        if reason is not None and reason != "accessDenied":
            return _google_rejected(exc, reason, text)
        if "bigquery.jobs.create" in text:
            # The account cannot run queries at all in the project it was configured with, whatever its dataset
            # access: "Access Denied: Project p: User does not have bigquery.jobs.create permission in project p."
            project_match = _BQ_PROJECT_RE.search(text)
            job_project = project_match["project"] if project_match else project or "the configured project"
            return DwhPermissionDeniedError(
                f"The configured BigQuery credentials cannot run queries in the project {job_project}.",
                hint=f"Grant the service account the BigQuery Job User role on the project {job_project}.",
                dwh_message=text,
            )
        table_match = _BQ_TABLE_RE.search(text)
        table = table_match["table"] if table_match else None
        if table_match:
            project = table_match["project"]
            dataset = table_match["dataset"]
            where = f"{project}.{dataset}"
        what = f'the table "{table}" in {where}' if table else where
        if project and dataset:
            hint = (
                f"Grant the service account the BigQuery Data Viewer role on the dataset {dataset} in the project "
                f"{project}."
            )
        else:
            hint = "Grant the service account the BigQuery Data Viewer role on the dataset."
        return DwhPermissionDeniedError(
            f"The configured BigQuery credentials do not have permission to read {what}.",
            hint=hint,
            dwh_message=text,
            table_name=table,
        )
    if isinstance(exc, google.api_core.exceptions.Unauthorized):
        return DwhAuthenticationError(
            "BigQuery rejected the configured credentials.", hint=HINT_CHECK_BQ_CREDENTIALS, dwh_message=text
        )
    if isinstance(exc, google.api_core.exceptions.NotFound):
        # Reflection turns a missing table into NoSuchTableError, but a query against one arrives as NotFound.
        table_match = _BQ_TABLE_RE.search(text) if "Not found: Table" in text else None
        if table_match:
            error = CannotFindTableError(table_match["table"])
            error.dwh_message = text
            return error
        return DwhDatabaseDoesNotExistError(
            f"The BigQuery dataset {where} does not exist.", hint=HINT_CHECK_BQ_DATASET, dwh_message=text
        )
    if isinstance(exc, google.api_core.exceptions.BadRequest):
        name_match = _BQ_UNRECOGNIZED_NAME_RE.search(text)
        if name_match:
            return DwhColumnNotFoundError(
                f'The column "{name_match["name"]}" does not exist.', hint=HINT_CHECK_COLUMN, dwh_message=text
            )
        # An invalid query may be a bug in the SQL we generated, and its message may quote the query's values.
        if reason is None or reason in {"invalid", "invalidQuery"}:
            return None
        return _google_rejected(exc, reason, text)
    if isinstance(exc, google.api_core.exceptions.DeadlineExceeded):
        return DwhTimeoutError("The BigQuery request did not finish before its deadline.")
    if isinstance(exc, google.api_core.exceptions.ServiceUnavailable):
        return DwhConnectionError("BigQuery is unavailable.", hint=HINT_TRY_AGAIN_LATER, dwh_message=text)
    if isinstance(exc, google.api_core.exceptions.RetryError):
        if isinstance(exc.cause, google.api_core.exceptions.DeadlineExceeded):
            return DwhTimeoutError("The BigQuery request did not finish before its deadline.")
        return DwhConnectionError("Could not reach BigQuery.", hint=HINT_TRY_AGAIN_LATER, dwh_message=text)
    return _google_rejected(exc, reason, text)


def _google_rejected(exc: google.api_core.exceptions.GoogleAPIError, reason: str | None, text: str) -> DwhQueryError:
    """Reports BigQuery's own reason code and message rather than explaining each one.

    Only used for reasons that describe the project's state (a quota, billing, an overloaded backend), never the
    query, so the server's message is safe to show.
    """
    return DwhQueryError(f"BigQuery rejected the request ({reason or type(exc).__name__}).", dwh_message=text)


def _google_first_error(exc: google.api_core.exceptions.GoogleAPIError) -> dict | None:
    """The first structured error of a REST response, if any; gRPC-origin exceptions carry other objects here."""
    errors = getattr(exc, "errors", None)
    if isinstance(errors, list | tuple) and errors and isinstance(errors[0], dict):
        return errors[0]
    return None


def _google_reason(exc: google.api_core.exceptions.GoogleAPIError) -> str | None:
    """BigQuery's machine-readable reason code, such as accessDenied or quotaExceeded."""
    first = _google_first_error(exc)
    reason = first.get("reason") if first else None
    return reason if isinstance(reason, str) and reason else None


def _google_text(exc: google.api_core.exceptions.GoogleAPIError) -> str:
    """The server's own message for a Google exception, without the decoration the client libraries add.

    Real exceptions differ from ones constructed directly: google.api_core prefixes REST error messages with the
    HTTP method and URL, and a failed QueryJob appends its location and job ID. The raw server message is kept in
    the first entry of ``errors`` when the response carried one, so prefer that, then strip the decoration from
    ``message`` otherwise.
    """
    first = _google_first_error(exc)
    if first is not None:
        server_message = first.get("message")
        if isinstance(server_message, str) and server_message:
            return _strip_google_decoration(server_message)
    message = getattr(exc, "message", None)
    text = message if isinstance(message, str) and message else _first_line(exc)
    return _strip_google_decoration(text)


def _strip_google_decoration(text: str) -> str:
    text = text.split(_BQ_JOB_FOOTER, 1)[0]
    return _BQ_HTTP_PREFIX_RE.sub("", text, count=1).strip()


def _first_line(exc: BaseException) -> str:
    """The first line of an exception's own text, or its type name when it has none."""
    text = exc.args[0] if exc.args and isinstance(exc.args[0], str) else str(exc)
    lines = text.strip().splitlines()
    return lines[0].strip() if lines else type(exc).__name__
