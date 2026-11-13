from enum import StrEnum
from typing import ClassVar


class LateValidationError(Exception):
    """Raised by API request validation failures that can only occur late in processing.

    Examples:
    - datetime value validations cannot happen until we know we are dealing with a datetime field,
    and that information is not available until we have table reflection data.
    - verifying requested dwh fields are present in its configuration for that use.
    """


class DwhErrorCode(StrEnum):
    """Stable, machine-readable categories of data warehouse errors."""

    # Network, TLS, or SQLSTATE class 08 failures, other failures while connecting, or the server ending the session
    # (SQLSTATE 57P01-57P05).
    CONNECTION = "dwh_connection"
    # The hostname does not resolve, or resolves to an address we refuse to connect to.
    DNS = "dwh_dns"
    # SQLSTATE class 28, an authentication failure while connecting, or invalid BigQuery credentials.
    AUTHENTICATION = "dwh_authentication"
    # SQLSTATE 3D000, or a BigQuery dataset NotFound.
    DATABASE_NOT_FOUND = "dwh_database_not_found"
    # SQLSTATE 42P01, or SQLAlchemy's NoSuchTableError.
    TABLE_NOT_FOUND = "dwh_table_not_found"
    # SQLSTATE 42703.
    COLUMN_NOT_FOUND = "dwh_column_not_found"
    # SQLSTATE 42501, or a BigQuery access denial.
    PERMISSION_DENIED = "dwh_permission_denied"
    # Our own deadline expired, SQLSTATE 57014 (query canceled), or the driver's connect timeout.
    TIMEOUT = "dwh_timeout"
    # The warehouse could not complete a query for reasons of its own, such as exhausted resources, a quota, or
    # billing; the message carries the warehouse's own code. Errors that may come from our own SQL are not DwhErrors.
    QUERY_FAILED = "dwh_query_failed"
    # A data source setting cannot be applied: SQLSTATE 22023 for an invalid search_path, or 3F000 for a search_path
    # schema that Redshift cannot find.
    CONFIGURATION = "dwh_configuration"


class DwhError(Exception):
    """Base class for errors raised by interactions with a customer data warehouse.

    Subclasses set a stable code. Every warehouse error answers 424 Failed Dependency: the request itself was
    acceptable, and a system Evidential depends on could not be reached, refused it, or lacks what it named; the
    code says which. One status for all of them keeps warehouse failures apart from the 502s and 504s that proxies
    emit on their own and may retry, and keeps 404 meaning that the Evidential resource in the URL does not exist.

    The message is user-facing and must never contain SQL text or bind parameters. dwh_message is the driver's
    primary message when it is safe to show. hint is an actionable next step, and hint_command a statement for the
    warehouse administrator to run when there is one.
    """

    code: ClassVar[DwhErrorCode]
    http_status: ClassVar[int] = 424

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        hint_command: str | None = None,
        dwh_message: str | None = None,
        table_name: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.hint_command = hint_command
        self.dwh_message = dwh_message
        self.table_name = table_name

    def __str__(self):
        return self.message


class DwhConnectionError(DwhError):
    """Raised when there is a connection error to the data warehouse."""

    code = DwhErrorCode.CONNECTION


class DwhDnsError(DwhError):
    """Raised when the data warehouse hostname does not resolve, or resolves to an address we refuse."""

    code = DwhErrorCode.DNS


class DwhAuthenticationError(DwhError):
    """Raised when the data warehouse rejects the configured credentials."""

    code = DwhErrorCode.AUTHENTICATION


class DwhDatabaseDoesNotExistError(DwhError):
    """Raised when the target database or dataset does not exist."""

    code = DwhErrorCode.DATABASE_NOT_FOUND


class CannotFindTableError(DwhError):
    """Raised when we cannot find a table in the database.

    The message names only the missing table. It must never list the other tables in the warehouse:
    their names are customer data and this message ends up in logs, Sentry titles and API responses.
    """

    code = DwhErrorCode.TABLE_NOT_FOUND

    def __init__(self, table_name: str):
        super().__init__(
            f"The table '{table_name}' does not exist. Check the table name, and the tables in the database, "
            f"and try again.",
            table_name=table_name,
        )


class UnsupportedTableNameError(CannotFindTableError):
    """Raised for a table name that the backend does not allow.

    BigQuery: BigQuery does not allow table names with a dot because it would be ambiguous with a dataset-scoped table.
    """

    def __init__(self, table_name: str):
        super().__init__(table_name)
        self.message = f"The table name '{table_name}' is not a valid BigQuery table name."


class DwhColumnNotFoundError(DwhError):
    """Raised when a query refers to a column the data warehouse table does not have."""

    code = DwhErrorCode.COLUMN_NOT_FOUND


class DwhPermissionDeniedError(DwhError):
    """Raised when the data warehouse user lacks a privilege the query needs."""

    code = DwhErrorCode.PERMISSION_DENIED


class DwhTimeoutError(DwhError, TimeoutError):
    """Raised when a data warehouse interaction outlives the deadline its caller gave it.

    Subclasses TimeoutError so that `except TimeoutError` still catches it, while the type name
    remains informative wherever we report the failure back to a user.
    """

    code = DwhErrorCode.TIMEOUT


class DwhConfigurationError(DwhError):
    """Raised when a setting of the data source itself, such as its search_path, cannot be applied."""

    code = DwhErrorCode.CONFIGURATION


class DwhQueryError(DwhError):
    """Raised when the data warehouse cannot complete a query for a reason that is not a bug in our SQL."""

    code = DwhErrorCode.QUERY_FAILED
