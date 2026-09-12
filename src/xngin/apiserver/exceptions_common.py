class LateValidationError(Exception):
    """Raised by API request validation failures that can only occur late in processing.

    Examples:
    - datetime value validations cannot happen until we know we are dealing with a datetime field,
    and that information is not available until we have table reflection data.
    - verifying requested dwh fields are present in its configuration for that use.
    """


class DwhDatabaseDoesNotExistError(Exception):
    """Raised when the target database or dataset does not exist."""


class DwhTimeoutError(TimeoutError):
    """Raised when a data warehouse interaction outlives the deadline its caller gave it.

    Subclasses TimeoutError so that `except TimeoutError` still catches it, while the type name
    remains informative wherever we report the failure back to a user.
    """


class DwhConnectionError(Exception):
    """Raised when there is a connection error to the data warehouse."""

    def __init__(self, original_error: Exception):
        super().__init__(f"CONNECTION ERROR: {type(original_error).__name__} | {original_error!s}")
