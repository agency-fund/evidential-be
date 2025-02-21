class LateValidationError(Exception):
    """Raised by API request validation failures that can only occur late in processing.

    Examples:
    - datetime value validations cannot happen until we know we are dealing with a datetime field,
    and that information is not available until we have table reflection data.
    - verifying requested dwh fields are present in its configuration for that use.
    """
