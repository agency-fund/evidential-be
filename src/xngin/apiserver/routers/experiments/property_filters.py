import uuid
from datetime import date, datetime, time, timedelta
from typing import Literal

from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import DataType, Filter, PropertyValueTypes
from xngin.apiserver.routers.common_enums import Relation


def str_to_date_or_datetime(
    col_name: str,
    s: int | float | str | date | datetime | None,
    target_type: Literal["date", "datetime"],
) -> date | datetime | None:
    """Convert an ISO8601 string to a date or datetime based on target_type.

    LateValidationError is raised if the ISO8601 string specifies a non-UTC timezone.

    For "datetime": microseconds are truncated to zero for maximum compatibility between backends.
        If `s` is already a datetime, it is returned as-is, but with microseconds set to zero.
    For "date": datetime strings are converted to dates, dropping time information.
        If `s` is already a date, it is returned as-is.
    """
    if s is None:
        return None

    if isinstance(s, datetime):
        return s.date() if target_type == "date" else s.replace(microsecond=0)

    if isinstance(s, date):
        # convert date to datetime at midnight if target_type is datetime
        return s if target_type == "date" else datetime.combine(s, time.min)

    if not isinstance(s, str):
        raise LateValidationError(
            f"{col_name}: {target_type}-type filter values must be strings containing an ISO8601 formatted date."
        )

    # Always parse as datetime first to validate timezone
    try:
        parsed = datetime.fromisoformat(s).replace(microsecond=0)
    except (ValueError, TypeError) as exc:
        raise LateValidationError(
            f"{col_name}: {target_type}-type filter values must be strings containing an ISO8601 formatted date."
        ) from exc

    if parsed.tzinfo:
        offset = parsed.tzinfo.utcoffset(parsed)
        if offset != timedelta():  # 0 timedelta is equivalent to UTC
            raise LateValidationError(
                f"{col_name}: {target_type}-type filter values must be in UTC, and not include timezone offsets: {s}"
            )
        parsed = parsed.replace(tzinfo=None)

    return parsed.date() if target_type == "date" else parsed


def passes_filters(props: dict[str, PropertyValueTypes], fields: dict[str, DataType], filters: list[Filter]) -> bool:
    """Check that a list of properties passes the list of filtering criteria."""
    if len(filters) == 0:
        return True

    for f in filters:
        field_type = fields.get(f.field_name)
        if not _passes_filter(f, field_type, props.get(f.field_name)):
            return False

    return True


def _passes_filter(exp_filter: Filter, field_type: DataType | None, value: PropertyValueTypes) -> bool:
    """Check that a value passes a filter."""
    py_value = validate_filter_value(exp_filter.field_name, value, field_type)
    parsed_values = [validate_filter_value(exp_filter.field_name, v, field_type) for v in exp_filter.value]

    match exp_filter.relation:
        case Relation.INCLUDES:
            return py_value in parsed_values
        case Relation.EXCLUDES:
            return py_value not in parsed_values
        case Relation.BETWEEN:
            if len(exp_filter.value) == 3 and py_value is None:
                return True

            if not isinstance(py_value, (int, float, datetime, date, type(None))):
                raise LateValidationError("BETWEEN relation is only supported for int/float/datetime/date fields.")

            match parsed_values:
                case (left, None):
                    return py_value >= left  # type: ignore
                case (None, right):
                    return py_value <= right  # type: ignore
                case (left, right):
                    return left <= py_value <= right  # type: ignore
                case _:
                    raise LateValidationError(f"Invalid between value: {exp_filter.value}")


def validate_filter_value(
    field_name: str, value: PropertyValueTypes, field_type: DataType | None
) -> str | int | float | bool | datetime | date | None:
    """Validate a value is of the appropriate type and possibly transform into the appropriate Python type.

    Raises:
        LateValidationError if:
        - field_type is missing
        - the value is not of the appropriate input type for the target DataType
        - the value is not formatted correctly for the target DataType
          (e.g. malformed uuid string, or a date/datetime string with a non-UTC timezone).
    """
    if not field_type:
        raise LateValidationError(f"Field {field_name} data type is missing (field not found?).")

    if value is None:
        return None

    match field_type:
        case DataType.BOOLEAN:
            if not isinstance(value, bool):
                raise LateValidationError("Boolean input is not a boolean.")
            return value

        case DataType.CHARACTER_VARYING:
            if not isinstance(value, str):
                raise LateValidationError("varchar input is not a string.")
            return value

        case DataType.UUID:
            if not isinstance(value, str):
                raise LateValidationError("UUID input must be a valid UUID string.")
            try:
                return str(uuid.UUID(value))  # must pass parsing but keep as a string
            except ValueError as exc:
                raise LateValidationError("UUID input must be a valid UUID string.") from exc

        case DataType.INTEGER:
            if not isinstance(value, int):
                raise LateValidationError("Integer input must be an int.")
            return value

        case DataType.DOUBLE_PRECISION | DataType.NUMERIC:
            if not isinstance(value, (int, float)):
                raise LateValidationError("Double/Numeric input must be an integer or float.")
            return value

        case DataType.BIGINT:
            if not isinstance(value, (int, str)):  # int for backwards compatibility
                raise LateValidationError("Bigint input must be a string to be converted to a bigint.")
            return int(value)

        case DataType.DATE:
            return str_to_date_or_datetime(field_name, value, "date")

        case DataType.TIMESTAMP_WITH_TIMEZONE:
            return str_to_date_or_datetime(field_name, value, "datetime")

        case DataType.TIMESTAMP_WITHOUT_TIMEZONE:
            return str_to_date_or_datetime(field_name, value, "datetime")

        case _:
            raise LateValidationError(f"Unsupported field type: {field_type}")
