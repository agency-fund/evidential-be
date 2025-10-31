import uuid
from datetime import date, datetime

from xngin.apiserver.dwh.queries import str_to_date_or_datetime
from xngin.apiserver.routers.common_api_types import DataType, Filter, PropertyValueTypes
from xngin.apiserver.routers.common_enums import Relation


def passes_filters(props: dict[str, PropertyValueTypes], fields: dict[str, DataType], filters: list[Filter]) -> bool:
    """Check that a list of properties passes the list of filtering criteria."""
    if len(filters) == 0:
        return True

    for f in filters:
        field_type = fields.get(f.field_name)
        if not field_type:
            raise ValueError(f"Field {f.field_name} not found in participant type.")

        if not _passes_filter(f, field_type, props.get(f.field_name)):
            return False

    return True


def _passes_filter(exp_filter: Filter, field_type: DataType, value: PropertyValueTypes) -> bool:
    """Check that a value passes a filter."""
    py_value = _validate_value(exp_filter.field_name, value, field_type)
    parsed_values = [_validate_value(exp_filter.field_name, v, field_type) for v in exp_filter.value]

    match exp_filter.relation:
        case Relation.INCLUDES:
            return py_value in parsed_values
        case Relation.EXCLUDES:
            return py_value not in parsed_values
        case Relation.BETWEEN:
            if len(exp_filter.value) == 3 and py_value is None:
                return True

            if not isinstance(py_value, (int, float, datetime, date, type(None))):
                raise TypeError("BETWEEN relation is only supported for int/float/datetime/date fields.")

            match parsed_values:
                case (left, None):
                    return py_value >= left  # type: ignore
                case (None, right):
                    return py_value <= right  # type: ignore
                case (left, right):
                    return left <= py_value <= right  # type: ignore
                case _:
                    raise ValueError(f"Invalid between value: {exp_filter.value}")


def _validate_value(
    field_name: str, value: PropertyValueTypes, field_type: DataType
) -> str | int | float | bool | datetime | date | None:
    """Validate a value is of the appropriate type and possibly cast it to the appropriate Python type."""
    if value is None:
        return None

    match field_type:
        case DataType.BOOLEAN:
            if not isinstance(value, bool):
                raise TypeError("Boolean input is not a boolean.")
            return value

        case DataType.CHARACTER_VARYING:
            if not isinstance(value, str):
                raise TypeError("varchar input is not a string.")
            return value

        case DataType.UUID:
            if not isinstance(value, str):
                raise TypeError("UUID input must be a valid UUID string.")
            return str(uuid.UUID(value))  # must pass parsing but keep as a string

        case DataType.INTEGER:
            if not isinstance(value, int):
                raise TypeError("Integer input must be an int.")
            return value

        case DataType.DOUBLE_PRECISION | DataType.NUMERIC:
            if not isinstance(value, (int, float)):
                raise TypeError("Double/Numeric input must be an integer or float.")
            return float(value)

        case DataType.BIGINT:
            if not isinstance(value, (int, str)):  # int for backwards compatibility
                raise TypeError("Bigint input must be a string to be converted to a bigint.")
            return int(value)

        case DataType.DATE:
            if not isinstance(value, str):
                raise TypeError("Date input must be an ISO8601 string to be converted to a date.")
            return str_to_date_or_datetime(field_name, value, "date")

        case DataType.TIMESTAMP_WITH_TIMEZONE:
            if not isinstance(value, str):
                raise TypeError("timestamp_tz input must be an ISO8601 string to be converted to a date.")
            return str_to_date_or_datetime(field_name, value, "datetime")

        case DataType.TIMESTAMP_WITHOUT_TIMEZONE:
            if not isinstance(value, str):
                raise TypeError("timestamp input must be an ISO8601 string to be converted to a date.")
            return str_to_date_or_datetime(field_name, value, "datetime")

        case _:
            raise ValueError(f"Unsupported field type: {field_type}")
