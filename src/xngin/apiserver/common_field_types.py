import re
from typing import Annotated

from pydantic import BeforeValidator, Field
from pydantic_core.core_schema import ValidationInfo

VALID_SQL_COLUMN_REGEX = r"^[a-zA-Z_][a-zA-Z0-9_]*$"


def validate_can_be_used_as_column_name(value: str, info: ValidationInfo) -> str:
    """Validates value is usable as a SQL column name."""
    if not isinstance(value, str):
        raise ValueError(f"{info.field_name} must be a string")  # noqa: TRY004
    if not re.match(VALID_SQL_COLUMN_REGEX, value):
        raise ValueError(
            f"{info.field_name} must start with letter/underscore and contain only letters, numbers, underscores"
        )
    return value


FieldName = Annotated[
    str,
    BeforeValidator(validate_can_be_used_as_column_name),
    Field(json_schema_extra={"pattern": VALID_SQL_COLUMN_REGEX}, examples=["field_name"]),
]
