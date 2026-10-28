import unicodedata
from typing import Annotated

from pydantic import BeforeValidator, Field, StringConstraints
from pydantic_core.core_schema import ValidationInfo

from xngin.apiserver.limits import MAX_LENGTH_OF_FIELD_NAME


def has_control_characters(name: str) -> bool:
    """Whether name has a control character, such as a tab, newline or NUL, which field names may not contain."""
    return any(unicodedata.category(c) == "Cc" for c in name)


def validate_column_name(value: str, info: ValidationInfo) -> str:
    """Validates that value could be the name of a column in a data warehouse.

    Any name a warehouse allows is accepted, including dots, spaces, dashes and mixed case: a name is always looked up
    in the reflected table before it is used, and quoted wherever it appears in SQL. Names with control characters
    are refused; inspection does not offer the Postgres and Redshift columns that have them.
    """
    if not isinstance(value, str):
        raise ValueError(f"{info.field_name} must be a string")  # noqa: TRY004
    if has_control_characters(value):
        raise ValueError(f"{info.field_name} must not contain control characters")
    return value


FieldName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_LENGTH_OF_FIELD_NAME),
    BeforeValidator(validate_column_name),
    Field(examples=["field_name"]),
]
