from collections import Counter
from typing import List

from pydantic import BaseModel, ValidationError, field_validator, model_validator

from xngin.sheets.gsheets import read_sheet
from xngin.apiserver.api_types import DataType


class InvalidSheetDetails(BaseModel):
    """Describes a problem with the configuration input."""

    row_number: int | None = None
    column: str | None = None
    msg: str

    @classmethod
    def from_pydantic_error(cls, row: int | None, pve: ValidationError):
        """Converts a pydantic ValidationError into an InvalidSheetDetails."""
        pve = pve.errors()[0]
        vals = {}
        if row is not None:
            vals["row_number"] = row
        if loc := pve.get("loc"):
            vals["column"] = loc[0]
        if ctx := pve.get("ctx"):
            vals["msg"] = str(ctx.get("error"))
        else:
            vals["msg"] = str(pve.get("msg"))
        return InvalidSheetDetails(**vals)


class InvalidSheetException(Exception):
    """Raised when a spreadsheet fails to parse into a valid configuration."""

    def __init__(self, err: List[InvalidSheetDetails]):
        super().__init__()
        self.errors = err

    def __str__(self):
        return "\n".join((err.model_dump_json() for err in self.errors))


class RowConfig(BaseModel):
    table: str
    column_name: str
    data_type: DataType
    column_group: str
    description: str
    is_strata: bool
    is_filter: bool
    is_metric: bool
    extra: dict[str, str] = dict()

    model_config = {
        "strict": True,
        "extra": "forbid",
    }

    @field_validator("description", "column_group", mode="before")
    @classmethod
    def to_string_loose(cls, value) -> str:
        if not isinstance(value, str):
            return str(value)
        return value

    @field_validator("data_type", mode="before")
    @classmethod
    def to_data_type(cls, value) -> DataType:
        return DataType(value.lower())

    @field_validator("is_strata", "is_filter", "is_metric", mode="before")
    @classmethod
    def to_boolean(cls, value):
        truthy = {"true", "t", "yes", "y", "1"}
        falsy = {"false", "f", "no", "n", "0", ""}
        normalized = str(value).lower().strip()
        if normalized in truthy:
            return True
        if normalized in falsy:
            return False
        raise ValueError(f"Value '{value}' cannot be converted to a boolean.")


class SheetConfig(BaseModel):
    rows: List[RowConfig]

    model_config = {
        "strict": True,
        "extra": "forbid",
    }

    @model_validator(mode="after")
    def check_unique_columns(self) -> "SheetConfig":
        counted = Counter([".".join((row.table, row.column_name)) for row in self.rows])
        duplicates = [item for item, count in counted.items() if count > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate 'column_name' values found: {", ".join(duplicates)}."
            )
        return self

    @model_validator(mode="after")
    def check_non_empty_rows(self) -> "SheetConfig":
        if len(self.rows) == 0:
            raise ValueError("SheetConfig must contain at least one RowConfig.")
        return self


def fetch_and_parse_sheet(url, worksheet):
    """Fetches a Google Spreadsheet and parses it into a SheetConfig.

    :raise InvalidSheetException if there are any problems with the sheet.
    """
    sheet = read_sheet(url, worksheet).get_all_records()
    num_rows = len(sheet)
    if num_rows == 0:
        raise InvalidSheetException([
            InvalidSheetDetails(
                row_number=None,
                column=None,
                msg="The sheet does not have any data rows.",
            )
        ])
    column_names = sheet[0].keys()
    required_column_names = RowConfig.model_fields.keys() - {"extra"}
    extra_column_names = column_names - required_column_names
    errors = []
    collector = []
    for row_index, values in enumerate(sheet):
        values["extra"] = {col: str(values.pop(col)) for col in extra_column_names}
        try:
            collector.append(RowConfig(**values))
        except ValidationError as pve:
            errors.append(
                InvalidSheetDetails.from_pydantic_error(row=row_index + 1, pve=pve)
            )
    try:
        parsed = SheetConfig(rows=collector)
    except ValidationError as pve:
        errors.append(InvalidSheetDetails.from_pydantic_error(row=None, pve=pve))
    if errors:
        raise InvalidSheetException(errors)
    return parsed
