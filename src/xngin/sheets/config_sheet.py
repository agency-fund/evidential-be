import csv
from collections import Counter
from typing import List, Union

import sqlalchemy
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from xngin.apiserver.settings import SheetRef
from xngin.sheets.gsheets import read_sheet_from_gsheet
from xngin.apiserver.api_types import DataType

GOOGLE_SHEET_PREFIX = "https://docs.google.com/spreadsheets/"


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
    table: str  # TODO: remove .table from RowConfig.
    column_name: str
    data_type: DataType
    column_group: str
    description: str
    is_unique_id: bool
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

    @field_validator(
        "is_unique_id", "is_strata", "is_filter", "is_metric", mode="before"
    )
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
    def check_one_unique_id(self) -> "SheetConfig":
        uniques = [r.column_name for r in self.rows if r.is_unique_id]
        if len(uniques) == 0:
            raise ValueError("There are no columns marked as unique ID.")
        if len(uniques) > 1:
            raise ValueError(
                f"There are {len(uniques)} columns marked as the unique ID, but there should "
                f"only be one: {", ".join(sorted(uniques))}"
            )
        return self

    @model_validator(mode="after")
    def check_unique_columns(self) -> "SheetConfig":
        # TODO: remove .table from RowConfig.
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


def parse_csv(filename: str) -> List[dict[str, Union[int, float, str]]]:
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            parsed_row = {}
            for key, value in row.items():
                try:
                    # Try to convert to int first
                    parsed_row[key] = int(value)
                except ValueError:
                    try:
                        # If not int, try float
                        parsed_row[key] = float(value)
                    except ValueError:
                        # If not float, keep as string
                        parsed_row[key] = value

            yield parsed_row


def read_sheet_from_file(path):
    """Reads a spreadsheet from a CSV file."""
    return list(parse_csv(path))


def fetch_and_parse_sheet(ref: SheetRef):
    """Fetches a Google Spreadsheet and parses it into a SheetConfig.

    :raise InvalidSheetException if there are any problems with the sheet.
    """
    if ref.url.startswith(GOOGLE_SHEET_PREFIX):
        sheet = read_sheet_from_gsheet(ref.url, ref.worksheet)
    elif ref.url.startswith("file://"):
        sheet = read_sheet_from_file(ref.url[len("file://") :])
    else:
        raise ValueError("Path to configuration spreadsheet is not usable.")
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


def create_sheetconfig_from_table(table: sqlalchemy.Table):
    collected = []
    # find the primary key
    pk_col = next((c.name for c in table.columns.values() if c.primary_key), None)
    # if the database doesn't have one, assume the existence of an "id" column.
    if not pk_col:
        pk_col = "id"
    for column in table.columns.values():
        type_hint = column.type
        collected.append(
            RowConfig(
                table=table.name,
                column_name=column.name,
                data_type=DataType.match(type_hint),
                column_group="",
                description="",
                is_unique_id=column.name == pk_col,
                is_strata=False,
                is_filter=False,
                is_metric=False,
            )
        )
    # Sort order is: unique ID first, then string fields, then the rest by name.
    rows = sorted(
        collected,
        key=lambda r: (
            not r.is_unique_id,
            r.data_type != DataType.CHARACTER_VARYING,
            r.column_name,
        ),
    )
    return SheetConfig(rows=rows)
