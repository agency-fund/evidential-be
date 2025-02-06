import csv

import sqlalchemy
from pydantic import BaseModel, ValidationError

from xngin.apiserver.settings import SheetRef
from xngin.sheets.gsheets import read_sheet_from_gsheet
from xngin.apiserver.api_types import DataType
from xngin.sheets.sheet_types import FieldDescriptor, ConfigWorksheet

GOOGLE_SHEET_PREFIX = "https://docs.google.com/spreadsheets/"


class InvalidSheetDetails(BaseModel):
    """Describes a problem with the configuration input."""

    row_number: int | None = None
    field: str | None = None
    msg: str

    @classmethod
    def from_pydantic_error(cls, row: int | None, pve: ValidationError):
        """Converts a pydantic ValidationError into an InvalidSheetDetails."""
        details = pve.errors()[0]
        vals: dict = {}
        if row is not None:
            vals["row_number"] = row
        if loc := details.get("loc"):
            vals["column"] = loc[0]
        if ctx := details.get("ctx"):
            vals["msg"] = str(ctx.get("error"))
        else:
            vals["msg"] = str(details.get("msg"))
        return InvalidSheetDetails(**vals)


class InvalidSheetError(Exception):
    """Raised when a spreadsheet fails to parse into a valid configuration."""

    def __init__(self, err: list[InvalidSheetDetails]):
        super().__init__()
        self.errors = err

    def __str__(self):
        return "\n".join(err.model_dump_json() for err in self.errors)


def parse_csv(filename: str) -> list[dict[str, int | float | str]]:
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            parsed_row: dict = {}
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
    """Fetches a Google Spreadsheet and parses it into a ConfigWorksheet.

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
        raise InvalidSheetError([
            InvalidSheetDetails(
                row_number=None,
                field=None,
                msg="The sheet does not have any data rows.",
            )
        ])
    field_names = sheet[0].keys()
    required_field_names = FieldDescriptor.model_fields.keys() - {"extra"}
    extra_field_names = field_names - required_field_names
    errors = []
    collector = []
    for row_index, values in enumerate(sheet):
        values["extra"] = {col: str(values.pop(col)) for col in extra_field_names}
        try:
            collector.append(FieldDescriptor(**values))
        except ValidationError as pve:
            errors.append(
                InvalidSheetDetails.from_pydantic_error(row=row_index + 1, pve=pve)
            )
    try:
        parsed = ConfigWorksheet(table_name=ref.worksheet, fields=collector)
        # Parsing succeeded, but also raise if there were /any/ errors from above.
        if errors:
            raise InvalidSheetError(errors)
    except ValidationError as pve:
        errors.append(InvalidSheetDetails.from_pydantic_error(row=None, pve=pve))
    else:
        return parsed
    raise InvalidSheetError(errors)


def create_configworksheet_from_table(
    table: sqlalchemy.Table, unique_id_col: str | None = None
):
    """Attempts to get name and type info from the database Table itself (formerly done via gsheets).

    If unique_id_col is explicitly set to None, we will look for a primary key else assume "id".
    (This mode should only be used if bootstrapping a sheet config from a table's schema.)
    """

    collected = []
    if unique_id_col is None:
        unique_id_col = next(
            (c.name for c in table.columns.values() if c.primary_key), "id"
        )
    for column in table.columns.values():
        type_hint = column.type
        collected.append(
            FieldDescriptor(
                field_name=column.name,
                data_type=DataType.match(type_hint),
                description="",
                is_unique_id=column.name == unique_id_col,
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
            r.field_name,
        ),
    )
    return ConfigWorksheet(table_name=table.name, fields=rows)
