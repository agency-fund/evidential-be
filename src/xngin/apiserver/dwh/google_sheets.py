"""Google Sheets datasource: tabs are read into an in-memory SQLite table per session for the warehouse queries."""

import base64
import json
import math
import re
import secrets
import sqlite3
import statistics
from dataclasses import dataclass
from typing import Self, cast

import gspread
import requests
import sqlalchemy as sa
from google.auth.exceptions import GoogleAuthError
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound
from gspread.utils import GridRangeType, ValueRenderOption, rowcol_to_a1
from sqlalchemy import event

from xngin.apiserver import flags
from xngin.apiserver.common_field_types import has_control_characters
from xngin.apiserver.exceptions_common import DwhConnectionError, LateValidationError
from xngin.apiserver.limits import MAX_LENGTH_OF_FIELD_NAME
from xngin.apiserver.settings import GoogleSheetsDsn

MAX_SHEET_ROWS = 2_000  # Designed to discourage customers from using GSheets for large experiments
MAX_SHEET_COLUMNS = 50
# Reading one row and column past the limits shows whether a tab exceeds them, while bounding what Google sends.
READ_RANGE = f"A1:{rowcol_to_a1(MAX_SHEET_ROWS + 2, MAX_SHEET_COLUMNS + 1)}"


def participant_id(value: object) -> str:
    # IDs are stored as strings in Evidential.
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def parse_cell(value: object) -> object:
    """Infer simple numeric/boolean cells while retaining text and leading zeroes."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if re.fullmatch(r"[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", stripped):
        if "." not in stripped and "e" not in stripped.lower():
            if len(stripped.lstrip("+-")) > 19:
                return value
            number = int(stripped)
            # Column will be stored as INTEGER if it fits in 64-bit signed range.
            return number if -(2**63) <= number < 2**63 else value
        number_float = float(stripped)
        return number_float if math.isfinite(number_float) else value
    return value


class PopulationStddev:
    """SQLite lacks stddev_pop, so use statistics.pstdev instead."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def step(self, value: float | None) -> None:
        if value is not None:
            self.values.append(value)

    def finalize(self) -> float | None:
        return statistics.pstdev(self.values) if self.values else None


@dataclass
class SheetData:
    headers: list[str]
    # Retain source row gaps in experiment outcome tabs.
    rows: list[tuple[int, list]]

    @classmethod
    def from_values(cls, values: list[list]) -> Self:
        if not values or not values[0]:
            raise LateValidationError("The sheet needs a header row in row 1.")
        headers = values[0]
        if len(headers) > MAX_SHEET_COLUMNS or len(values) > MAX_SHEET_ROWS + 1:
            raise LateValidationError(
                f"Google Sheets supports up to {MAX_SHEET_ROWS:,} rows and {MAX_SHEET_COLUMNS} columns."
            )
        if any(
            not isinstance(h, str) or not h or len(h) > MAX_LENGTH_OF_FIELD_NAME or has_control_characters(h)
            for h in headers
        ):
            raise LateValidationError(
                f"Every sheet column needs a header in row 1 of at most {MAX_LENGTH_OF_FIELD_NAME} characters, "
                "without line breaks or tabs."
            )
        if len({h.lower() for h in headers}) != len(headers):
            raise LateValidationError("Each sheet column needs a unique header (ignoring case).")
        rows = []
        for row_number, row in enumerate(values[1:], start=2):
            if len(row) > len(headers):
                raise LateValidationError("Every populated sheet column needs a header in row 1.")
            if any(cell not in {"", None} for cell in row):
                rows.append((row_number, [None if v == "" else v for v in row] + [None] * (len(headers) - len(row))))
        return cls(headers=headers, rows=rows)

    def _column_values(self, index: int) -> tuple[sa.types.TypeEngine, list]:
        header = self.headers[index]
        original = [row[index] for _, row in self.rows]
        # Keep identifier and assignment columns textual, including entirely empty ones.
        if (
            header.lower() in {"id", "hash"}
            or header.lower().endswith(("_id", "_hash"))
            or header.startswith("evidential_")
        ):
            return sa.String(), original
        parsed = [parse_cell(value) for value in original]
        values = [value for value in parsed if value is not None]
        if values and all(isinstance(value, bool) for value in values):
            return sa.Boolean(), parsed
        if values and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            return sa.BigInteger(), parsed
        if all(isinstance(value, (float, int)) and not isinstance(value, bool) for value in values):
            # Empty outcome columns are numeric so they can be selected as metrics during setup.
            return sa.Double(), parsed
        # Keep mixed-type text columns intact rather than partially parsing their cells.
        return sa.String(), original

    def to_table(self, table_name: str, engine: sa.Engine) -> sa.Table:
        columns = []
        typed_rows = [list(row) for _, row in self.rows]
        for i, header in enumerate(self.headers):
            column_type, values = self._column_values(i)
            for row, value in zip(typed_rows, values, strict=True):
                row[i] = value
            columns.append(sa.Column(header, column_type, nullable=True))
        table = sa.Table(table_name, sa.MetaData(), *columns)
        table.create(engine)
        records = [
            {
                col.name: participant_id(value) if isinstance(col.type, sa.String) and value is not None else value
                for col, value in zip(columns, row, strict=True)
            }
            for row in typed_rows
        ]
        if records:
            with engine.begin() as connection:
                connection.execute(table.insert(), records)
        return table

    def assignments_values(
        self,
        unique_id_field: str,
        experiment_id: str,
        assignments: dict[str, str],
        field_names: set[str] | None = None,
    ) -> list[list]:
        """Copy selected source cells and assignments using literal, typed Sheets values."""
        if unique_id_field not in self.headers:
            raise LateValidationError("The participant ID column is missing from the sheet.")
        id_col = self.headers.index(unique_id_field)
        _, id_values = self._column_values(id_col)
        ids = set()
        for value in id_values:
            if value is None:
                raise LateValidationError("Every populated sheet row needs a participant ID before saving assignments.")
            pid = participant_id(value)
            if pid in ids:
                raise LateValidationError("Participant IDs in the sheet must be unique before saving assignments.")
            ids.add(pid)
        if assignments.keys() - ids:
            raise LateValidationError("Assigned participants are missing from the sheet. Restore their IDs first.")

        header = f"evidential_{experiment_id.replace('-', '_')}_arm"
        if field_names is not None and (missing := field_names - set(self.headers)):
            raise LateValidationError(f"Experiment columns are missing from the sheet: {', '.join(sorted(missing))}.")
        headers = [
            name for name in self.headers if field_names is None or name in field_names or name == unique_id_field
        ]
        arm_col = next((index for index, name in enumerate(headers) if name.lower() == header.lower()), None)
        if arm_col is None:
            arm_col = len(headers)
            headers.append(header)
        if len(headers) > MAX_SHEET_COLUMNS:
            raise LateValidationError(
                f"Leave space for an assignment column within the {MAX_SHEET_COLUMNS}-column limit."
            )
        source_indices = [self.headers.index(name) if name in self.headers else None for name in headers]
        output_id_col = headers.index(unique_id_field)
        values: list[list] = [headers]
        next_row = 2
        for (row_number, original), value in zip(self.rows, id_values, strict=True):
            while next_row < row_number:
                values.append([None] * len(headers))
                next_row += 1
            row = [original[index] if index is not None else None for index in source_indices]
            row[output_id_col] = participant_id(row[output_id_col])
            row[arm_col] = assignments.get(participant_id(value))
            values.append(row)
            next_row = row_number + 1
        for output_col, source_col in enumerate(source_indices):
            if source_col is None or output_col in {output_id_col, arm_col}:
                continue
            column_type, column_values = self._column_values(source_col)
            if isinstance(column_type, sa.String):
                continue
            for (row_number, _), value in zip(self.rows, column_values, strict=True):
                values[row_number - 1][output_col] = value
        return values


class SheetsAccessError(DwhConnectionError):
    """A sharing instruction that can be shown directly to the user."""

    def __init__(self, service_account_email: str) -> None:
        self.message = f"Please share this spreadsheet with {service_account_email} as Editor, then try again."
        super().__init__(ValueError(self.message))

    def __str__(self) -> str:
        return self.message


def _decode_credentials(encoded: str) -> dict:
    credentials = json.loads(base64.b64decode(encoded, validate=True))
    if not isinstance(credentials, dict):
        raise TypeError("Google Sheets service-account credentials must be a JSON object.")
    return credentials


class GoogleSheetsClient:
    """A spreadsheet connection whose tables are its tabs, named by tab ID (the gid in a tab's URL).

    Each tab is read once per session. The URL's gid selects the default tab, or the first tab when it has none.
    """

    def __init__(self, dsn: GoogleSheetsDsn) -> None:
        self.spreadsheet_id = dsn.spreadsheet_id
        self._default_gid = dsn.default_gid
        self._data: dict[int, SheetData] = {}
        if not flags.GSHEETS_DEMO_CREDS_BASE64:
            raise DwhConnectionError(
                ValueError(
                    "Google Sheets service-account credentials are not configured. Set GSHEETS_DEMO_CREDS_BASE64."
                )
            )
        try:
            self.client = gspread.service_account_from_dict(
                info=_decode_credentials(flags.GSHEETS_DEMO_CREDS_BASE64),
                # Editor scope to create outcome tab.
                scopes=["https://www.googleapis.com/auth/spreadsheets"],
            )
        except (TypeError, ValueError, GoogleAuthError) as exc:
            raise DwhConnectionError(
                ValueError(
                    "Could not load the Google Sheets service-account credentials. Check GSHEETS_DEMO_CREDS_BASE64."
                )
            ) from exc
        self.client.set_timeout(10)
        try:
            self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
        except (
            PermissionError,
            SpreadsheetNotFound,
            APIError,
            GoogleAuthError,
            requests.RequestException,
            ValueError,
        ) as exc:
            self.close()
            raise self._connection_error(exc) from exc
        except Exception:
            self.close()
            raise

    def default_tab_id(self) -> str:
        return str(self._worksheet(None).id)

    def _worksheet(self, tab_id: str | None) -> gspread.Worksheet:
        if tab_id is not None and not tab_id.isdecimal():
            raise LateValidationError("Google Sheets tables are tab IDs: the number after gid= in the tab's URL.")
        try:
            if tab_id is not None:
                worksheet = self.spreadsheet.get_worksheet_by_id(int(tab_id))
            elif self._default_gid is not None:
                worksheet = self.spreadsheet.get_worksheet_by_id(self._default_gid)
            else:
                worksheet = self.spreadsheet.get_worksheet(0)
        except WorksheetNotFound as exc:
            raise DwhConnectionError(
                ValueError("Google Sheets could not find the selected tab. Restore it in the spreadsheet.")
            ) from exc
        except (
            PermissionError,
            SpreadsheetNotFound,
            APIError,
            GoogleAuthError,
            requests.RequestException,
            ValueError,
        ) as exc:
            raise self._connection_error(exc) from exc

        if worksheet is None:
            raise DwhConnectionError(ValueError("The Google Sheets spreadsheet has no tabs."))
        return worksheet

    def _connection_error(self, exc: Exception) -> DwhConnectionError:
        if isinstance(exc, (PermissionError, SpreadsheetNotFound)) or (
            isinstance(exc, APIError) and exc.response.status_code in {401, 403, 404}
        ):
            credentials = cast(Credentials, self.client.http_client.auth)
            return SheetsAccessError(credentials.service_account_email)
        if isinstance(exc, APIError):
            message = (
                "Google Sheets rate limit reached. Try again in a minute."
                if exc.response.status_code == 429
                else f"Google Sheets request failed (HTTP {exc.response.status_code})."
            )
        elif isinstance(exc, GoogleAuthError):
            message = "Google Sheets could not authenticate the configured service account. Check its credentials."
        else:
            message = "Google Sheets did not return a readable response. Please try again."
        return DwhConnectionError(ValueError(message))

    def close(self) -> None:
        self.client.http_client.session.close()

    def read(self, tab_id: str) -> SheetData:
        return self._read(self._worksheet(tab_id))

    def load_table(self, tab_id: str, engine: sa.Engine) -> sa.Table:
        """Loads a tab into engine as a table named by its tab ID, with a readable name in its info."""
        worksheet = self._worksheet(tab_id)
        table = self._read(worksheet).to_table(tab_id, engine)
        table.info["display_name"] = f"{self.spreadsheet.title} - {worksheet.title}"
        return table

    def _read(self, worksheet: gspread.Worksheet) -> SheetData:
        if worksheet.id not in self._data:
            try:
                values = worksheet.get(
                    READ_RANGE,
                    value_render_option=ValueRenderOption.formatted,
                    return_type=GridRangeType.ListOfLists,
                )
            except (APIError, GoogleAuthError, requests.RequestException, ValueError) as exc:
                raise self._connection_error(exc) from exc
            self._data[worksheet.id] = SheetData.from_values(values)
        return self._data[worksheet.id]

    @staticmethod
    def create_engine() -> sa.Engine:
        # Each DWH block gets an independent snapshot. Its queries all run on the same helper thread.
        engine = sa.create_engine("sqlite:///:memory:")

        @event.listens_for(engine, "connect")
        def register_aggregates(connection: sqlite3.Connection, _record: object) -> None:
            # typeshed's aggregate protocol only allows ints; stddev_pop takes and returns floats.
            connection.create_aggregate("stddev_pop", 1, PopulationStddev)  # type: ignore[arg-type]

        return engine

    def create_experiment_tab(
        self,
        source_tab_id: str,
        unique_id_field: str,
        experiment_id: str,
        assignments: dict[str, str],
        field_names: set[str] | None = None,
    ) -> str:
        """Creates the experiment's outcomes tab and returns its tab ID.

        1. If a tab already tagged with this experiment ID exists (an earlier commit wrote it but the
           database save was rolled back), return its ID without touching the source tab.
        2. Read the source tab and build the rows: the participant ID, the experiment's field columns
           (all columns when field_names is None) and an "evidential_<experiment_id>_arm" column.
        3. Add, fill and tag the new tab in one batchUpdate, so a failure leaves nothing behind.

        Users enter outcomes in the new tab; the source tab is never modified.
        """
        spreadsheet = self.spreadsheet
        try:
            sheets = spreadsheet.fetch_sheet_metadata(params={"fields": "sheets(properties,developerMetadata)"}).get(
                "sheets", []
            )
            for sheet in sheets:
                sheet_id = sheet["properties"]["sheetId"]
                if any(
                    metadata.get("metadataKey") == "evidential_experiment_id"
                    and metadata.get("metadataValue") == experiment_id
                    and metadata.get("location", {}).get("sheetId") == sheet_id
                    for metadata in sheet.get("developerMetadata", [])
                ):
                    return str(sheet_id)

            values = self.read(source_tab_id).assignments_values(
                unique_id_field, experiment_id, assignments, field_names
            )
            existing_ids = {sheet["properties"]["sheetId"] for sheet in sheets}
            sheet_id = secrets.randbelow(2**31)
            while sheet_id in existing_ids:
                sheet_id = secrets.randbelow(2**31)
            title = f"Evidential {experiment_id}"
            existing_titles = {sheet["properties"]["title"] for sheet in sheets}
            if title in existing_titles:
                suffix = 2
                while f"{title} ({suffix})" in existing_titles:
                    suffix += 1
                title = f"{title} ({suffix})"
            rows = []
            for row in values:
                cells = []
                for value in row:
                    if value is None:
                        cell: dict[str, dict[str, str | int | float | bool]] = {}
                    elif isinstance(value, bool):
                        cell = {"userEnteredValue": {"boolValue": value}}
                    elif isinstance(value, (int, float)):
                        cell = {"userEnteredValue": {"numberValue": value}}
                    else:
                        cell = {"userEnteredValue": {"stringValue": value}}
                    cells.append(cell)
                rows.append({"values": cells})
            spreadsheet.batch_update({
                "requests": [
                    {
                        "addSheet": {
                            "properties": {
                                "sheetId": sheet_id,
                                "title": title,
                                "gridProperties": {"rowCount": len(values), "columnCount": len(values[0])},
                            }
                        }
                    },
                    {
                        "updateCells": {
                            "start": {"sheetId": sheet_id, "rowIndex": 0, "columnIndex": 0},
                            "rows": rows,
                            "fields": "userEnteredValue",
                        }
                    },
                    {
                        "createDeveloperMetadata": {
                            "developerMetadata": {
                                "metadataKey": "evidential_experiment_id",
                                "metadataValue": experiment_id,
                                "location": {"sheetId": sheet_id},
                                "visibility": "DOCUMENT",
                            }
                        }
                    },
                ]
            })
        except (
            PermissionError,
            SpreadsheetNotFound,
            APIError,
            GoogleAuthError,
            requests.RequestException,
            ValueError,
        ) as exc:
            raise self._connection_error(exc) from exc
        else:
            return str(sheet_id)
