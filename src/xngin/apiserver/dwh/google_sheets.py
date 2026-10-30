"""Small Google Sheets snapshots adapted to the existing warehouse query path."""

import math
import re
import secrets
from dataclasses import dataclass
from typing import Self, cast

import gspread
import requests
import sqlalchemy as sa
from google.auth.exceptions import GoogleAuthError
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound
from gspread.utils import GridRangeType, ValueRenderOption
from sqlalchemy import event

from xngin.apiserver import flags
from xngin.apiserver.common_field_types import VALID_SQL_COLUMN_REGEX
from xngin.apiserver.exceptions_common import DwhConnectionError, LateValidationError
from xngin.apiserver.limits import MAX_LENGTH_OF_NAME_VALUE
from xngin.apiserver.sheets_url import parse_spreadsheet_url

MAX_SHEET_ROWS = 5_000
MAX_SHEET_COLUMNS = 100
MAX_SHEET_BYTES = 8 * 1024 * 1024


def participant_id(value) -> str:
    # IDs are stored as strings in Evidential.
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def parse_csv_cell(value):
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
            return number if -(2**63) <= number < 2**63 else value
        number_float = float(stripped)
        return number_float if math.isfinite(number_float) else value
    return value


class PopulationStddev:
    """SQLite aggregate for the power check's stddev_pop, using Welford's algorithm."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def step(self, value):
        if value is not None:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (value - self.mean)

    def finalize(self):
        return math.sqrt(max(0, self.m2 / self.count)) if self.count else None


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
            raise LateValidationError("Google Sheets supports up to 5,000 rows and 100 columns.")
        if any(not isinstance(h, str) or not re.fullmatch(VALID_SQL_COLUMN_REGEX, h) for h in headers):
            raise LateValidationError(
                "Sheet headers must start with a letter or underscore and contain only letters, numbers, underscores."
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
        parsed = [parse_csv_cell(value) for value in original]
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
            raise LateValidationError("Leave space for an assignment column within the 100-column limit.")
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


class SheetsHTTPClient(gspread.HTTPClient):
    """Bound Google API responses before gspread decodes them."""

    def __init__(self, auth, session=None):
        super().__init__(auth, session)
        self.session.trust_env = False
        self.session.stream = True
        # Hooks run before Requests prepares redirects or google-auth retries an error response.
        self.session.hooks["response"].append(self._read_response)

    @staticmethod
    def _read_response(response: requests.Response, **_kwargs) -> requests.Response:
        with response:
            if response.is_redirect:
                raise DwhConnectionError(ValueError("Google Sheets returned an unexpected redirect."))
            payload = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if len(payload) + len(chunk) > MAX_SHEET_BYTES:
                    raise LateValidationError("Google Sheets API responses must be smaller than 8 MB.")
                payload.extend(chunk)
            # Cache the bounded body for gspread's response.json() and APIError handling.
            response._content = bytes(payload)
            return response


class SheetsAccessError(DwhConnectionError):
    """A sharing instruction that can be shown directly to the user."""

    def __init__(self, service_account_email: str, *, editor: bool = False):
        role = "Editor" if editor else "Viewer"
        self.message = f"Please share this spreadsheet with {service_account_email} as {role}, then try again."
        super().__init__(ValueError(self.message))

    def __str__(self) -> str:
        return self.message


class GoogleSheetsClient:
    """A spreadsheet connection whose tables are tabs, read once per tab per session.

    The URL's gid selects the default source tab, exposed under its saved table name.
    Other tabs can be selected by ID or title. Launched experiments use tab IDs so
    renaming or removing the source tab does not affect their outcomes or recovery.
    """

    def __init__(self, spreadsheet_url: str, table_name: str | None = None):
        spreadsheet_id, gid = parse_spreadsheet_url(spreadsheet_url)
        self.spreadsheet_id = spreadsheet_id
        self._default_gid = gid
        self._table_name = table_name
        self._data: dict[int, SheetData] = {}
        if not flags.GOOGLE_SHEETS_CREDENTIALS_FILE:
            raise DwhConnectionError(
                ValueError(
                    "Google Sheets service-account credentials are not configured. "
                    "Set XNGIN_GOOGLE_SHEETS_CREDENTIALS_FILE."
                )
            )
        try:
            self.client = gspread.service_account(
                filename=flags.GOOGLE_SHEETS_CREDENTIALS_FILE,
                scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
                http_client=SheetsHTTPClient,
            )
        except (OSError, ValueError, GoogleAuthError) as exc:
            raise DwhConnectionError(
                ValueError(
                    "Could not load the Google Sheets service-account credentials. "
                    "Check XNGIN_GOOGLE_SHEETS_CREDENTIALS_FILE."
                )
            ) from exc
        self.client.set_timeout(10)
        try:
            self.spreadsheet = self.client.open_by_key(spreadsheet_id)
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

    @property
    def table_name(self) -> str:
        """The source tab's captured display name, retained for existing design requests."""
        if self._table_name is None:
            worksheet = self._worksheet()
            name = f"{self.spreadsheet.title} - {worksheet.title}"
            if len(name) > MAX_LENGTH_OF_NAME_VALUE:
                raise LateValidationError(
                    f"Google Sheets table names must be at most {MAX_LENGTH_OF_NAME_VALUE} characters. "
                    "Shorten the spreadsheet or tab name before connecting."
                )
            self._table_name = name
        return self._table_name

    def _worksheet(self, table_name: str | None = None) -> gspread.Worksheet:
        try:
            if table_name is None or table_name == self._table_name:
                worksheet = (
                    self.spreadsheet.get_worksheet(0)
                    if self._default_gid is None
                    else self.spreadsheet.get_worksheet_by_id(self._default_gid)
                )
            elif table_name.isdecimal():
                worksheet = self.spreadsheet.get_worksheet_by_id(int(table_name))
            else:
                try:
                    worksheet = self.spreadsheet.worksheet(table_name)
                except WorksheetNotFound:
                    # Before a display alias is saved, accept the name returned by discovery.
                    title = table_name.removeprefix(f"{self.spreadsheet.title} - ")
                    if title == table_name:
                        raise
                    worksheet = self.spreadsheet.worksheet(title)
        except WorksheetNotFound as exc:
            raise DwhConnectionError(
                ValueError(
                    "Google Sheets could not open the selected tab. Restore the intended tab in the spreadsheet "
                    "or select its current table reference."
                )
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
            raise DwhConnectionError(ValueError("The Google Sheets spreadsheet has no source tab."))
        return worksheet

    def _connection_error(self, exc: Exception, *, editor: bool = False) -> DwhConnectionError:
        if isinstance(exc, (PermissionError, SpreadsheetNotFound)) or (
            isinstance(exc, APIError) and exc.response.status_code in {401, 403, 404}
        ):
            credentials = cast(Credentials, self.client.http_client.auth)
            return SheetsAccessError(credentials.service_account_email, editor=editor)
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

    def close(self):
        self.client.http_client.session.close()

    def read(self, table_name: str) -> SheetData:
        worksheet = self._worksheet(table_name)
        if worksheet.id not in self._data:
            try:
                values = worksheet.get(
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
        def register_aggregates(connection, _record):
            connection.create_aggregate("stddev_pop", 1, PopulationStddev)

        return engine

    def create_experiment_tab(
        self,
        table_name: str,
        unique_id_field: str,
        experiment_id: str,
        assignments: dict[str, str],
        field_names: set[str] | None = None,
    ) -> str:
        """Return the outcomes tab ID, recovering an interrupted launch before reading source cells."""
        credentials = cast(Credentials, self.client.http_client.auth)
        writer = None
        try:
            writer = gspread.authorize(
                credentials.with_scopes(["https://www.googleapis.com/auth/spreadsheets"]),
                http_client=SheetsHTTPClient,
            )
            writer.set_timeout(10)
            spreadsheet = writer.open_by_key(self.spreadsheet_id)
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

            values = self.read(table_name).assignments_values(unique_id_field, experiment_id, assignments, field_names)
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
            raise self._connection_error(exc, editor=True) from exc
        else:
            return str(sheet_id)
        finally:
            if writer is not None:
                writer.http_client.session.close()
