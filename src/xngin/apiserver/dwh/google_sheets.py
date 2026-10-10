"""Small Google Sheets snapshots adapted to the existing warehouse query path."""

import csv
import io
import math
import re
import time
from dataclasses import dataclass
from typing import Self

import requests
import sqlalchemy as sa
from sqlalchemy import event

from xngin.apiserver.common_field_types import VALID_SQL_COLUMN_REGEX
from xngin.apiserver.exceptions_common import DwhConnectionError, LateValidationError
from xngin.apiserver.sheets_url import parse_spreadsheet_url

MAX_DEMO_ROWS = 5_000
MAX_DEMO_COLUMNS = 100
MAX_DEMO_BYTES = 8 * 1024 * 1024
PUBLIC_SHEET_TABLE = "linked_sheet"


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
    # Retain row gaps when generating a CSV for import into the same tab.
    rows: list[tuple[int, list]]

    @classmethod
    def from_values(cls, values: list[list]) -> Self:
        if not values or not values[0]:
            raise LateValidationError("The sheet needs a header row in row 1.")
        headers = values[0]
        if len(headers) > MAX_DEMO_COLUMNS or len(values) > MAX_DEMO_ROWS + 1:
            raise LateValidationError("Demo sheets support up to 5,000 rows and 100 columns.")
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

    def assignments_csv(
        self,
        unique_id_field: str,
        experiment_id: str,
        assignments: dict[str, str],
        field_names: set[str] | None = None,
    ) -> str:
        """Preserve selected source columns and append arm names for an experiment tab."""
        if unique_id_field not in self.headers:
            raise LateValidationError("The participant ID column is missing from the sheet.")
        id_col = self.headers.index(unique_id_field)
        _, id_values = self._column_values(id_col)
        ids = set()
        for value in id_values:
            if value is None:
                raise LateValidationError(
                    "Every populated sheet row needs a participant ID before downloading the CSV."
                )
            pid = participant_id(value)
            if pid in ids:
                raise LateValidationError("Participant IDs in the sheet must be unique before downloading the CSV.")
            ids.add(pid)
        if assignments.keys() - ids:
            raise LateValidationError("Assigned participants are missing from the sheet. Restore their IDs first.")

        header = f"evidential_{experiment_id.replace('-', '_')}_arm"
        if field_names is not None and (missing := field_names - set(self.headers)):
            raise LateValidationError(f"Experiment columns are missing from the sheet: {', '.join(sorted(missing))}.")
        headers = [
            name for name in self.headers if field_names is None or name in field_names or name == unique_id_field
        ]
        if header not in headers:
            headers.append(header)
        if len(headers) > MAX_DEMO_COLUMNS:
            raise LateValidationError("Leave space for an assignment column within the demo's 100-column limit.")
        arm_col = headers.index(header)
        source_indices = [self.headers.index(name) if name in self.headers else None for name in headers]
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(headers)
        next_row = 2
        for (row_number, original), value in zip(self.rows, id_values, strict=True):
            while next_row < row_number:
                writer.writerow([None] * len(headers))
                next_row += 1
            row = [original[index] if index is not None else None for index in source_indices]
            row[arm_col] = assignments.get(participant_id(value))
            writer.writerow(row)
            next_row = row_number + 1
        return output.getvalue()


class GoogleSheetsClient:
    def __init__(self, spreadsheet_url: str):
        spreadsheet_id, gid = parse_spreadsheet_url(spreadsheet_url)
        self.http = requests.Session()
        # Public downloads must never pick up server credentials from .netrc.
        self.http.trust_env = False
        self.sheets = [PUBLIC_SHEET_TABLE]
        try:
            params = {"format": "csv", "_evidential": str(time.time_ns())}
            if gid is not None:
                params["gid"] = str(gid)
            self.data = self._download(f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export", params)
        except Exception:
            self.close()
            raise

    def close(self):
        self.http.close()

    def _download(self, url: str, params: dict[str, str]) -> SheetData:
        try:
            with self.http.get(
                url, params=params, timeout=10, stream=True, headers={"Cache-Control": "no-cache"}
            ) as response:
                if not response.ok:
                    if response.status_code == 400 and "gid" in params:
                        message = (
                            "Google Sheets could not open the linked tab. Importing a CSV can replace a tab and "
                            "change its ID. Open the intended tab, copy its current URL, and use "
                            "'Connect Experiment tab' to update the datasource."
                        )
                    elif response.status_code in {401, 403, 404}:
                        message = (
                            "Cannot read this Google Sheets tab. Set sharing to 'Anyone with the link: Viewer', "
                            "allow viewers to download, and copy the tab's URL again."
                        )
                    elif response.status_code == 429:
                        message = "Google Sheets rate limit reached. Stop the live demo and try again in a minute."
                    else:
                        message = f"Google Sheets request failed (HTTP {response.status_code})."
                    raise DwhConnectionError(ValueError(message))
                payload = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    payload.extend(chunk)
                    if len(payload) > MAX_DEMO_BYTES:
                        raise LateValidationError("Demo sheet CSV downloads must be smaller than 8 MB.")
                content = payload.decode("utf-8-sig")
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise DwhConnectionError(
                        ValueError(
                            "Google Sheets returned a sign-in page. Set sharing to 'Anyone with the link: Viewer'."
                        )
                    )
                return SheetData.from_values(list(csv.reader(io.StringIO(content), strict=True)))
        except requests.RequestException as exc:
            raise DwhConnectionError(ValueError("Google Sheets did not respond. Please try again.")) from exc
        except (UnicodeError, csv.Error) as exc:
            raise DwhConnectionError(ValueError("Google Sheets returned an unreadable CSV download.")) from exc

    def read(self, table_name: str) -> SheetData:
        if table_name != PUBLIC_SHEET_TABLE:
            raise LateValidationError("Select linked_sheet to use the tab in the spreadsheet URL.")
        return self.data

    @staticmethod
    def create_engine() -> sa.Engine:
        # Each DWH block gets an independent snapshot. Its queries all run on the same helper thread.
        engine = sa.create_engine("sqlite:///:memory:")

        @event.listens_for(engine, "connect")
        def register_aggregates(connection, _record):
            connection.create_aggregate("stddev_pop", 1, PopulationStddev)

        return engine

    def export_assignments(
        self,
        table_name: str,
        unique_id_field: str,
        experiment_id: str,
        assignments: dict[str, str],
        field_names: set[str] | None = None,
    ) -> str:
        return self.read(table_name).assignments_csv(unique_id_field, experiment_id, assignments, field_names)
