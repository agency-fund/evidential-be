# mypy: disable-error-code="misc"

from fastapi.responses import StreamingResponse
from psycopg import sql
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_enums import ExperimentsType
from xngin.apiserver.sqla import tables

CSV_STREAM_CHUNK_SIZE_BYTES = 256 * 1024


class CsvStreamingResponse(StreamingResponse):
    media_type = "text/csv"


def _get_assignment_csv_strata_names_from_experiment(experiment: tables.Experiment) -> list[str]:
    if experiment.design_spec_fields is None:
        return []
    stored_strata = experiment.design_spec_fields.get("strata") or []
    return sorted(stratum["field_name"] for stratum in stored_strata)


def _build_experiment_assignments_copy_query(experiment_id: str, strata_names: list[str]):
    if strata_names:
        projected_columns = [sql.Identifier("strata", strata_name) for strata_name in strata_names]
        joined_projected_columns = sql.SQL(", ").join(projected_columns)
        extra_columns = t", {joined_projected_columns:q}"

        # Use MAX(... FILTER ...) to pivot validated single-value strata rows into columns.
        lateral_columns = [
            t"MAX(elem.strata_value) FILTER (WHERE elem.field_name = {strata_name}) AS {strata_name:i}"
            for strata_name in strata_names
        ]
        joined_lateral_columns = sql.SQL(", ").join(lateral_columns)
        lateral_join = t"""
            LEFT JOIN LATERAL (
                SELECT {joined_lateral_columns:q}
                FROM jsonb_to_recordset(aa.strata) AS elem(field_name text, strata_value text)
            ) AS strata ON TRUE
        """
    else:
        extra_columns = sql.SQL("")
        lateral_join = sql.SQL("")
    return t"""
        COPY (
            SELECT
                aa.participant_id AS participant_id,
                aa.arm_id AS arm_id,
                a.name AS arm_name,
                to_char(aa.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at
                {extra_columns:q}
            FROM arm_assignments AS aa
            JOIN arms AS a
                ON a.id = aa.arm_id
                AND a.experiment_id = aa.experiment_id
            {lateral_join:q}
            WHERE aa.experiment_id = {experiment_id}
        ) TO STDOUT WITH (FORMAT CSV, HEADER TRUE)
    """


def _validate_experiment_assignments_csv_export(experiment: tables.Experiment) -> None:
    if experiment.experiment_type not in {
        ExperimentsType.FREQ_ONLINE.value,
        ExperimentsType.FREQ_PREASSIGNED.value,
    }:
        raise LateValidationError(f"CSV export is not supported for experiment type: {experiment.experiment_type}")


async def get_experiment_assignments_as_csv_impl(
    xngin_session: AsyncSession,
    experiment: tables.Experiment,
) -> CsvStreamingResponse:
    _validate_experiment_assignments_csv_export(experiment)
    strata_names = _get_assignment_csv_strata_names_from_experiment(experiment)
    copy_query = _build_experiment_assignments_copy_query(experiment.id, strata_names)

    async def csv_generator():
        async_conn = await xngin_session.connection()
        raw_conn = await async_conn.get_raw_connection()
        driver_conn = raw_conn.driver_connection
        if driver_conn is None:
            raise RuntimeError("Expected psycopg driver connection for CSV export.")
        buffer = bytearray()
        async with driver_conn.cursor() as cursor, cursor.copy(copy_query) as copy:
            async for chunk in copy:
                buffer.extend(chunk)
                if len(buffer) >= CSV_STREAM_CHUNK_SIZE_BYTES:
                    yield bytes(buffer)
                    buffer.clear()
        if buffer:
            yield bytes(buffer)

    filename = f"experiment_{experiment.id}_assignments.csv"
    return CsvStreamingResponse(
        csv_generator(),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
