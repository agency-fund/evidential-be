# mypy: disable-error-code="misc"

from fastapi.responses import StreamingResponse
from psycopg import sql
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_enums import ExperimentsType
from xngin.apiserver.sqla import tables

CSV_STREAM_CHUNK_SIZE_BYTES = 256 * 1024


def _get_assignment_csv_strata_names_from_experiment(experiment: tables.Experiment) -> list[str]:
    if experiment.design_spec_fields is None:
        return []
    stored_strata = experiment.design_spec_fields.get("strata") or []
    return [stratum["field_name"] for stratum in stored_strata]


def _build_experiment_assignments_copy_query(experiment_id: str, strata_names: list[str]):
    strata_columns = []
    for strata_name in strata_names:
        strata_column = t"""(
            SELECT elem->>'strata_value'
            FROM jsonb_array_elements(aa.strata) AS elem
            WHERE elem->>'field_name' = {strata_name}
            LIMIT 1
        ) AS {strata_name:i}"""
        strata_columns.append(strata_column)
    joined_strata_columns = sql.SQL(", ").join(strata_columns)
    if strata_names:
        extra_columns = t", {joined_strata_columns:q}"
    else:
        extra_columns = sql.SQL("")
    return t"""
        COPY (
            SELECT
                aa.participant_id AS participant_id,
                aa.arm_id AS arm_id,
                a.name AS arm_name,
                aa.created_at AS created_at
                {extra_columns:q}
            FROM arm_assignments AS aa
            JOIN arms AS a
                ON a.id = aa.arm_id
                AND a.experiment_id = aa.experiment_id
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
) -> StreamingResponse:
    _validate_experiment_assignments_csv_export(experiment)
    strata_names = _get_assignment_csv_strata_names_from_experiment(experiment)
    copy_query = _build_experiment_assignments_copy_query(experiment.id, strata_names)
    filename = f"experiment_{experiment.id}_assignments.csv"

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

    return StreamingResponse(
        csv_generator(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
