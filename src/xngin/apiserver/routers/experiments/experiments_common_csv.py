# mypy: disable-error-code="misc"


from fastapi.responses import StreamingResponse
from psycopg import sql
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_enums import ExperimentsType
from xngin.apiserver.sql.queries import select_as_csv
from xngin.apiserver.sqla import tables

CSV_STREAM_CHUNK_SIZE_BYTES = 1 << 20


class CsvStreamingResponse(StreamingResponse):
    media_type = "text/csv"


def _get_assignment_csv_strata_names_from_experiment(experiment: tables.Experiment) -> list[str]:
    return sorted([ef.field_name for ef in experiment.experiment_fields if ef.is_strata])


def _build_experiment_assignments_select_query(experiment_id: str, strata_names: list[str]):
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
    select_query = _build_experiment_assignments_select_query(experiment.id, strata_names)

    filename = f"experiment_{experiment.id}_assignments.csv"
    return CsvStreamingResponse(
        select_as_csv(xngin_session, select_query, buffer_size_bytes=CSV_STREAM_CHUNK_SIZE_BYTES, include_header=True),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
