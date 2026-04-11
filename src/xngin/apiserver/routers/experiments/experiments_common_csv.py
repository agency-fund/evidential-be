# mypy: disable-error-code="misc"
from collections.abc import AsyncGenerator

from fastapi.responses import StreamingResponse
from psycopg import sql
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_enums import ExperimentsType
from xngin.apiserver.sql.queries import select_as_csv, stream
from xngin.apiserver.sqla import tables

CSV_STREAM_CHUNK_SIZE_BYTES = 1 << 20
JSON_STREAM_FETCH_SIZE_ROWS = 16_384


class CsvStreamingResponse(StreamingResponse):
    media_type = "text/csv"


async def _get_assignment_csv_strata_names_from_experiment(experiment: tables.Experiment) -> list[str]:
    if experiment.experiment_type not in {ExperimentsType.FREQ_ONLINE.value, ExperimentsType.FREQ_PREASSIGNED.value}:
        return []
    return sorted([ef.field_name for ef in await experiment.awaitable_attrs.experiment_fields if ef.is_strata])


def _build_experiment_assignments_select_query(experiment_id: str, experiment_type: str, strata_names: list[str]):
    match experiment_type:
        case ExperimentsType.FREQ_ONLINE.value | ExperimentsType.FREQ_PREASSIGNED.value:
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
        case ExperimentsType.MAB_ONLINE.value | ExperimentsType.CMAB_ONLINE.value:
            extra_columns = sql.SQL("")
            if experiment_type == ExperimentsType.CMAB_ONLINE.value:
                extra_columns = sql.SQL(", draw.context_vals AS context_vals")
            return t"""
                SELECT
                    draw.participant_id AS participant_id,
                    draw.arm_id AS arm_id,
                    a.name AS arm_name,
                    to_char(draw.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at,
                    draw.outcome as outcome
                    {extra_columns:q}
                FROM draws AS draw
                JOIN arms AS a
                    ON a.id = draw.arm_id
                    AND a.experiment_id = draw.experiment_id
                WHERE draw.experiment_id = {experiment_id}
            """
        case _:
            raise LateValidationError(f"unsupported experiment type for CSV export: {experiment_type}")


def _build_bandit_experiment_assignments_json_select_query(experiment_id: str, experiment_type: str):
    extra_columns = sql.SQL(", NULL::float8[] AS context_vals")
    if experiment_type == ExperimentsType.CMAB_ONLINE.value:
        extra_columns = sql.SQL(", draw.context_vals AS context_vals")
    return t"""
        SELECT
            draw.participant_id AS participant_id,
            draw.arm_id AS arm_id,
            a.name AS arm_name,
            to_char(draw.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at,
            to_char(draw.observed_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS observed_at,
            draw.outcome AS outcome
            {extra_columns:q}
        FROM draws AS draw
        JOIN arms AS a
            ON a.id = draw.arm_id
            AND a.experiment_id = draw.experiment_id
        WHERE draw.experiment_id = {experiment_id}
    """


async def get_experiment_assignments_as_csv_impl(
    xngin_session: AsyncSession,
    experiment: tables.Experiment,
) -> CsvStreamingResponse:
    strata_names = await _get_assignment_csv_strata_names_from_experiment(experiment)
    select_query = _build_experiment_assignments_select_query(experiment.id, experiment.experiment_type, strata_names)
    filename = f"experiment_{experiment.id}_assignments.csv"
    return CsvStreamingResponse(
        select_as_csv(xngin_session, select_query, buffer_size_bytes=CSV_STREAM_CHUNK_SIZE_BYTES, include_header=True),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def get_experiment_assignments_impl(
    xngin_session: AsyncSession, experiment: tables.Experiment
) -> AsyncGenerator[dict[str, object]]:
    match experiment.experiment_type:
        case ExperimentsType.FREQ_ONLINE.value | ExperimentsType.FREQ_PREASSIGNED.value:
            strata_names = await _get_assignment_csv_strata_names_from_experiment(experiment)
            select_query = _build_experiment_assignments_select_query(
                experiment.id, experiment.experiment_type, strata_names
            )
            async for assignment in stream(xngin_session, select_query, JSON_STREAM_FETCH_SIZE_ROWS):
                participant_id, arm_id, arm_name, created_at, *strata_values = assignment
                yield {
                    "participant_id": participant_id,
                    "arm_id": arm_id,
                    "arm_name": arm_name,
                    "created_at": created_at,
                    "strata": [
                        {"field_name": strata_names[i], "strata_value": strata_values[i]}
                        for i, _ in enumerate(strata_names)
                    ],
                    "observed_at": None,
                    "outcome": None,
                    "context_values": None,
                }
        case ExperimentsType.MAB_ONLINE.value | ExperimentsType.CMAB_ONLINE.value:
            select_query = _build_bandit_experiment_assignments_json_select_query(
                experiment.id, experiment.experiment_type
            )
            async for assignment in stream(xngin_session, select_query, JSON_STREAM_FETCH_SIZE_ROWS):
                participant_id, arm_id, arm_name, created_at, observed_at, outcome, context_values = assignment
                yield {
                    "participant_id": participant_id,
                    "arm_id": arm_id,
                    "arm_name": arm_name,
                    "created_at": created_at,
                    "strata": None,
                    "observed_at": observed_at,
                    "outcome": outcome,
                    "context_values": context_values,
                }
        case _:
            raise LateValidationError(f"unsupported experiment type for JSON export: {experiment.experiment_type}")
