# mypy: disable-error-code="misc"
from collections.abc import AsyncGenerator

from fastapi.responses import StreamingResponse
from psycopg import sql
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import AssignmentTypedDict, StrataTypedDict
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


def _build_freq_experiment_assignments_select_query(
    experiment_id: str,
    experiment_type: str,
    strata_names: list[str],
    *,
    with_microseconds: bool = False,
):
    if experiment_type not in {ExperimentsType.FREQ_ONLINE.value, ExperimentsType.FREQ_PREASSIGNED.value}:
        raise LateValidationError(f"unsupported experiment type for frequentist export: {experiment_type}")

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

    created_at_column = sql.SQL("aa.created_at AS created_at")
    if not with_microseconds:
        created_at_column = sql.SQL(
            """to_char(aa.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at"""
        )

    return t"""
        SELECT
            aa.participant_id AS participant_id,
            aa.arm_id AS arm_id,
            a.name AS arm_name,
            {created_at_column:q}
            {extra_columns:q}
        FROM arm_assignments AS aa
        JOIN arms AS a
            ON a.id = aa.arm_id
            AND a.experiment_id = aa.experiment_id
        {lateral_join:q}
        WHERE aa.experiment_id = {experiment_id}
    """


def _build_bandit_experiment_assignments_select_query(
    experiment_id: str,
    experiment_type: str,
    *,
    with_microseconds: bool = False,
    include_observed_at: bool = False,
    include_context_vals: bool = False,
):
    if experiment_type not in {ExperimentsType.MAB_ONLINE.value, ExperimentsType.CMAB_ONLINE.value}:
        raise LateValidationError(f"unsupported experiment type for bandit export: {experiment_type}")

    created_at_column = sql.SQL(
        """to_char(draw.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at"""
    )
    if with_microseconds:
        created_at_column = sql.SQL("draw.created_at AS created_at")

    projected_columns = [
        sql.SQL("draw.participant_id AS participant_id"),
        sql.SQL("draw.arm_id AS arm_id"),
        sql.SQL("a.name AS arm_name"),
        created_at_column,
        sql.SQL("draw.outcome AS outcome"),
    ]
    if include_observed_at:
        observed_at_column = sql.SQL(
            """to_char(draw.observed_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS observed_at"""
        )
        if with_microseconds:
            observed_at_column = sql.SQL("draw.observed_at AS observed_at")
        projected_columns.append(observed_at_column)
    if include_context_vals:
        context_values_column = sql.SQL("NULL::float8[] AS context_vals")
        if experiment_type == ExperimentsType.CMAB_ONLINE.value:
            context_values_column = sql.SQL("draw.context_vals AS context_vals")
        projected_columns.append(context_values_column)
    joined_projected_columns = sql.SQL(", ").join(projected_columns)

    return t"""
        SELECT
            {joined_projected_columns:q}
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
    if experiment.experiment_type in {ExperimentsType.FREQ_ONLINE.value, ExperimentsType.FREQ_PREASSIGNED.value}:
        select_query = _build_freq_experiment_assignments_select_query(
            experiment.id, experiment.experiment_type, strata_names
        )
    else:
        select_query = _build_bandit_experiment_assignments_select_query(
            experiment.id,
            experiment.experiment_type,
            include_context_vals=experiment.experiment_type == ExperimentsType.CMAB_ONLINE.value,
        )
    filename = f"experiment_{experiment.id}_assignments.csv"
    return CsvStreamingResponse(
        select_as_csv(xngin_session, select_query, buffer_size_bytes=CSV_STREAM_CHUNK_SIZE_BYTES, include_header=True),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def get_experiment_assignments_impl(
    xngin_session: AsyncSession, experiment: tables.Experiment
) -> AsyncGenerator[AssignmentTypedDict]:
    match experiment.experiment_type:
        case ExperimentsType.FREQ_ONLINE.value | ExperimentsType.FREQ_PREASSIGNED.value:
            strata_names = await _get_assignment_csv_strata_names_from_experiment(experiment)
            select_query = _build_freq_experiment_assignments_select_query(
                experiment.id,
                experiment.experiment_type,
                strata_names,
                with_microseconds=True,
            )
            async for assignment in stream(xngin_session, select_query, JSON_STREAM_FETCH_SIZE_ROWS):
                participant_id, arm_id, arm_name, created_at, *strata_values = assignment
                strata: list[StrataTypedDict] = [
                    {"field_name": strata_names[i], "strata_value": strata_values[i]}
                    for i, _ in enumerate(strata_names)
                ]
                yield {
                    "participant_id": participant_id,
                    "arm_id": arm_id,
                    "arm_name": arm_name,
                    "created_at": created_at,
                    "strata": strata,
                    "observed_at": None,
                    "outcome": None,
                    "context_values": None,
                }
        case ExperimentsType.MAB_ONLINE.value | ExperimentsType.CMAB_ONLINE.value:
            select_query = _build_bandit_experiment_assignments_select_query(
                experiment.id,
                experiment.experiment_type,
                with_microseconds=True,
                include_observed_at=True,
                include_context_vals=True,
            )
            async for assignment in stream(xngin_session, select_query, JSON_STREAM_FETCH_SIZE_ROWS):
                participant_id, arm_id, arm_name, created_at, outcome, observed_at, context_values = assignment
                yield {
                    "participant_id": participant_id,
                    "arm_id": arm_id,
                    "arm_name": arm_name,
                    "created_at": created_at,
                    "strata": [],
                    "observed_at": observed_at,
                    "outcome": outcome,
                    "context_values": context_values,
                }
        case _:
            raise LateValidationError(f"unsupported experiment type for JSON export: {experiment.experiment_type}")
