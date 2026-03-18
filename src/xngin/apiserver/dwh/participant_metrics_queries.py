from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import sqlalchemy
from loguru import logger
from sqlalchemy import Float, Integer, Label, String, Table, cast, or_, select
from sqlalchemy.orm import Session

from xngin.apiserver.dwh.analysis_types import MetricValue, ParticipantOutcome
from xngin.apiserver.dwh.query_constructors import create_one_filter
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import DesignSpecMetricRequest, Filter
from xngin.apiserver.routers.common_enums import MetricType, Relation

if TYPE_CHECKING:
    from numpy._typing import NDArray

# Target number of rows to return when using an IN-expression.
PARTICIPANT_BATCH_SIZE = 10_000

# Maximum size of a range used in a BETWEEN-expression.
MAXIMUM_ROWS_FOR_BETWEEN = PARTICIPANT_BATCH_SIZE * 3


def to_np_int_arr(values: Sequence[str]) -> NDArray[np.int64] | None:
    """Converts a sequence of strings to a numpy integer array.

    Returns None if not all values are integers.
    """
    arr = np.array(values, dtype=str)
    try:
        return arr.astype(np.int64)
    except ValueError:
        return None


def identify_runs(arr: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return parallel arrays of inclusive start and end values for each consecutive integer run."""
    # Sort the participant ids so consecutive values become adjacent.
    arr = np.sort(arr)
    # Find the positions where adjacency breaks.
    breaks = np.where(np.diff(arr) != 1)[0] + 1
    # Build the indices of each range's first element.
    starts_idx = np.concatenate(([0], breaks))
    # Build the indices of each range's last element.
    ends_idx = np.concatenate((breaks - 1, [len(arr) - 1]))
    # Return the values at those boundary indices as parallel start/end arrays.
    return arr[starts_idx], arr[ends_idx]


@dataclass(frozen=True, slots=True)
class ParticipantChunk:
    """Compact intermediate representation of a participant ID filters INCLUDES(...) or BETWEEN(low, high)."""

    is_includes: bool
    value_str: list[str] | None
    value_int: list[int] | None
    size: int

    def relation(self) -> Relation:
        return Relation.INCLUDES if self.is_includes else Relation.BETWEEN

    def to_filter(self, unique_id_field: str, unique_id_type: sqlalchemy.types.TypeEngine) -> Filter:
        if self.value_int is not None:
            return Filter(
                field_name=unique_id_field,
                relation=self.relation(),
                value=self.value_int,
            )
        if self.value_str is not None:
            return Filter(
                field_name=unique_id_field,
                relation=self.relation(),
                value=[Filter.cast_participant_id(pid, unique_id_type) for pid in self.value_str],
            )
        raise RuntimeError("Bug: chunk has neither value_int nor value_str")


@dataclass(frozen=True)
class QueryPlan:
    query: sqlalchemy.Select
    chunks: list[ParticipantChunk]

    def summary(self) -> str:
        expected_results = sum(chunk.size for chunk in self.chunks)
        op_counts = ",".join([f"{t}={c}" for t, c in Counter(chunk.relation().value for chunk in self.chunks).items()])
        return f"relations={op_counts} expected={expected_results}"


@dataclass(frozen=True)
class ParticipantMetricsPlans:
    field_names: list[str]
    plans: list[QueryPlan]


def naive_strategy(participant_ids: list[str]) -> list[ParticipantChunk]:
    """Generates ParticipantChunks using INCLUDES operators for up to PARTICIPANT_BATCH_SIZE ids at a time."""
    pids = sorted(participant_ids)
    return [
        ParticipantChunk(
            is_includes=True,
            value_str=pids[batch_start : batch_start + PARTICIPANT_BATCH_SIZE],
            value_int=None,
            size=len(pids[batch_start : batch_start + PARTICIPANT_BATCH_SIZE]),
        )
        for batch_start in range(0, len(pids), PARTICIPANT_BATCH_SIZE)
    ]


def between_strategy(integer_ids: NDArray[np.int64]) -> list[ParticipantChunk]:
    """Generates ParticipantChunks using BETWEEN (and sometimes INCLUDES) for integer IDs."""
    # Extract ranges of consecutive integers so that we can generate BETWEEN and IN queries from them.
    range_starts, range_ends = identify_runs(integer_ids)

    chunks: list[ParticipantChunk] = []
    singleton_ids: list[int] = []
    for start, end in zip(range_starts, range_ends, strict=False):
        start_int = int(start)
        end_int = int(end)
        # When indexes are equal, we have a value that is not consecutive with its neighbors.
        if start_int == end_int:
            singleton_ids.append(start_int)
            if len(singleton_ids) == PARTICIPANT_BATCH_SIZE:
                chunks.append(
                    ParticipantChunk(
                        is_includes=True,
                        value_str=None,
                        value_int=singleton_ids,
                        size=len(singleton_ids),
                    )
                )
                singleton_ids = []
            continue
        # When indexes are not equal, we have a range of consecutive integers. These turn into BETWEEN expressions;
        # because BETWEEN is cheaper than IN here, we allow larger range chunks than the general participant batch size.
        for range_start in range(start_int, end_int + 1, MAXIMUM_ROWS_FOR_BETWEEN):
            range_end = min(range_start + MAXIMUM_ROWS_FOR_BETWEEN - 1, end_int)
            chunks.append(
                ParticipantChunk(
                    is_includes=False,
                    value_str=None,
                    value_int=[range_start, range_end],
                    size=range_end - range_start + 1,
                )
            )
    if singleton_ids:
        chunks.append(
            ParticipantChunk(
                is_includes=True,
                value_str=None,
                value_int=singleton_ids,
                size=len(singleton_ids),
            )
        )
    return chunks


def make_participant_chunks(
    participant_id_column: sqlalchemy.Column, participant_ids: list[str]
) -> list[ParticipantChunk]:
    """Transform a list of participant IDs into a list of selection operations (Chunks)."""
    if not participant_ids:
        return []

    # If the unique ID column is an integer type, we can try an optimization strategy.
    try_integer_optimization = isinstance(participant_id_column.type, sqlalchemy.sql.sqltypes.Integer)

    # If we aren't trying an optimization, use a naive approach of using IN statements.
    if not try_integer_optimization:
        return naive_strategy(participant_ids)

    # Is the list of participants exclusively integers?
    integer_participant_ids = to_np_int_arr(participant_ids)
    if integer_participant_ids is None:
        return naive_strategy(participant_ids)

    return between_strategy(integer_participant_ids)


def coalesce_chunks_into_disjunctives(participant_filters: list[ParticipantChunk]) -> list[list[ParticipantChunk]]:
    query_groups: list[list[ParticipantChunk]] = []
    current_group: list[ParticipantChunk] = []
    current_group_size = 0

    for participant_filter in participant_filters:
        if current_group and current_group_size + participant_filter.size > PARTICIPANT_BATCH_SIZE:
            query_groups.append(current_group)
            current_group = []
            current_group_size = 0
        current_group.append(participant_filter)
        current_group_size += participant_filter.size

    if current_group:
        query_groups.append(current_group)
    return query_groups


def build_participant_metrics_plan(
    sa_table: Table,
    metrics: list[DesignSpecMetricRequest],
    unique_id_field: str,
    participant_ids: list[str],
) -> ParticipantMetricsPlans:
    unique_id_col: sqlalchemy.Column = sa_table.c[unique_id_field]
    select_columns: list[Label] = [cast(unique_id_col, String).label("participant_id")]

    # query for a participant_id column (the unique id column) and the metrics columns.
    field_names = ["participant_id"]
    metric_types = [MetricType.from_python_type(sa_table.c[m.field_name].type.python_type) for m in metrics]
    for metric, metric_type in zip(metrics, metric_types, strict=False):
        field_name = metric.field_name
        field_names.append(field_name)
        col = sa_table.c[field_name]
        # Coerce everything to Float to avoid Decimal/Integer/Boolean issues across backends.
        if metric_type is MetricType.NUMERIC:
            cast_column = cast(col, Float)
        else:  # re: avg(boolean) doesn't work on pg-like backends
            cast_column = cast(cast(col, Integer), Float)
        select_columns.append(cast_column.label(field_name))

    chunks = make_participant_chunks(unique_id_col, participant_ids)
    coalesced_chunks = coalesce_chunks_into_disjunctives(chunks)

    between_filter_count = sum(1 for chunk in chunks if not chunk.is_includes)
    includes_filter_count = len(chunks) - between_filter_count
    logger.info(
        f"Coalesced {len(chunks)} filters into {len(coalesced_chunks)} queries: "
        f"# of between={between_filter_count}, # of includes={includes_filter_count}, "
        f"batch size={PARTICIPANT_BATCH_SIZE}"
    )

    query_plans: list[QueryPlan] = []
    for batch_chunks in coalesced_chunks:
        batch_filters = [filter_op.to_filter(unique_id_field, unique_id_col.type) for filter_op in batch_chunks]
        group_filter = or_(*[create_one_filter(filter_, sa_table) for filter_ in batch_filters])
        query_plans.append(QueryPlan(query=select(*select_columns).filter(group_filter), chunks=batch_chunks))

    return ParticipantMetricsPlans(
        field_names=field_names,
        plans=query_plans,
    )


def get_participant_metrics(
    session: Session,
    sa_table: Table,
    metrics: list[DesignSpecMetricRequest],
    unique_id_field: str,
    participant_ids: list[str],
) -> list[ParticipantOutcome]:
    logger.info(
        "Fetching participant metrics: table={} unique_id_field={} #participant_ids={} metrics={}",
        sa_table.name,
        unique_id_field,
        len(participant_ids),
        [metric.field_name for metric in metrics],
    )
    missing_metrics = {m.field_name for m in metrics if m.field_name not in sa_table.c}
    if len(missing_metrics) > 0:
        raise LateValidationError(f"Missing metrics (check your Datasource configuration): {missing_metrics}")
    if unique_id_field not in sa_table.columns:
        raise LateValidationError(f"Unique ID field {unique_id_field} not found in table.")

    pmplans = build_participant_metrics_plan(
        sa_table=sa_table,
        metrics=metrics,
        unique_id_field=unique_id_field,
        participant_ids=participant_ids,
    )
    participant_outcomes: list[ParticipantOutcome] = []
    for batch_index, plan in enumerate(pmplans.plans, start=1):
        logger.info("Running participant metrics batch {}/{}: {}", batch_index, len(pmplans.plans), plan.summary())
        results = session.execute(plan.query)

        batch_outcome_count = 0
        for result in results:
            metric_values: list[MetricValue] = []
            participant_id = None
            for i, field_name in enumerate(pmplans.field_names):
                if field_name == "participant_id":
                    participant_id = result[i]
                else:
                    metric_values.append(MetricValue(metric_name=field_name, metric_value=result[i]))
            if participant_id is None:
                # Should never happen as we filter on the participant_id field.
                raise LateValidationError("Participant ID is required.")
            participant_outcomes.append(
                ParticipantOutcome(participant_id=str(participant_id), metric_values=metric_values)
            )
            batch_outcome_count += 1
        logger.info(
            "Finished participant metrics batch {}/{} outcomes={}",
            batch_index,
            len(pmplans.plans),
            batch_outcome_count,
        )
    logger.info(
        "Finished fetching participant metrics outcomes={}",
        len(participant_outcomes),
    )
    return participant_outcomes
