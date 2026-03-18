import numpy as np
from sqlalchemy import Column, Integer, String

from xngin.apiserver.dwh.analysis_types import MetricValue, ParticipantOutcome
from xngin.apiserver.dwh.participant_metrics_queries import (
    MAXIMUM_ROWS_FOR_BETWEEN,
    PARTICIPANT_BATCH_SIZE,
    ParticipantChunk,
    between_strategy,
    build_participant_metrics_plan,
    coalesce_chunks_into_disjunctives,
    get_participant_metrics,
    identify_runs,
    make_participant_chunks,
    to_np_int_arr,
)
from xngin.apiserver.routers.common_api_types import DesignSpecMetricRequest

pytest_plugins = ("xngin.apiserver.dwh.dwh_test_support",)


def test_returns_none_when_not_all_integer_values():
    assert to_np_int_arr(["1", "abc", "3"]) is None


def test_to_np_int_arr_returns_none_for_float_values():
    float_arr = [1.0]
    assert to_np_int_arr(float_arr) is None  # type: ignore [arg-type]
    assert to_np_int_arr(["1.0"]) is None
    assert to_np_int_arr(["NaN"]) is None


def test_to_np_int_arr_converts_integer_strings():
    actual = to_np_int_arr(["1", "-3", "42", "0"])
    assert actual is not None
    assert actual.tolist() == [1, -3, 42, 0]


def test_identify_runs_sorts_and_groups_consecutive_values():
    starts, ends = identify_runs(np.array([9, 3, 2, 6, 5]))
    assert list(zip(starts.tolist(), ends.tolist(), strict=True)) == [(2, 3), (5, 6), (9, 9)]


def test_make_participant_chunks_uses_naive_strategy_for_string_columns():
    chunks = make_participant_chunks(Column("participant_id", String()), ["30", "10", "20"])

    assert chunks == [
        ParticipantChunk(
            is_includes=True,
            value_str=["10", "20", "30"],
            value_int=None,
            size=3,
        )
    ]


def test_make_participant_chunks_uses_naive_strategy_for_non_integer_values():
    chunks = make_participant_chunks(Column("participant_id", Integer()), ["10", "abc", "20"])

    assert chunks == [
        ParticipantChunk(
            is_includes=True,
            value_str=["10", "20", "abc"],
            value_int=None,
            size=3,
        )
    ]


def test_between_strategy_splits_large_ranges():
    chunks = between_strategy(np.arange(1, MAXIMUM_ROWS_FOR_BETWEEN + 5, dtype=np.int64))

    assert [(chunk.is_includes, chunk.value_int, chunk.size) for chunk in chunks] == [
        (False, [1, MAXIMUM_ROWS_FOR_BETWEEN], MAXIMUM_ROWS_FOR_BETWEEN),
        (False, [MAXIMUM_ROWS_FOR_BETWEEN + 1, MAXIMUM_ROWS_FOR_BETWEEN + 4], 4),
    ]


def test_coalesce_chunks_into_disjunctives_splits_on_size_and_not_values():
    chunks = [
        ParticipantChunk(is_includes=False, value_str=None, value_int=[1, 4_000], size=4_000),
        ParticipantChunk(is_includes=False, value_str=None, value_int=[5_000, 8_999], size=4_000),
        ParticipantChunk(is_includes=True, value_str=None, value_int=[20_000, 20_002], size=3_000),
    ]

    actual = coalesce_chunks_into_disjunctives(chunks)

    assert actual == [
        chunks[:2],
        chunks[2:],
    ]


def test_get_participant_metrics(queries_dwh_session, shared_sample_tables):
    participant_ids = ["100", "200"]
    rows = get_participant_metrics(
        queries_dwh_session,
        shared_sample_tables.sample_table,
        [
            DesignSpecMetricRequest(field_name="float_col", metric_pct_change=0.1),
            DesignSpecMetricRequest(field_name="bool_col", metric_pct_change=0.1),
        ],
        unique_id_field="id",
        participant_ids=participant_ids,
    )

    expected = [
        ParticipantOutcome(
            participant_id="100",
            metric_values=[
                MetricValue(metric_name="float_col", metric_value=3.14),
                MetricValue(metric_name="bool_col", metric_value=True),
            ],
        ),
        ParticipantOutcome(
            participant_id="200",
            metric_values=[
                MetricValue(metric_name="float_col", metric_value=2.718),
                MetricValue(metric_name="bool_col", metric_value=False),
            ],
        ),
    ]

    assert len(rows) == len(expected)
    rows = sorted(rows, key=lambda row: row.participant_id)
    for actual, exp in zip(rows, expected, strict=False):
        assert actual.participant_id == exp.participant_id
        assert actual.metric_values[0].metric_name == exp.metric_values[0].metric_name
        assert actual.metric_values[0].metric_value == exp.metric_values[0].metric_value


def test_build_participant_metrics_query_plans_batches_sorted_ids(shared_sample_tables):
    participant_ids = [str(i) for i in range(PARTICIPANT_BATCH_SIZE * 8 + 2, 2, -4)]
    participant_ids.extend(["100", "200", "300"])

    query_plan_set = build_participant_metrics_plan(
        shared_sample_tables.sample_table,
        [
            DesignSpecMetricRequest(field_name="float_col", metric_pct_change=0.1),
            DesignSpecMetricRequest(field_name="bool_col", metric_pct_change=0.1),
        ],
        unique_id_field="id",
        participant_ids=participant_ids,
    )
    query_plans = query_plan_set.plans

    assert len(query_plans) == 3

    batch_values = [plan.query.compile().params["id_1"] for plan in query_plans]
    assert [len(batch) for batch in batch_values] == [10_000, 10_000, 3]
    assert all(" IN " in str(plan.query.compile()) for plan in query_plans)
    all_batch_values = [value for batch in batch_values for value in batch]
    assert all_batch_values == sorted({int(pid) for pid in participant_ids})


def test_build_participant_metrics_query_plans_batches_integer_ranges(shared_sample_tables):
    participant_ids = [str(i) for i in range(PARTICIPANT_BATCH_SIZE, 0, -1)]
    participant_ids.extend(["20000", "20002"])

    query_plan_set = build_participant_metrics_plan(
        shared_sample_tables.sample_table,
        [
            DesignSpecMetricRequest(field_name="float_col", metric_pct_change=0.1),
            DesignSpecMetricRequest(field_name="bool_col", metric_pct_change=0.1),
        ],
        unique_id_field="id",
        participant_ids=participant_ids,
    )
    query_plans = query_plan_set.plans

    assert len(query_plans) == 2

    between_query = query_plans[0].query.compile()
    includes_query = query_plans[1].query.compile()

    assert " BETWEEN " in str(between_query)
    assert between_query.params == {"id_1": 1, "id_2": PARTICIPANT_BATCH_SIZE}

    assert " IN " in str(includes_query)
    assert includes_query.params == {"id_1": [20_000, 20_002]}


def test_build_participant_metrics_query_plans_coalesces_between_filters(shared_sample_tables):
    participant_ids = [str(i) for start in range(1, PARTICIPANT_BATCH_SIZE * 3 // 2, 3) for i in (start, start + 1)]

    query_plan_set = build_participant_metrics_plan(
        shared_sample_tables.sample_table,
        [
            DesignSpecMetricRequest(field_name="float_col", metric_pct_change=0.1),
            DesignSpecMetricRequest(field_name="bool_col", metric_pct_change=0.1),
        ],
        unique_id_field="id",
        participant_ids=participant_ids,
    )
    query_plans = query_plan_set.plans

    assert len(query_plans) == 1

    query = query_plans[0].query.compile()
    query_sql = str(query)
    assert query_sql.count(" BETWEEN ") > 1
    assert " OR " in query_sql


def test_build_participant_metrics_query_plans_keeps_single_large_between_range(shared_sample_tables):
    participant_ids = [str(i) for i in range(PARTICIPANT_BATCH_SIZE + 5, 0, -1)]

    query_plan_set = build_participant_metrics_plan(
        shared_sample_tables.sample_table,
        [
            DesignSpecMetricRequest(field_name="float_col", metric_pct_change=0.1),
            DesignSpecMetricRequest(field_name="bool_col", metric_pct_change=0.1),
        ],
        unique_id_field="id",
        participant_ids=participant_ids,
    )
    query_plans = query_plan_set.plans

    assert len(query_plans) == 1

    query = query_plans[0].query.compile()
    assert " BETWEEN " in str(query)
    assert query.params == {"id_1": 1, "id_2": PARTICIPANT_BATCH_SIZE + 5}
