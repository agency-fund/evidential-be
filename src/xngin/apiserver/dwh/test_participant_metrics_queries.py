from xngin.apiserver.dwh.analysis_types import MetricValue, ParticipantOutcome
from xngin.apiserver.dwh.participant_metrics_queries import get_participant_metrics
from xngin.apiserver.routers.common_api_types import DesignSpecMetricRequest

pytest_plugins = ("xngin.apiserver.dwh.dwh_test_support",)


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
