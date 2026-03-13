"""Tests for queries.py."""

import pytest

from xngin.apiserver.dwh.queries import get_stats_on_metrics
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import DesignSpecMetric, DesignSpecMetricRequest
from xngin.apiserver.routers.common_enums import MetricType

pytest_plugins = ("xngin.apiserver.dwh.dwh_test_support",)


def test_get_stats_on_missing_metric_raises_error(queries_dwh_session, shared_sample_tables):
    with pytest.raises(LateValidationError) as exc:
        get_stats_on_metrics(
            queries_dwh_session,
            shared_sample_tables.sample_table,
            [DesignSpecMetricRequest(field_name="missing_col", metric_pct_change=0.1)],
            filters=[],
        )
    assert "Missing metrics (check your Datasource configuration): {'missing_col'}" in str(exc.value)


def test_get_stats_on_integer_metric(queries_dwh_session, shared_sample_tables):
    """Test would fail on postgres and redshift without a cast to float for different reasons."""
    rows = get_stats_on_metrics(
        queries_dwh_session,
        shared_sample_tables.sample_table,
        [DesignSpecMetricRequest(field_name="int_col", metric_pct_change=0.1)],
        filters=[],
    )

    expected = DesignSpecMetric(
        field_name="int_col",
        metric_type=MetricType.NUMERIC,
        metric_baseline=41.666666666666664,
        metric_stddev=47.76563153100307,
        available_nonnull_n=3,
        available_n=3,
    )
    assert len(rows) == 1
    actual = rows[0]
    numeric_fields = {
        "metric_baseline",
        "metric_stddev",
        "available_nonnull_n",
        "available_n",
    }
    assert actual.field_name == expected.field_name
    assert actual.metric_type == expected.metric_type
    # PG: assertion would fail due to a float vs decimal.Decimal comparison.
    # RS: assertion would fail due to avg() on int types keeps them as integers.
    assert actual.model_dump(include=numeric_fields) == pytest.approx(expected.model_dump(include=numeric_fields))


def test_get_stats_on_nullable_integer_metric(queries_dwh_session, shared_sample_tables):
    rows = get_stats_on_metrics(
        queries_dwh_session,
        shared_sample_tables.sample_nullable_table,
        [DesignSpecMetricRequest(field_name="int_col", metric_pct_change=0.1)],
        filters=[],
    )

    expected = DesignSpecMetric(
        field_name="int_col",
        metric_type=MetricType.NUMERIC,
        metric_baseline=2.0,
        metric_stddev=1.0,
        available_nonnull_n=2,
        available_n=3,
    )
    assert len(rows) == 1
    actual = rows[0]
    numeric_fields = {
        "metric_baseline",
        "metric_stddev",
        "available_nonnull_n",
        "available_n",
    }
    assert actual.field_name == expected.field_name
    assert actual.metric_type == expected.metric_type
    assert actual.model_dump(include=numeric_fields) == pytest.approx(expected.model_dump(include=numeric_fields))


def test_get_stats_on_boolean_metric(queries_dwh_session, shared_sample_tables):
    """Test would fail on postgres and redshift without casting to int to float."""
    rows = get_stats_on_metrics(
        queries_dwh_session,
        shared_sample_tables.sample_table,
        [DesignSpecMetricRequest(field_name="bool_col", metric_pct_change=0.1)],
        filters=[],
    )

    expected = DesignSpecMetric(
        field_name="bool_col",
        metric_type=MetricType.BINARY,
        metric_baseline=0.6666666666666666,
        metric_stddev=None,
        available_nonnull_n=3,
        available_n=3,
    )
    assert len(rows) == 1
    actual = rows[0]
    numeric_fields = {
        "metric_baseline",
        "metric_stddev",
        "available_nonnull_n",
        "available_n",
    }
    assert actual.field_name == expected.field_name
    assert actual.metric_type == expected.metric_type
    assert actual.model_dump(include=numeric_fields) == pytest.approx(expected.model_dump(include=numeric_fields))


def test_get_stats_on_numeric_metric(queries_dwh_session, shared_sample_tables):
    rows = get_stats_on_metrics(
        queries_dwh_session,
        shared_sample_tables.sample_table,
        [DesignSpecMetricRequest(field_name="float_col", metric_pct_change=0.1)],
        filters=[],
    )

    expected = DesignSpecMetric(
        field_name="float_col",
        metric_type=MetricType.NUMERIC,
        metric_baseline=2.492,
        metric_stddev=0.6415751449882287,
        available_nonnull_n=3,
        available_n=3,
    )
    assert len(rows) == 1
    actual = rows[0]
    numeric_fields = {
        "metric_baseline",
        "metric_stddev",
        "available_nonnull_n",
        "available_n",
    }
    assert actual.field_name == expected.field_name
    assert actual.metric_type == expected.metric_type
    # pytest.approx does a reasonable fuzzy comparison of floats for non-nested dictionaries.
    assert actual.model_dump(include=numeric_fields) == pytest.approx(expected.model_dump(include=numeric_fields))
