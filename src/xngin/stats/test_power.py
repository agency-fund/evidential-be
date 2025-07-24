import pytest

from xngin.apiserver.routers.common_api_types import (
    DesignSpecMetric,
    MetricPowerAnalysisMessageType,
    MetricType,
)
from xngin.stats.power import (
    analyze_metric_power,
    check_power,
)
from xngin.stats.stats_errors import StatsPowerError


def test_analyze_metric_power_numeric():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_nonnull_n=1000,
        available_n=1000,
    )

    result = analyze_metric_power(metric, n_arms=2)

    assert result.metric_spec.field_name == "test_metric"
    assert result.metric_spec.metric_type == MetricType.NUMERIC
    assert result.metric_spec.metric_baseline == 100
    assert result.metric_spec.metric_target == 110
    assert result.metric_spec.available_nonnull_n == 1000
    assert result.metric_spec.available_n == 1000
    assert result.target_n == 128.0
    assert result.sufficient_n
    assert result.msg is not None
    assert result.msg.type == MetricPowerAnalysisMessageType.SUFFICIENT
    assert result.msg.values == {
        "available_n": 1000,
        "available_nonnull_n": 1000,
        "target_n": 128.0,
    }
    assert result.msg.msg == result.msg.source_msg.format_map(result.msg.values)


def test_analyze_metric_power_binary():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.BINARY,
        metric_baseline=0.5,
        metric_target=0.55,
        available_nonnull_n=1000,
        available_n=1000,
    )

    result = analyze_metric_power(metric, n_arms=2)

    assert result.metric_spec.metric_type == MetricType.BINARY
    assert result.metric_spec.metric_baseline == 0.5
    assert result.metric_spec.metric_target == 0.55
    assert result.target_n == 3132
    # Given the available_n, here's the best we can do (cross-checked with R's power.prop.test):
    # (since it's 2-sided, an equivalent change down is fine, too)
    assert result.target_possible == pytest.approx(1 - 0.588163, abs=1e-4)
    assert result.pct_change_possible == pytest.approx(1 - 1.176327, abs=1e-4)
    assert not result.sufficient_n
    assert result.msg is not None
    assert result.msg.type == MetricPowerAnalysisMessageType.INSUFFICIENT
    assert result.msg.values == {
        "available_n": 1000,
        "available_nonnull_n": 1000,
        "target_n": 3132,
        "additional_n_needed": 3132 - 1000,
        "metric_baseline": 0.5,
        "target_possible": pytest.approx(1 - 0.588163, abs=1e-4),
        "metric_target": 0.55,
    }
    assert result.msg.msg == result.msg.source_msg.format_map(result.msg.values)


def test_check_power_multiple_metrics():
    metrics = [
        DesignSpecMetric(
            field_name="metric1",
            metric_type=MetricType.NUMERIC,
            metric_baseline=100,
            metric_target=110,
            metric_stddev=20,
            available_nonnull_n=1000,
            available_n=1000,
        ),
        DesignSpecMetric(
            field_name="metric2",
            metric_type=MetricType.BINARY,
            metric_baseline=0.5,
            metric_target=0.55,
            available_nonnull_n=1000,
            available_n=1000,
        ),
    ]

    results = check_power(metrics, n_arms=2)

    assert len(results) == 2
    assert results[0].metric_spec.field_name == "metric1"
    assert results[1].metric_spec.field_name == "metric2"


def test_check_missing_metric_type_raises_error():
    metrics = [
        DesignSpecMetric(
            field_name="bad_metric",
            metric_baseline=0.5,
            metric_target=0.5,
            available_n=1000,
            available_nonnull_n=1000,
        ),
    ]

    with pytest.raises(StatsPowerError) as excinfo:
        check_power(metrics, n_arms=2)
    assert "Unknown metric_type" in str(excinfo.value)


def test_analyze_metric_with_no_available_n_returns_friendly_error():
    metric = DesignSpecMetric(
        field_name="no_available_n",
        metric_type=MetricType.BINARY,
        metric_baseline=None,
        metric_target=None,
        available_n=0,
        available_nonnull_n=0,
    )

    result = analyze_metric_power(metric, n_arms=2)

    assert result.msg is not None
    assert result.msg.type == MetricPowerAnalysisMessageType.NO_AVAILABLE_N
    assert "Adjust your filters to target more units." in result.msg.msg


def test_analyze_metric_missing_baseline_returns_friendly_error():
    metric = DesignSpecMetric(
        field_name="missing_baseline",
        metric_type=MetricType.BINARY,
        metric_baseline=None,
        metric_target=None,
        available_n=1000,
        available_nonnull_n=100,
    )

    result = analyze_metric_power(metric, n_arms=2)

    assert result.msg is not None
    assert result.msg.type == MetricPowerAnalysisMessageType.NO_BASELINE
    assert "Could not calculate metric baseline" in result.msg.msg


def test_analyze_metric_with_no_available_nonnull_n_returns_ok():
    metric = DesignSpecMetric(
        field_name="no_available_nonnull_n",
        metric_type=MetricType.BINARY,
        metric_baseline=0.5,
        metric_target=0.6,
        available_n=1000,
        available_nonnull_n=0,
    )

    result = analyze_metric_power(metric, n_arms=2)

    assert result.msg
    assert result.msg.type == MetricPowerAnalysisMessageType.SUFFICIENT
    assert "There are enough units available." in result.msg.msg
    assert result.msg.values == {
        "available_n": 1000,
        "available_nonnull_n": 0,
        "target_n": 778,
    }


def test_analyze_metric_zero_effect_size_returns_friendly_error():
    metric = DesignSpecMetric(
        field_name="zero_effect_size",
        metric_type=MetricType.BINARY,
        metric_baseline=0.5,
        metric_target=0.5,
        available_n=1000,
        available_nonnull_n=1000,
    )

    result = analyze_metric_power(metric, n_arms=2)

    assert result.msg is not None
    assert result.msg.type == MetricPowerAnalysisMessageType.ZERO_EFFECT_SIZE
    assert "Cannot detect an effect-size of 0" in result.msg.msg


def test_analyze_metric_zero_stddev_returns_friendly_error():
    metric = DesignSpecMetric(
        field_name="zero_stddev",
        metric_type=MetricType.NUMERIC,
        metric_baseline=0.5,
        metric_target=0.6,
        metric_stddev=0,
        available_n=1000,
        available_nonnull_n=1000,
    )

    result = analyze_metric_power(metric, n_arms=2)

    assert result.msg is not None
    assert result.msg.type == MetricPowerAnalysisMessageType.ZERO_STDDEV
    assert "There is no variation in the metric" in result.msg.msg
