import pytest
from xngin.stats.power import (
    DesignSpecMetric,
    MetricType,
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
        available_n=1000,
    )

    result = analyze_metric_power(metric, n_arms=2)

    assert result.metric_spec.field_name == "test_metric"
    assert result.metric_spec.metric_type == MetricType.NUMERIC
    assert result.metric_spec.metric_baseline == 100
    assert result.metric_spec.metric_target == 110
    assert result.metric_spec.available_n == 1000
    assert result.target_n == 128.0
    assert result.sufficient_n
    assert result.msg is not None


def test_analyze_metric_power_binary():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.BINARY,
        metric_baseline=0.5,
        metric_target=0.55,
        available_n=1000,
    )

    result = analyze_metric_power(metric, n_arms=2)

    assert result.metric_spec.metric_type == MetricType.BINARY
    assert result.metric_spec.metric_baseline == 0.5
    assert result.metric_spec.metric_target == 0.55
    assert result.target_n == 3132
    # Given the available_n, here's the best we can do (cross-checked with R's power.prop.test):
    # (since it's 2-sided, an equivalent change down is fine, too)
    assert result.needed_target == pytest.approx(1 - 0.588163, abs=1e-4)
    assert result.pct_change_possible == pytest.approx(1 - 1.176327, abs=1e-4)
    assert not result.sufficient_n


def test_check_power_multiple_metrics():
    metrics = [
        DesignSpecMetric(
            field_name="metric1",
            metric_type=MetricType.NUMERIC,
            metric_baseline=100,
            metric_target=110,
            metric_stddev=20,
            available_n=1000,
        ),
        DesignSpecMetric(
            field_name="metric2",
            metric_type=MetricType.BINARY,
            metric_baseline=0.5,
            metric_target=0.55,
            available_n=1000,
        ),
    ]

    results = check_power(metrics, n_arms=2)

    assert len(results) == 2
    assert results[0].metric_spec.field_name == "metric1"
    assert results[1].metric_spec.field_name == "metric2"


def test_check_power_effect_size_zero_raises_error():
    metrics = [
        DesignSpecMetric(
            field_name="test_metric",
            metric_type=MetricType.BINARY,
            metric_baseline=0.5,
            metric_target=0.55,
            available_n=1000,
        ),
        DesignSpecMetric(
            field_name="bad_metric",
            metric_type=MetricType.BINARY,
            metric_baseline=0.5,
            metric_target=0.5,
            available_n=1000,
        ),
    ]

    with pytest.raises(StatsPowerError) as excinfo:
        check_power(metrics, n_arms=2)
    assert "bad_metric" in str(excinfo.value)


def test_check_missing_metric_type_raises_error():
    metrics = [
        DesignSpecMetric(
            field_name="bad_metric",
            metric_baseline=0.5,
            metric_target=0.5,
            available_n=1000,
        ),
    ]

    with pytest.raises(StatsPowerError) as excinfo:
        check_power(metrics, n_arms=2)
    assert "Unknown metric_type" in str(excinfo.value)
