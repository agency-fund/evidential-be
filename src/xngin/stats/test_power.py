from xngin.stats.power import (
    DesignSpecMetric,
    MetricType,
    analyze_metric_power,
    check_power,
)


def test_analyze_metric_power_numeric():
    metric = DesignSpecMetric(
        metric_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_n=1000,
    )

    result = analyze_metric_power(metric, n_arms=2)

    assert result.metric_spec.metric_name == "test_metric"
    assert result.metric_spec.metric_type == MetricType.NUMERIC
    assert result.metric_spec.metric_baseline == 100
    assert result.metric_spec.metric_target == 110
    assert result.available_n == 1000
    assert result.target_n == 128.0
    assert result.sufficient_n
    assert result.msg is not None


def test_analyze_metric_power_binary():
    metric = DesignSpecMetric(
        metric_name="test_metric",
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
    assert not result.sufficient_n


def test_check_power_multiple_metrics():
    metrics = [
        DesignSpecMetric(
            metric_name="metric1",
            metric_type=MetricType.NUMERIC,
            metric_baseline=100,
            metric_target=110,
            metric_stddev=20,
            available_n=1000,
        ),
        DesignSpecMetric(
            metric_name="metric2",
            metric_type=MetricType.BINARY,
            metric_baseline=0.5,
            metric_target=0.55,
            available_n=1000,
        ),
    ]

    results = check_power(metrics, n_arms=2)

    assert len(results) == 2
    assert results[0].metric_spec.metric_name == "metric1"
    assert results[1].metric_spec.metric_name == "metric2"
