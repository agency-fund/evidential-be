import pytest

from xngin.apiserver.routers.common_api_types import DesignSpecMetric
from xngin.apiserver.routers.common_enums import (
    MetricPowerAnalysisMessageType,
    MetricType,
)
from xngin.stats.power import analyze_metric_power, calculate_mde_with_desired_n, check_power
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


@pytest.mark.parametrize(
    "available_nonnull_n,available_n,target_possible",
    [
        (789, 789, 23.8468),
        (104, 789, 42.9997),
        # This last test would pass if we compared against available_n instead of nonulls.
        (789, 100000, 23.8468),
    ],
)
def test_analyze_metric_power_numeric_insufficient(available_nonnull_n, available_n, target_possible):
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_pct_change=0.1,
        metric_type=MetricType.NUMERIC,
        metric_baseline=13.206590621039297,
        metric_target=14.527249683143227,
        metric_stddev=37.601056495700554,
        available_nonnull_n=available_nonnull_n,
        available_n=available_n,
    )

    result = analyze_metric_power(metric, n_arms=4, power=0.8, alpha=0.05)

    assert result.metric_spec.field_name == "test_metric"
    assert result.metric_spec.metric_type == MetricType.NUMERIC
    assert result.metric_spec.metric_baseline == pytest.approx(13.2065, rel=1e-4)
    assert result.metric_spec.metric_target == pytest.approx(14.5273, rel=1e-4)
    assert result.metric_spec.available_nonnull_n == available_nonnull_n
    assert result.metric_spec.available_n == available_n
    assert result.target_n == 50904
    assert not result.sufficient_n
    assert result.msg is not None
    assert result.msg.type == MetricPowerAnalysisMessageType.INSUFFICIENT
    assert result.msg.values == {
        "available_n": available_n,
        "target_n": 50904,
        "available_nonnull_n": available_nonnull_n,
        "additional_n_needed": 50904 - available_nonnull_n,
        "metric_baseline": 13.2066,
        "target_possible": target_possible,
        "metric_target": 14.5272,
    }

    # Check that null warning appears when there are nulls
    if available_nonnull_n != available_n:
        assert "WARNING" in result.msg.msg
    else:
        assert "WARNING" not in result.msg.msg

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


def test_analyze_metric_with_zero_available_nonnull_n_returns_insufficient():
    metric = DesignSpecMetric(
        field_name="no_available_nonnull_n",
        metric_type=MetricType.BINARY,
        metric_baseline=0.5,
        metric_target=0.6,
        available_n=1000,
        available_nonnull_n=0,
    )

    result = analyze_metric_power(metric, n_arms=2)

    assert result.msg is not None
    assert result.msg.type == MetricPowerAnalysisMessageType.INSUFFICIENT
    assert "You have no units with non-null values" in result.msg.msg
    # When returning early error, values is None
    assert result.msg.values is None
    assert result.target_n is None
    assert result.sufficient_n is None


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


def test_analyze_metric_power_unbalanced_two_arms():
    """Test power calculation for unbalanced arms (20-80 split).

    (c.f. test_analyze_metric_power_numeric for balanced case)
    """
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_nonnull_n=1000,
        available_n=1000,
    )

    # 20-80 split: control gets 20%, treatment gets 80%
    result = analyze_metric_power(metric, n_arms=2, arm_weights=[20.0, 80.0])

    assert result.metric_spec.field_name == "test_metric"
    assert result.target_n is not None
    # Unbalanced design (ratio=80/20) requires more total participants than the balanced case above.
    assert result.target_n == 200
    assert result.sufficient_n
    assert result.msg is not None
    assert result.msg.type == MetricPowerAnalysisMessageType.SUFFICIENT


def test_analyze_metric_power_unbalanced_three_arms():
    """Test power calculation with unbalanced three arms (20-20-60 split)"""
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_nonnull_n=1000,
        available_n=1000,
    )

    result = analyze_metric_power(metric, n_arms=3, arm_weights=[20.0, 20.0, 60.0])

    assert result.metric_spec.field_name == "test_metric"
    # Using smallest arm (20%) for conservative estimate: ratio=1 (20/20)
    # This requires more samples than using the largest arm (old buggy behavior)
    assert result.target_n == 320
    assert result.sufficient_n
    assert result.msg is not None
    assert result.msg.type == MetricPowerAnalysisMessageType.SUFFICIENT


def test_analyze_metric_power_unbalanced_binary():
    """Test power calculation for unbalanced arms for binary metric.

    (c.f. test_analyze_metric_power_binary for balanced case)
    """
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.BINARY,
        metric_baseline=0.5,
        metric_target=0.55,
        available_nonnull_n=5000,
        available_n=5000,
    )

    result = analyze_metric_power(metric, n_arms=2, arm_weights=[33.3, 66.7])

    assert result.metric_spec.metric_type == MetricType.BINARY
    assert result.target_n is not None
    # Unbalanced requires more than the balanced case above.
    assert result.target_n == 3526
    assert result.sufficient_n
    assert result.msg is not None
    assert result.msg.type == MetricPowerAnalysisMessageType.SUFFICIENT


def test_check_power_unbalanced():
    """Test check_power with unbalanced arms"""
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
            available_nonnull_n=5000,
            available_n=5000,
        ),
    ]

    results = check_power(metrics, n_arms=2, arm_weights=[20, 80])

    assert len(results) == 2
    assert results[0].metric_spec.field_name == "metric1"
    assert results[1].metric_spec.field_name == "metric2"
    # Both should have sufficient power
    assert results[0].sufficient_n
    assert results[1].sufficient_n
    # Same as test_analyze_metric_power_unbalanced_two_arms since it's the same params.
    assert results[0].target_n == 200
    # Even larger than test_analyze_metric_power_unbalanced_binary since the ratio is also larger.
    assert results[1].target_n == 4895


def test_calculate_mde_with_desired_n():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_nonnull_n=1000,
        available_n=1000,
    )

    target_n, pct_change = calculate_mde_with_desired_n(20000, metric, n_arms=2)
    assert target_n == pytest.approx(100.792, rel=1e-3)
    assert pct_change == pytest.approx(0.00793, rel=1e-3)


def test_calculate_mde_with_desired_n_binary():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.BINARY,
        metric_baseline=0.5,
        metric_target=0.55,
        available_nonnull_n=1000,
        available_n=1000,
    )

    target_n, pct_change = calculate_mde_with_desired_n(20000, metric, n_arms=2)
    assert target_n == pytest.approx(0.480, rel=1e-3)
    assert pct_change == pytest.approx(-0.0396, rel=1e-3)


def test_calculate_mde_with_desired_n_zero_n_raises_error():
    with pytest.raises(ValueError):
        calculate_mde_with_desired_n(
            0,
            DesignSpecMetric(
                field_name="test_metric",
                metric_type=MetricType.NUMERIC,
                metric_baseline=100,
                metric_target=110,
                metric_stddev=20,
                available_nonnull_n=1000,
                available_n=1000,
            ),
            n_arms=2,
        )


def test_calculate_mde_with_unbalanced_arms():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_nonnull_n=1000,
        available_n=1000,
    )

    target_n, pct_change = calculate_mde_with_desired_n(20000, metric, n_arms=2, arm_weights=[20, 80])
    assert target_n == pytest.approx(100.991, rel=1e-3)
    assert pct_change == pytest.approx(0.00991, rel=1e-3)


def test_analyze_metric_power_numeric_with_desired_n():
    """Test MDE calculation when desired_n is specified for numeric metric."""
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100.0,
        metric_stddev=20.0,
        available_n=10000,
        available_nonnull_n=10000,
    )

    result = analyze_metric_power(metric, n_arms=2, desired_n=500)

    # Should return MDE results
    assert result.target_n == 500
    assert result.target_possible == pytest.approx(105.0213)
    assert result.pct_change_possible == pytest.approx(0.0502, abs=1e-4)
    assert result.sufficient_n is None  # Not applicable in MDE mode

    # Message should mention MDE
    assert result.msg is not None
    assert "minimum detectable effect" in result.msg.msg.lower()


def test_analyze_metric_power_binary_with_desired_n():
    """Test MDE calculation when desired_n is specified for binary metric."""
    metric = DesignSpecMetric(
        field_name="conversion_rate",
        metric_type=MetricType.BINARY,
        metric_baseline=0.05,  # 5% conversion
        available_n=10000,
        available_nonnull_n=10000,
    )

    result = analyze_metric_power(metric, n_arms=2, desired_n=1000)

    # Should return MDE results
    assert result.target_n == 1000
    assert result.target_possible == pytest.approx(0.0186, abs=1e-4)
    # re: % change = (target possible / baseline) - 1
    assert result.pct_change_possible == pytest.approx(-0.6274, abs=1e-4)
    assert result.sufficient_n is None

    # Message should mention MDE
    assert result.msg is not None
    assert "minimum detectable effect" in result.msg.msg.lower()


def test_check_power_with_desired_n():
    """Test check_power with desired_n for multiple metrics."""
    metrics = [
        DesignSpecMetric(
            field_name="metric1",
            metric_type=MetricType.NUMERIC,
            metric_baseline=100.0,
            metric_stddev=20.0,
            available_n=10000,
            available_nonnull_n=10000,
        ),
        DesignSpecMetric(
            field_name="metric2",
            metric_type=MetricType.BINARY,
            metric_baseline=0.05,
            available_n=10000,
            available_nonnull_n=10000,
        ),
    ]

    results = check_power(metrics, n_arms=2, desired_n=500)

    # Should get results for both metrics
    assert len(results) == 2

    # Both should have MDE calculated
    for result in results:
        assert result.target_n == 500
        assert result.sufficient_n is None

    assert results[0].target_possible == pytest.approx(105.0213)
    assert results[0].pct_change_possible == pytest.approx(0.0502, abs=1e-4)
    # standardized effect size (Cohen's h) = 0.2505810918259752
    assert results[1].target_possible == pytest.approx(0.0100, abs=1e-4)
    assert results[1].pct_change_possible == pytest.approx(-0.7998, abs=1e-4)


def test_analyze_metric_power_without_desired_n_still_works():
    """Test that original behavior still works when desired_n not provided."""
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100.0,
        metric_target=110.0,
        metric_stddev=20.0,
        available_n=10000,
        available_nonnull_n=10000,
    )

    # Call WITHOUT desired_n (original behavior)
    result = analyze_metric_power(metric, n_arms=2)

    # Should calculate required sample size (not MDE)
    assert result.target_n is not None
    assert result.target_n > 0
    assert result.sufficient_n is True
    assert result.target_possible is None


def test_analyze_metric_power_desired_n_missing_baseline():
    """Test that error is returned when baseline is missing in MDE mode."""
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=None,
        metric_stddev=20.0,
    )

    result = analyze_metric_power(metric, n_arms=2, desired_n=500)

    # Should return error
    assert result.msg is not None
    assert result.msg.type == MetricPowerAnalysisMessageType.NO_BASELINE


def test_analyze_metric_power_desired_n_zero_stddev():
    """Test that error is returned when stddev is zero for numeric metric in MDE mode."""
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100.0,
        metric_stddev=0.0,
    )

    result = analyze_metric_power(metric, n_arms=2, desired_n=500)

    # Should return error
    assert result.msg is not None
    assert result.msg.type == MetricPowerAnalysisMessageType.ZERO_STDDEV


def test_analyze_metric_power_desired_n_with_unbalanced_arms():
    """Test MDE calculation with unbalanced arm allocation."""
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.BINARY,
        metric_baseline=0.05,
        available_n=10000,
        available_nonnull_n=10000,
    )

    # 20% control, 80% treatment
    result = analyze_metric_power(metric, n_arms=2, arm_weights=[20.0, 80.0], desired_n=1000)

    # Should still work with unbalanced allocation
    assert result.target_n == 1000
    assert metric.metric_baseline is not None
    assert result.target_possible is not None
    assert result.target_possible < metric.metric_baseline
