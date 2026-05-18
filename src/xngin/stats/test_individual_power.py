import pytest

from xngin.apiserver.routers.common_api_types import DesignSpecMetric, MetricPowerAnalysis
from xngin.apiserver.routers.common_enums import MetricType
from xngin.stats.individual_power import (
    solve_for_mde_individual_impl,
    solve_for_sample_size_individual,
)


def test_solve_for_sample_size_individual_has_no_cluster_fields():
    metric = DesignSpecMetric(
        field_name="test_score",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_n=1000,
        available_nonnull_n=1000,
    )

    result = solve_for_sample_size_individual(metric=metric, n_arms=2)

    assert isinstance(result, MetricPowerAnalysis)
    assert result.num_clusters_total is None


def test_solve_for_mde_individual_impl():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_nonnull_n=1000,
        available_n=1000,
    )

    target_n, pct_change = solve_for_mde_individual_impl(metric, desired_n=20000, n_arms=2)
    assert target_n == pytest.approx(100.792, rel=1e-3)
    assert pct_change == pytest.approx(0.00793, rel=1e-3)


def test_solve_for_mde_individual_impl_binary():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.BINARY,
        metric_baseline=0.5,
        metric_target=0.55,
        available_nonnull_n=1000,
        available_n=1000,
    )

    target_n, pct_change = solve_for_mde_individual_impl(metric, desired_n=20000, n_arms=2)
    assert target_n == pytest.approx(0.480, rel=1e-3)
    assert pct_change == pytest.approx(-0.0396, rel=1e-3)


def test_solve_for_mde_individual_impl_zero_n_raises_error():
    with pytest.raises(ValueError):
        solve_for_mde_individual_impl(
            DesignSpecMetric(
                field_name="test_metric",
                metric_type=MetricType.NUMERIC,
                metric_baseline=100,
                metric_target=110,
                metric_stddev=20,
                available_nonnull_n=1000,
                available_n=1000,
            ),
            desired_n=0,
            n_arms=2,
        )


def test_solve_for_mde_individual_impl_unbalanced_arms():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_nonnull_n=1000,
        available_n=1000,
    )

    target_n, pct_change = solve_for_mde_individual_impl(metric, desired_n=20000, n_arms=2, arm_weights=[20, 80])
    assert target_n == pytest.approx(100.991, rel=1e-3)
    assert pct_change == pytest.approx(0.00991, rel=1e-3)
