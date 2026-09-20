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

    result = solve_for_mde_individual_impl(metric, desired_n=20000, n_arms=2)
    assert result.target_possible == pytest.approx(100.792, rel=1e-3)
    assert result.pct_change_possible == pytest.approx(0.00793, rel=1e-3)
    # Numeric bounds are symmetric around the baseline.
    assert result.target_possible_lower == pytest.approx(99.208, rel=1e-3)
    assert result.pct_change_possible_lower == pytest.approx(-0.00793, rel=1e-3)


def test_solve_for_mde_individual_impl_binary():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.BINARY,
        metric_baseline=0.5,
        metric_target=0.55,
        available_nonnull_n=1000,
        available_n=1000,
    )

    result = solve_for_mde_individual_impl(metric, desired_n=20000, n_arms=2)
    # The primary bound is the positive lift; the _lower fields hold the detectable decrease
    # (symmetric here because the baseline is 0.5).
    assert result.target_possible == pytest.approx(0.5198, rel=1e-3)
    assert result.pct_change_possible == pytest.approx(0.0396, rel=1e-3)
    assert result.target_possible_lower == pytest.approx(0.480, rel=1e-3)
    assert result.pct_change_possible_lower == pytest.approx(-0.0396, rel=1e-3)


def test_solve_for_mde_individual_impl_binary_asymmetric_bounds():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.BINARY,
        metric_baseline=0.05,
        available_nonnull_n=1000,
        available_n=1000,
    )

    result = solve_for_mde_individual_impl(metric, desired_n=1000, n_arms=2)
    # The MDE is symmetric in Cohen's h space but not in probability space, so the
    # detectable lift is larger in magnitude than the detectable decrease.
    assert result.target_possible - 0.05 == pytest.approx(0.0455, abs=1e-4)
    assert 0.05 - result.target_possible_lower == pytest.approx(0.0314, abs=1e-4)


def test_solve_for_mde_individual_impl_binary_clamps_to_valid_proportions():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.BINARY,
        metric_baseline=0.95,
        available_nonnull_n=40,
        available_n=40,
    )

    # With such a small sample, the needed Cohen's h exceeds what any proportion <= 1.0
    # can produce in the positive direction, so the bound is clamped to 1.0.
    result = solve_for_mde_individual_impl(metric, desired_n=40, n_arms=2)
    assert result.target_possible == 1.0
    assert 0.0 <= result.target_possible_lower < 0.95


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

    result = solve_for_mde_individual_impl(metric, desired_n=20000, n_arms=2, arm_weights=[20, 80])
    assert result.target_possible == pytest.approx(100.991, rel=1e-3)
    assert result.pct_change_possible == pytest.approx(0.00991, rel=1e-3)
