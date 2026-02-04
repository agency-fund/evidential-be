"""Tests for cluster randomization power analysis."""

import pytest

from xngin.apiserver.routers.common_api_types import DesignSpecMetric
from xngin.apiserver.routers.common_enums import MetricType
from xngin.stats.cluster_power import (
    analyze_metric_power_cluster,
    calculate_design_effect,
    calculate_num_clusters_needed,
)


def test_calculate_design_effect_no_clustering():
    """When ICC=0, there is no clustering effect (DEFF=1)."""
    deff = calculate_design_effect(icc=0.0, avg_cluster_size=30)
    assert deff == 1.0


def test_calculate_design_effect_perfect_clustering():
    """When ICC=1, DEFF equals cluster size."""
    deff = calculate_design_effect(icc=1.0, avg_cluster_size=30)
    assert deff == 30.0


def test_calculate_design_effect_typical_school():
    """Typical school scenario: ICC=0.15, m=30."""
    deff = calculate_design_effect(icc=0.15, avg_cluster_size=30)
    assert deff == pytest.approx(5.35)


def test_calculate_design_effect_invalid_icc():
    """ICC must be between 0 and 1."""
    with pytest.raises(ValueError, match="ICC must be"):
        calculate_design_effect(icc=1.5, avg_cluster_size=30)

    with pytest.raises(ValueError, match="ICC must be"):
        calculate_design_effect(icc=-0.1, avg_cluster_size=30)


def test_calculate_design_effect_invalid_cluster_size():
    """Cluster size must be >= 1."""
    with pytest.raises(ValueError, match="Cluster size must be"):
        calculate_design_effect(icc=0.15, avg_cluster_size=0)


def test_calculate_num_clusters_needed_typical_school():
    """Example: Need 63 individuals, clusters of 30, DEFF=5.35."""
    n_clusters = calculate_num_clusters_needed(n_individual=63, avg_cluster_size=30, deff=5.35)
    assert n_clusters == 12


def test_calculate_num_clusters_needed_rounds_up():
    """Always round up to ensure adequate power."""
    n = calculate_num_clusters_needed(n_individual=60, avg_cluster_size=30, deff=5.0)
    assert n == 10

    n = calculate_num_clusters_needed(n_individual=60.03, avg_cluster_size=30, deff=5.0)
    assert n == 11


def test_calculate_num_clusters_needed_no_clustering():
    """With DEFF=1, clusters equal individuals divided by cluster size."""
    n = calculate_num_clusters_needed(n_individual=100, avg_cluster_size=25, deff=1.0)
    assert n == 4


def test_calculate_num_clusters_needed_high_deff():
    """Higher DEFF means more clusters needed."""
    n_low = calculate_num_clusters_needed(n_individual=60, avg_cluster_size=30, deff=2.0)

    n_high = calculate_num_clusters_needed(n_individual=60, avg_cluster_size=30, deff=10.0)

    assert n_high > n_low
    assert n_low == 4
    assert n_high == 20


def test_analyze_metric_power_cluster_missing_baseline():
    """When baseline is missing, should return analysis with None cluster fields."""

    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=None,  # Missing!
        metric_target=110,
        metric_stddev=20,
        available_n=1000,
        available_nonnull_n=1000,
    )

    result = analyze_metric_power_cluster(
        metric=metric,
        n_arms=2,
        icc=0.15,
        avg_cluster_size=30,
    )

    # Should return error state
    assert result.target_n is None
    assert result.num_clusters_total is None
    assert result.clusters_per_arm is None
    assert result.n_per_arm is None
    assert result.design_effect is None
    assert result.effective_sample_size is None

    # But ICC and cluster size should be preserved
    assert result.icc == 0.15
    assert result.avg_cluster_size == 30

    # Should have an error message
    assert result.msg is not None


def test_analyze_metric_power_cluster_balanced():
    """Test balanced 2-arm cluster design."""
    metric = DesignSpecMetric(
        field_name="reading_score",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_n=1000,
        available_nonnull_n=1000,
    )

    result = analyze_metric_power_cluster(
        metric=metric,
        n_arms=2,
        icc=0.15,
        avg_cluster_size=30,
    )

    # Should succeed
    assert result.target_n is not None
    assert result.num_clusters_total == 24
    assert result.clusters_per_arm == [12, 12]
    assert result.n_per_arm == [360, 360]
    assert result.design_effect == pytest.approx(5.35)
    assert result.icc == 0.15
    assert result.avg_cluster_size == 30
    assert result.effective_sample_size == 134


def test_analyze_metric_power_cluster_no_clustering():
    """With ICC=0, should have DEFF=1 and minimal inflation."""
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_n=1000,
        available_nonnull_n=1000,
    )

    result = analyze_metric_power_cluster(
        metric=metric,
        n_arms=2,
        icc=0.0,  # No clustering
        avg_cluster_size=30,
    )

    # With ICC=0, DEFF should be 1
    assert result.design_effect == 1.0

    # Should need minimal clusters
    assert result.num_clusters_total is not None
    assert result.num_clusters_total < 10  # Much fewer than with clustering

    # Effective N should equal target N
    assert result.effective_sample_size == result.target_n


def test_analyze_metric_power_cluster_unbalanced():
    """Test unbalanced allocation (20% control, 80% treatment)."""
    metric = DesignSpecMetric(
        field_name="conversion",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_n=2000,
        available_nonnull_n=2000,
    )

    result = analyze_metric_power_cluster(
        metric=metric,
        n_arms=2,
        icc=0.15,
        avg_cluster_size=30,
        arm_weights=[20, 80],  # 20% control, 80% treatment
    )

    # Should succeed
    assert result.target_n is not None
    assert result.num_clusters_total is not None
    assert result.clusters_per_arm is not None
    assert result.n_per_arm is not None

    # Should have 2 arms
    assert len(result.clusters_per_arm) == 2
    assert len(result.n_per_arm) == 2

    # Treatment arm should have more clusters than control
    assert result.clusters_per_arm[1] > result.clusters_per_arm[0]
    assert result.n_per_arm[1] > result.n_per_arm[0]

    # Total should sum correctly
    assert result.num_clusters_total == sum(result.clusters_per_arm)
    assert result.target_n == sum(result.n_per_arm)


def test_analyze_metric_power_cluster_three_arms():
    """Test 3-arm design."""
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_n=2000,
        available_nonnull_n=2000,
    )

    result = analyze_metric_power_cluster(
        metric=metric,
        n_arms=3,
        icc=0.15,
        avg_cluster_size=30,
    )

    # Add assertions that fields are not None
    assert result.clusters_per_arm is not None
    assert result.n_per_arm is not None

    # Should have 3 arms
    assert len(result.clusters_per_arm) == 3
    assert len(result.n_per_arm) == 3

    # All arms should be equal (balanced)
    assert result.clusters_per_arm[0] == result.clusters_per_arm[1]
    assert result.clusters_per_arm[1] == result.clusters_per_arm[2]

    # Totals should sum correctly
    assert result.num_clusters_total == sum(result.clusters_per_arm)
    assert result.target_n == sum(result.n_per_arm)


def test_analyze_metric_power_cluster_binary_metric():
    """Test with binary metric."""
    metric = DesignSpecMetric(
        field_name="conversion",
        metric_type=MetricType.BINARY,
        metric_baseline=0.10,
        metric_target=0.15,
        available_n=5000,
        available_nonnull_n=5000,
    )

    result = analyze_metric_power_cluster(
        metric=metric,
        n_arms=2,
        icc=0.05,
        avg_cluster_size=50,
    )

    # Should succeed
    assert result.target_n is not None
    assert result.num_clusters_total is not None
    assert result.design_effect == pytest.approx(3.45)  # 1 + (50-1)*0.05


def test_analyze_metric_power_cluster_high_icc():
    """Higher ICC should require more clusters."""
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_n=2000,
        available_nonnull_n=2000,
    )

    # Low ICC
    result_low = analyze_metric_power_cluster(
        metric=metric,
        n_arms=2,
        icc=0.05,
        avg_cluster_size=30,
    )

    # High ICC
    result_high = analyze_metric_power_cluster(
        metric=metric,
        n_arms=2,
        icc=0.30,
        avg_cluster_size=30,
    )
    # Add assertions before comparisons
    assert result_low.num_clusters_total is not None
    assert result_high.num_clusters_total is not None
    assert result_low.design_effect is not None
    assert result_high.design_effect is not None

    # Higher ICC should need more clusters
    assert result_high.num_clusters_total > result_low.num_clusters_total
    assert result_high.design_effect > result_low.design_effect
