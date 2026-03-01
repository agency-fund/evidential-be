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
    deff = calculate_design_effect(icc=0.0, avg_cluster_size=30)
    assert deff == 1.0


def test_calculate_design_effect_perfect_clustering():
    deff = calculate_design_effect(icc=1.0, avg_cluster_size=30)
    assert deff == 30.0


def test_calculate_design_effect_typical_school():
    deff = calculate_design_effect(icc=0.15, avg_cluster_size=30)
    assert deff == pytest.approx(5.35)


def test_calculate_design_effect_invalid_icc():
    with pytest.raises(ValueError, match="ICC must be"):
        calculate_design_effect(icc=1.5, avg_cluster_size=30)

    with pytest.raises(ValueError, match="ICC must be"):
        calculate_design_effect(icc=-0.1, avg_cluster_size=30)


def test_calculate_design_effect_invalid_cluster_size():
    with pytest.raises(ValueError, match="Cluster size must be"):
        calculate_design_effect(icc=0.15, avg_cluster_size=0)


def test_calculate_num_clusters_needed_typical_school():
    n_clusters = calculate_num_clusters_needed(n_individual=63, avg_cluster_size=30, deff=5.35)
    assert n_clusters == 12


def test_calculate_num_clusters_needed_rounds_up():
    n = calculate_num_clusters_needed(n_individual=60, avg_cluster_size=30, deff=5.0)
    assert n == 10

    n = calculate_num_clusters_needed(n_individual=60.03, avg_cluster_size=30, deff=5.0)
    assert n == 11


def test_calculate_num_clusters_needed_no_clustering():
    n = calculate_num_clusters_needed(n_individual=100, avg_cluster_size=25, deff=1.0)
    assert n == 4


def test_calculate_num_clusters_needed_high_deff():
    n_low = calculate_num_clusters_needed(n_individual=60, avg_cluster_size=30, deff=2.0)

    n_high = calculate_num_clusters_needed(n_individual=60, avg_cluster_size=30, deff=10.0)

    assert n_high > n_low
    assert n_low == 4
    assert n_high == 20


def test_analyze_metric_power_cluster_missing_baseline():

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

    assert result.target_n is None
    assert result.num_clusters_total is None
    assert result.clusters_per_arm is None
    assert result.n_per_arm is None
    assert result.design_effect is None
    assert result.effective_sample_size is None

    assert result.icc == 0.15
    assert result.avg_cluster_size == 30

    assert result.msg is not None


def test_analyze_metric_power_cluster_balanced():
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

    assert result.target_n is not None
    assert result.num_clusters_total == 24
    assert result.clusters_per_arm == [12, 12]
    assert result.n_per_arm == [360, 360]
    assert result.design_effect == pytest.approx(5.35)
    assert result.icc == 0.15
    assert result.avg_cluster_size == 30
    assert result.effective_sample_size == 134


def test_analyze_metric_power_cluster_no_clustering():
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
        icc=0.0,
        avg_cluster_size=30,
    )

    assert result.design_effect == 1.0

    assert result.num_clusters_total is not None
    assert result.num_clusters_total == 6

    assert result.effective_sample_size == result.target_n


def test_analyze_metric_power_cluster_unbalanced():
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
        arm_weights=[20, 80],
    )

    assert result.target_n is not None
    assert result.num_clusters_total is not None
    assert result.clusters_per_arm is not None
    assert result.n_per_arm is not None

    assert len(result.clusters_per_arm) == 2
    assert len(result.n_per_arm) == 2

    assert result.clusters_per_arm == [8, 29]
    assert result.n_per_arm == [240, 870]

    assert result.num_clusters_total == sum(result.clusters_per_arm)
    assert result.target_n == sum(result.n_per_arm)


def test_analyze_metric_power_cluster_three_arms():
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

    assert result.clusters_per_arm is not None
    assert result.n_per_arm is not None

    assert len(result.clusters_per_arm) == 3
    assert len(result.n_per_arm) == 3

    assert result.clusters_per_arm == [12, 12, 12]
    assert result.n_per_arm == [360, 360, 360]

    assert result.num_clusters_total == sum(result.clusters_per_arm)
    assert result.target_n == sum(result.n_per_arm)


def test_analyze_metric_power_cluster_binary_metric():
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

    assert result.target_n is not None
    assert result.num_clusters_total is not None
    assert result.design_effect == pytest.approx(3.45)  # 1 + (50-1)*0.05


def test_analyze_metric_power_cluster_high_icc():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_n=2000,
        available_nonnull_n=2000,
    )

    result_low = analyze_metric_power_cluster(
        metric=metric,
        n_arms=2,
        icc=0.05,
        avg_cluster_size=30,
    )

    result_high = analyze_metric_power_cluster(
        metric=metric,
        n_arms=2,
        icc=0.30,
        avg_cluster_size=30,
    )

    assert result_low.num_clusters_total is not None
    assert result_high.num_clusters_total is not None
    assert result_low.design_effect is not None
    assert result_high.design_effect is not None

    assert result_low.num_clusters_total == 12
    assert result_high.num_clusters_total == 42
    assert result_high.design_effect > result_low.design_effect


def test_calculate_design_effect_with_cv():
    """Test CV-adjusted design effect with specific values."""

    deff_no_cv = calculate_design_effect(icc=0.15, avg_cluster_size=30, cv=0.0)
    assert deff_no_cv == pytest.approx(5.35)  # 1 + (30-1)*0.15 = 5.35

    deff_cv = calculate_design_effect(icc=0.15, avg_cluster_size=30, cv=1.5)
    # DEFF = 1 + 0.15 * [(30-1) + 30*(1.5²)] = 15.475
    assert deff_cv == pytest.approx(15.475)

    assert deff_cv > deff_no_cv
    assert deff_cv / deff_no_cv == pytest.approx(2.89, rel=0.01)


def test_analyze_metric_power_cluster_cv_increases_clusters():
    """Test that higher CV requires more clusters."""
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_n=5000,
        available_nonnull_n=5000,
    )

    result_low_cv = analyze_metric_power_cluster(
        metric=metric,
        n_arms=2,
        icc=0.15,
        avg_cluster_size=30,
        cv=0.3,
    )

    result_high_cv = analyze_metric_power_cluster(
        metric=metric,
        n_arms=2,
        icc=0.15,
        avg_cluster_size=30,
        cv=2.0,
    )

    assert result_low_cv.num_clusters_total is not None
    assert result_high_cv.num_clusters_total is not None
    assert result_low_cv.design_effect is not None
    assert result_high_cv.design_effect is not None

    assert result_high_cv.num_clusters_total > result_low_cv.num_clusters_total
    assert result_high_cv.design_effect > result_low_cv.design_effect

    assert result_low_cv.design_effect == pytest.approx(5.755, rel=0.01)  # 1 + 0.15*[(29) + 30*0.09]
    assert result_high_cv.design_effect == pytest.approx(23.35, rel=0.01)  # 1 + 0.15*[(29) + 30*4]

    assert result_high_cv.num_clusters_total is not None
    assert result_low_cv.num_clusters_total is not None

    ratio = result_high_cv.num_clusters_total / result_low_cv.num_clusters_total
    assert ratio == pytest.approx(4.06, rel=0.1)


def test_analyze_metric_power_cluster_cv_affects_all_arms():
    """Test that CV adjustment applies to all arms in multi-arm design."""
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        available_n=5000,
        available_nonnull_n=5000,
    )

    result = analyze_metric_power_cluster(
        metric=metric,
        n_arms=3,
        icc=0.15,
        avg_cluster_size=30,
        cv=1.2,
    )

    assert result.clusters_per_arm is not None

    assert len(result.clusters_per_arm) == 3
    assert result.cv == 1.2
    assert result.design_effect == pytest.approx(11.86, rel=0.01)  # 1 + 0.15*[(29) + 30*1.44]


def test_analyze_metric_power_cluster_cv_warning_high():
    """Test that high CV (>1.0) adds warning to message."""
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
        n_arms=2,
        icc=0.15,
        avg_cluster_size=30,
        cv=1.5,  # High CV
    )

    # Check warning is in message
    assert result.msg is not None
    assert "Warning" in result.msg.msg
    assert "CV=1.5" in result.msg.msg or "CV=1.50" in result.msg.msg
    assert "cluster size variation" in result.msg.msg.lower()


def test_analyze_metric_power_cluster_cv_warning_low():
    """Test that low CV (<=1.0) does NOT add warning."""
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
        n_arms=2,
        icc=0.15,
        avg_cluster_size=30,
        cv=0.5,  # Low CV
    )

    # Check NO warning in message
    assert result.msg is not None
    assert "Warning" not in result.msg.msg
    assert "CV=" not in result.msg.msg


def test_analyze_metric_power_cluster_cv_zero_no_warning():
    """Test that CV=0 (equal clusters) has no warning."""
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
        n_arms=2,
        icc=0.15,
        avg_cluster_size=30,
        cv=0.0,  # Equal clusters
    )

    # Check NO warning
    assert result.msg is not None
    assert "Warning" not in result.msg.msg


def test_analyze_metric_power_cluster_cv_stored():
    """Test that CV value is stored in result."""
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
        n_arms=2,
        icc=0.15,
        avg_cluster_size=30,
        cv=1.8,
    )

    # CV should be stored
    assert result.cv == 1.8
