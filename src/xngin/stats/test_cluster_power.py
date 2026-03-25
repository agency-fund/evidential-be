"""Tests for cluster randomization power analysis, including using wide_dwh test data."""

import math
from pathlib import Path

import pandas as pd
import pytest

from xngin.apiserver.routers.common_api_types import DesignSpecMetric
from xngin.apiserver.routers.common_enums import MetricType
from xngin.stats.cluster_power import (
    analyze_metric_power_cluster,
    calculate_design_effect,
    calculate_effective_sample_size,
    calculate_mde_cluster,
    calculate_num_clusters_needed,
)
from xngin.stats.power import calculate_mde_with_desired_n


@pytest.fixture(scope="module")
def wide_dwh_data():
    """Load the wide_dwh test dataset."""
    data_path = Path(__file__).parent.parent / "apiserver" / "testdata" / "wide_dwh.csv.zst"
    return pd.read_csv(data_path)


class TestDataExploration:
    """Explore and verify the wide_dwh test data structure."""

    def test_cluster_structure(self, wide_dwh_data):
        """Verify cluster structure."""
        assert len(wide_dwh_data) == 1000
        assert "cluster_id" in wide_dwh_data.columns

        n_clusters = wide_dwh_data["cluster_id"].nunique()
        cluster_sizes = wide_dwh_data.groupby("cluster_id").size()

        assert n_clusters == 20
        assert cluster_sizes.min() > 0  # No empty clusters


class TestCalculateDesignEffect:
    def test_calculate_design_effect(self):
        """Test design effect calculation."""
        # ICC=0 → DEFF=1 (no clustering effect)
        assert calculate_design_effect(icc=0.0, avg_cluster_size=50) == 1.0

        # ICC=0.05, m=50 → DEFF = 1 + (50-1)*0.05 = 3.45
        deff = calculate_design_effect(icc=0.05, avg_cluster_size=50)
        assert deff == pytest.approx(3.45)

    def test_deff_world_bank_example(self):
        """
        Verify DEFF formula against World Bank blog example.

        Example values:
        - ICC = 0.39
        - Average cluster size = 6.0
        - CV = 5.16

        World Bank reports:
        - Standard design effect (DEFT) = 1.72 → DEFF = 2.95
        - With CV adjustment (DEFT) = 8.08 → DEFF = 65.25
        """
        icc = 0.39
        avg_cluster_size = 6.0
        cv = 5.16

        # Test standard DEFF (without CV)
        deff_standard = calculate_design_effect(icc, avg_cluster_size, cv=0.0)
        expected_deff_standard = 1 + (avg_cluster_size - 1) * icc  # 2.95
        assert deff_standard == pytest.approx(expected_deff_standard, abs=0.01)

        # World Bank reports DEFT = 1.72, which means DEFF = 1.72² = 2.9584
        deft_standard = math.sqrt(deff_standard)
        assert deft_standard == pytest.approx(1.72, abs=0.01)

        # Test DEFF with CV adjustment
        # World Bank reports DEFT = 8.08, which means DEFF = 8.08² = 65.2864
        deff_with_cv = calculate_design_effect(icc, avg_cluster_size, cv)
        assert deff_with_cv == pytest.approx(65.2864, abs=0.04)

        # Verify DEFT matches what World Bank reports
        deft_with_cv = math.sqrt(deff_with_cv)
        assert deft_with_cv == pytest.approx(8.08, abs=0.01)

    def test_world_bank_mde_calculation(self):
        """
        Replicate World Bank MDE calculation to verify DEFF formula.

        Their parameters:
        - N = 31,068 workers
        - 5,172 firms (clusters)
        - Average cluster size = 6.0
        - ICC = 0.39
        - CV = 5.16

        They report:
        - Standard MDE = 0.055 s.d. (with basic DEFF)
        - MDE with CV = 0.26 s.d. (from simulations)
        - Ratio = 4.73
        """
        # World Bank parameters
        icc = 0.39
        avg_cluster_size = 6.0
        cv = 5.16

        # Calculate DEFFs
        deff_standard = calculate_design_effect(icc, avg_cluster_size, cv=0.0)
        deff_with_cv = calculate_design_effect(icc, avg_cluster_size, cv)

        # Calculate DEFTs
        deft_standard = math.sqrt(deff_standard)
        deft_with_cv = math.sqrt(deff_with_cv)

        # MDE scales with DEFT (square root of DEFF)
        # So the ratio of MDEs should equal the ratio of DEFTs
        mde_ratio = 0.26 / 0.055  # From World Bank
        deft_ratio = deft_with_cv / deft_standard

        # The ratios should match
        assert deft_ratio == pytest.approx(mde_ratio, abs=0.05)

    @pytest.mark.parametrize(
        "_test_name, icc, avg_cluster_size, cv, expected",
        [
            ("______no_clustering", 0.0, 30, 0.0, 1.0),
            ("_perfect_clustering", 1.0, 30, 0.0, 30.0),
            # DEFF = 1 + (30-1)*0.15 = 5.35
            ("_fake_school_no_cv", 0.15, 30, 0.0, pytest.approx(5.35)),
            # DEFF = 1 + 0.15 * [(30-1) + 30*(1.5²)] = 15.475
            ("fake_school_cv_1.5", 0.15, 30, 1.5, pytest.approx(15.475)),
        ],
    )
    def test_calculate_design_effect_parametrized(self, _test_name, icc, avg_cluster_size, cv, expected):
        if cv == 0.0:
            deff = calculate_design_effect(icc=icc, avg_cluster_size=avg_cluster_size)
        else:
            deff = calculate_design_effect(icc=icc, avg_cluster_size=avg_cluster_size, cv=cv)
        assert deff == expected

    @pytest.mark.parametrize(
        "_test_name, icc, avg_cluster_size, cv, expected_error",
        [
            ("invalid_icc", 1.5, 30, 0.0, "ICC must be between 0 and 1, got 1.5"),
            ("invalid_cluster_size", 0.15, 0, 0.0, "Cluster size must be >= 1, got 0"),
            ("invalid_cv", 0.15, 30, -1.0, "CV must be >= 0, got -1.0"),
        ],
    )
    def test_calculate_design_effect_invalid_params(self, _test_name, icc, avg_cluster_size, cv, expected_error):
        with pytest.raises(ValueError, match=expected_error):
            calculate_design_effect(icc=icc, avg_cluster_size=avg_cluster_size, cv=cv)


def test_calculate_effective_sample_size():
    """Test effective sample size calculation."""
    # 1000 participants / DEFF of 2 = 500 effective
    effective_n = calculate_effective_sample_size(total_n=1000, deff=2.0)
    assert effective_n == 500

    effective_n = calculate_effective_sample_size(total_n=720, deff=5.35)
    assert effective_n == 134

    effective_n = calculate_effective_sample_size(total_n=1000, deff=1.0)
    assert effective_n == 1000


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
    assert n_low == 4
    assert n_high == 20


def test_calculate_mde_cluster_basic():
    metric = DesignSpecMetric(
        field_name="reading_score",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_stddev=20,
    )

    target, pct_change = calculate_mde_cluster(
        available_n=600,
        metric=metric,
        n_arms=2,
        icc=0.15,
        avg_cluster_size=30,
    )

    assert target == pytest.approx(110.68, rel=0.01)
    assert pct_change == pytest.approx(0.1068, rel=0.01)


def test_calculate_mde_cluster_no_clustering():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_stddev=20,
    )

    ind_target, ind_pct = calculate_mde_with_desired_n(
        desired_n=1000,
        metric=metric,
        n_arms=2,
    )

    clust_target, clust_pct = calculate_mde_cluster(
        available_n=1000,
        metric=metric,
        n_arms=2,
        icc=0.0,
        avg_cluster_size=30,
    )

    assert clust_target == pytest.approx(ind_target)
    assert clust_pct == pytest.approx(ind_pct)


def test_calculate_mde_cluster_higher_with_clustering():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_stddev=20,
    )

    ind_target, _ = calculate_mde_with_desired_n(
        desired_n=600,
        metric=metric,
        n_arms=2,
    )

    clust_target, _ = calculate_mde_cluster(
        available_n=600,
        metric=metric,
        n_arms=2,
        icc=0.15,
        avg_cluster_size=30,
    )

    assert ind_target == pytest.approx(104.6, abs=0.1)
    assert clust_target == pytest.approx(110.7, abs=0.1)


def test_calculate_mde_cluster_binary_metric():
    metric = DesignSpecMetric(
        field_name="conversion",
        metric_type=MetricType.BINARY,
        metric_baseline=0.10,
    )

    target, pct_change = calculate_mde_cluster(
        available_n=1000,
        metric=metric,
        n_arms=2,
        icc=0.05,
        avg_cluster_size=50,
    )

    assert target == pytest.approx(0.0243, rel=0.01)
    assert pct_change == pytest.approx(-0.7566, rel=0.01)


def test_calculate_mde_cluster_unbalanced():
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_stddev=20,
    )

    target, pct_change = calculate_mde_cluster(
        available_n=1000,
        metric=metric,
        n_arms=2,
        icc=0.15,
        avg_cluster_size=30,
        arm_weights=[20, 80],
    )

    assert target == pytest.approx(110.35, rel=0.01)
    assert pct_change == pytest.approx(0.1035, rel=0.01)


def test_calculate_mde_cluster_with_cv():
    """Test that CV increases MDE (makes detection harder)."""
    metric = DesignSpecMetric(
        field_name="test_metric",
        metric_type=MetricType.NUMERIC,
        metric_baseline=100,
        metric_stddev=20,
    )

    target_no_cv, pct_no_cv = calculate_mde_cluster(
        available_n=720,
        metric=metric,
        n_arms=2,
        icc=0.15,
        avg_cluster_size=30,
        cv=0.0,
    )

    target_high_cv, pct_high_cv = calculate_mde_cluster(
        available_n=720,
        metric=metric,
        n_arms=2,
        icc=0.15,
        avg_cluster_size=30,
        cv=0.5,
    )

    assert target_high_cv > target_no_cv
    assert pct_high_cv > pct_no_cv


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
    assert result.num_clusters_total == 6
    assert result.clusters_per_arm == [3, 3]
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

    assert result.clusters_per_arm == [48, 48]
    assert result.n_per_arm == [2400, 2400]
    assert result.num_clusters_total == sum(result.clusters_per_arm)
    assert result.target_n == sum(result.n_per_arm)
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

    assert result_low.num_clusters_total == 12
    assert result_high.num_clusters_total == 42
    assert result_low.design_effect is not None
    assert result_high.design_effect is not None
    assert result_high.design_effect > result_low.design_effect


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

    assert result_low_cv.design_effect is not None
    assert result_high_cv.design_effect is not None
    assert result_high_cv.design_effect > result_low_cv.design_effect
    assert result_low_cv.design_effect == pytest.approx(5.755, rel=0.01)  # 1 + 0.15*[(29) + 30*0.09]
    assert result_high_cv.design_effect == pytest.approx(23.35, rel=0.01)  # 1 + 0.15*[(29) + 30*4]

    assert result_low_cv.num_clusters_total is not None
    assert result_high_cv.num_clusters_total is not None
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


@pytest.mark.parametrize(
    "cv,expected_warning",
    [(0, None), (1.0, None), (1.5, "Warning: High cluster size variation (CV=1.50)")],
)
def test_analyze_metric_power_cluster_cv_warning_message(cv: float, expected_warning: str | None):
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
        cv=cv,
    )

    # CV should be stored
    assert result.cv == cv

    assert result.msg is not None
    if expected_warning is not None:
        assert expected_warning in result.msg.msg
    else:
        assert "Warning:" not in result.msg.msg
