"""Tests for cluster randomization power analysis - MDE calculation."""

import pytest

from xngin.apiserver.routers.common_api_types import DesignSpecMetric
from xngin.apiserver.routers.common_enums import MetricType
from xngin.stats.cluster_power import (
    calculate_design_effect,
    calculate_effective_sample_size,
    calculate_mde_cluster,
)
from xngin.stats.power import calculate_mde_with_desired_n


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


def test_calculate_effective_sample_size():
    effective_n = calculate_effective_sample_size(total_n=720, deff=5.35)
    assert effective_n == 134

    effective_n = calculate_effective_sample_size(total_n=1000, deff=1.0)
    assert effective_n == 1000


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

    assert target > 100
    assert pct_change > 0


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

    assert clust_target > ind_target


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

    assert 0 < target < 1
    assert target != 0.10
    assert pct_change != 0


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

    assert target > 100
    assert pct_change > 0
