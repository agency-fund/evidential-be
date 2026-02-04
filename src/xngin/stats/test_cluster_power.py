"""Tests for cluster randomization power analysis."""

import pytest

from xngin.stats.cluster_power import (
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
