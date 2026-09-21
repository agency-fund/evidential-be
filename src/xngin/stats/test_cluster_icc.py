"""Tests for ICC calculations from dataframes."""

import numpy as np
import pandas as pd
import pytest

from xngin.stats.cluster_icc import calculate_icc_from_dataframe, calculate_icc_from_sufficient_stats


class TestICCFromDataFrame:
    """Test ICC calculation from DataFrame (no database required)."""

    def test_calculate_icc_perfect_clustering(self):
        """Test ICC when all variance is between clusters (ICC = 1)."""
        df = pd.DataFrame({
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "y": [10, 10, 10, 20, 20, 20, 30, 30, 30],
        })
        icc = calculate_icc_from_dataframe(df, cluster_column="id", outcome_column="y")
        assert icc == pytest.approx(1.0)

    def test_calculate_icc_no_clustering(self):
        """Test ICC when all variance is within clusters (ICC = 0)."""
        df = pd.DataFrame({
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "y": [10, 20, 30, 10, 20, 30, 10, 20, 30],
        })
        icc = calculate_icc_from_dataframe(df, cluster_column="id", outcome_column="y")
        assert icc == pytest.approx(0.0, abs=0.01)

    def test_calculate_icc_moderate_clustering(self):
        """Test ICC with moderate clustering (ICC < 0.1)."""
        num_clusters = 100
        rows_per_cluster = 10

        cluster_ids = np.repeat(np.arange(num_clusters), rows_per_cluster)
        # generate independent cluster means
        rng = np.random.default_rng(42)
        cluster_means = rng.normal(loc=100, scale=10, size=num_clusters)
        y = np.repeat(cluster_means, rows_per_cluster) + rng.normal(0, 30, size=num_clusters * rows_per_cluster)

        df = pd.DataFrame({"id": cluster_ids, "y": y})
        icc = calculate_icc_from_dataframe(df, cluster_column="id", outcome_column="y")
        assert icc == pytest.approx(0.06, abs=0.01)

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        df = pd.DataFrame({"cluster_id": [], "outcome": []})

        with pytest.raises(ValueError, match="Cannot calculate ICC from empty dataframe"):
            calculate_icc_from_dataframe(df, cluster_column="cluster_id", outcome_column="outcome")

    def test_single_cluster_raises_error(self):
        """Test that single cluster raises ValueError."""
        df = pd.DataFrame({
            "cluster_id": [1, 1, 1],
            "outcome": [10, 20, 30],
        })

        with pytest.raises(ValueError, match="Need at least 2 clusters"):
            calculate_icc_from_dataframe(df, cluster_column="cluster_id", outcome_column="outcome")

    def test_custom_column_names(self):
        """ICC works when cluster/outcome columns are named explicitly, and a string cluster_column is OK."""
        df = pd.DataFrame({
            "school": ["1", "1", "1", "2", "2", "2", "3", "3", "3"],
            "score": [10, 10, 10, 20, 20, 20, 30, 30, 30],
        })
        icc = calculate_icc_from_dataframe(df, cluster_column="school", outcome_column="score")
        assert icc == pytest.approx(1.0, abs=0.01)

    def test_missing_custom_column_raises(self):
        with pytest.raises(ValueError, match="DataFrame is missing columns"):
            calculate_icc_from_dataframe(
                pd.DataFrame({"a": [1], "b": [2]}),
                cluster_column="school",
                outcome_column="score",
            )

    def test_nan_raises(self):
        """Test that NaN values in cluster or outcome columns raise ValueError."""
        df = pd.DataFrame({
            "id": [1, 1, 2, 2],
            "y": [1.0, float("nan"), 3.0, 4.0],
        })
        with pytest.raises(ValueError, match="outcome column 'y' contains NaN values"):
            calculate_icc_from_dataframe(df, cluster_column="id", outcome_column="y")

        df = pd.DataFrame({
            "id": [1, 1, None, 2],
            "y": [1.0, 2.0, 3.0, 4.0],
        })
        with pytest.raises(ValueError, match="cluster column 'id' contains NaN values"):
            calculate_icc_from_dataframe(df, cluster_column="id", outcome_column="y")

    def test_calculate_icc_unbalanced_cluster_sizes(self):
        """ICC stays in [0, 1] with unequal cluster sizes."""
        df = pd.DataFrame({
            "id": [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
            "_y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        icc = calculate_icc_from_dataframe(df, cluster_column="id", outcome_column="_y")
        assert 0.0 <= icc <= 1.0
        assert icc == pytest.approx(0.8571, abs=1e-4)


def _sufficient_stats(
    cluster_ids: np.ndarray, y: np.ndarray, shift: float = 0.0
) -> tuple[list[float], list[float], list[float]]:
    """Per-cluster (count, sum, sumsq) triples, as get_cluster_sufficient_stats returns them."""
    shifted = y - shift
    counts, sums, sumsqs = [], [], []
    for cluster_id in np.unique(cluster_ids):
        values = shifted[cluster_ids == cluster_id]
        counts.append(float(len(values)))
        sums.append(float(values.sum()))
        sumsqs.append(float((values**2).sum()))
    return counts, sums, sumsqs


def test_sufficient_stats_matches_dataframe_icc():
    """The sufficient-statistics estimator equals the per-observation one, shifted or not."""
    rng = np.random.default_rng(7)
    cluster_sizes = rng.integers(2, 30, size=50)
    cluster_ids = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)
    cluster_means = rng.normal(loc=100, scale=10, size=len(cluster_sizes))
    y = np.repeat(cluster_means, cluster_sizes) + rng.normal(0, 20, size=len(cluster_ids))

    expected = calculate_icc_from_dataframe(
        pd.DataFrame({"id": cluster_ids, "y": y}), cluster_column="id", outcome_column="y"
    )
    for shift in (0.0, float(y.mean())):
        counts, sums, sumsqs = _sufficient_stats(cluster_ids, y, shift)
        icc = calculate_icc_from_sufficient_stats(counts=counts, sums=sums, sumsqs=sumsqs)
        assert icc == pytest.approx(expected, rel=1e-9)


def test_sufficient_stats_matches_dataframe_icc_binary():
    """Equivalence also holds for 0/1 outcomes, where sumsq equals sum."""
    rng = np.random.default_rng(11)
    cluster_ids = np.repeat(np.arange(40), 25)
    p_by_cluster = rng.uniform(0.2, 0.8, size=40)
    y = (rng.uniform(size=len(cluster_ids)) < np.repeat(p_by_cluster, 25)).astype(np.float64)

    expected = calculate_icc_from_dataframe(
        pd.DataFrame({"id": cluster_ids, "y": y}), cluster_column="id", outcome_column="y"
    )
    counts, sums, sumsqs = _sufficient_stats(cluster_ids, y)
    assert sumsqs == sums
    icc = calculate_icc_from_sufficient_stats(counts=counts, sums=sums, sumsqs=sumsqs)
    assert icc == pytest.approx(expected, rel=1e-9)


def test_sufficient_stats_shift_restores_precision_for_large_means():
    """Raw sums of squares lose precision when the mean dwarfs the variance; shifting fixes it."""
    rng = np.random.default_rng(3)
    cluster_ids = np.repeat(np.arange(30), 20)
    y = 1e9 + np.repeat(rng.normal(0, 1, size=30), 20) + rng.normal(0, 2, size=len(cluster_ids))

    expected = calculate_icc_from_dataframe(
        pd.DataFrame({"id": cluster_ids, "y": y}), cluster_column="id", outcome_column="y"
    )
    counts, sums, sumsqs = _sufficient_stats(cluster_ids, y, shift=1e9)
    icc = calculate_icc_from_sufficient_stats(counts=counts, sums=sums, sumsqs=sumsqs)
    assert icc == pytest.approx(expected, rel=1e-6)


def test_sufficient_stats_perfect_and_zero_clustering():
    assert calculate_icc_from_sufficient_stats(
        counts=[3, 3, 3],
        sums=[30, 60, 90],  # constant within cluster: 10s, 20s, 30s
        sumsqs=[300, 1200, 2700],
    ) == pytest.approx(1.0)
    assert calculate_icc_from_sufficient_stats(
        counts=[3, 3, 3],
        sums=[60, 60, 60],  # identical clusters: 10, 20, 30 in each
        sumsqs=[1400, 1400, 1400],
    ) == pytest.approx(0.0, abs=0.01)


def test_sufficient_stats_constant_outcome_returns_zero():
    assert calculate_icc_from_sufficient_stats(counts=[2, 2], sums=[10, 10], sumsqs=[50, 50]) == 0.0


def test_sufficient_stats_validation_errors():
    with pytest.raises(ValueError, match="Need at least 2 clusters"):
        calculate_icc_from_sufficient_stats(counts=[3], sums=[30], sumsqs=[300])
    with pytest.raises(ValueError, match="Insufficient within-cluster data"):
        calculate_icc_from_sufficient_stats(counts=[1, 1], sums=[10, 20], sumsqs=[100, 400])
    with pytest.raises(ValueError, match="same length"):
        calculate_icc_from_sufficient_stats(counts=[2, 2], sums=[10], sumsqs=[50, 50])
    with pytest.raises(ValueError, match="counts must all be positive"):
        calculate_icc_from_sufficient_stats(counts=[2, 0], sums=[10, 0], sumsqs=[50, 0])
