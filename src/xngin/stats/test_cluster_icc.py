"""
Tests for ICC calculation from dataframes.
"""

import numpy as np
import pandas as pd
import pytest

from xngin.stats.cluster_icc import calculate_icc_from_dataframe


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

    def test_calculate_icc_unbalanced_cluster_sizes(self):
        """ICC stays in [0, 1] with unequal cluster sizes."""
        df = pd.DataFrame({
            "id": [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
            "_y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        icc = calculate_icc_from_dataframe(df, cluster_column="id", outcome_column="_y")
        assert 0.0 <= icc <= 1.0
        assert icc == pytest.approx(0.8571, abs=1e-4)
