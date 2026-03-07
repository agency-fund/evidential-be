"""
Tests for ICC calculation from dataframes.
"""

import pandas as pd
import pytest

from xngin.stats.cluster_icc import calculate_icc_from_dataframe


class TestICCFromDataFrame:
    """Test ICC calculation from DataFrame (no database required)."""

    def test_calculate_icc_perfect_clustering(self):
        """Test ICC when all variance is between clusters (ICC = 1)."""
        df = pd.DataFrame({
            "cluster_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "outcome": [10, 10, 10, 20, 20, 20, 30, 30, 30],
        })

        icc = calculate_icc_from_dataframe(df)

        assert icc == pytest.approx(1.0, abs=0.01)

    def test_calculate_icc_no_clustering(self):
        """Test ICC when all variance is within clusters (ICC = 0)."""
        df = pd.DataFrame({
            "cluster_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "outcome": [10, 20, 30, 10, 20, 30, 10, 20, 30],
        })

        icc = calculate_icc_from_dataframe(df)

        assert icc == pytest.approx(0.0, abs=0.1)

    def test_calculate_icc_moderate_clustering(self):
        """Test ICC with moderate clustering (0 < ICC < 1)."""
        df = pd.DataFrame({
            "cluster_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "outcome": [8, 9, 10, 11, 18, 19, 20, 21, 28, 29, 30, 31],
        })

        icc = calculate_icc_from_dataframe(df)

        assert 0 < icc < 1
        assert icc == pytest.approx(0.95, abs=0.1)

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        df = pd.DataFrame({"cluster_id": [], "outcome": []})

        with pytest.raises(ValueError, match="Cannot calculate ICC from empty dataframe"):
            calculate_icc_from_dataframe(df)

    def test_single_cluster_raises_error(self):
        """Test that single cluster raises ValueError."""
        df = pd.DataFrame({
            "cluster_id": [1, 1, 1],
            "outcome": [10, 20, 30],
        })

        with pytest.raises(ValueError, match="Need at least 2 clusters"):
            calculate_icc_from_dataframe(df)

    def test_icc_bounds(self):
        """Test that ICC is always between 0 and 1."""
        test_cases = [
            pd.DataFrame({
                "cluster_id": [1] * 10 + [2] * 10,
                "outcome": [10] * 10 + [100] * 10,
            }),
            pd.DataFrame({
                "cluster_id": [1, 1, 2, 2, 3, 3],
                "outcome": [1, 100, 1, 100, 1, 100],
            }),
        ]

        for df in test_cases:
            icc = calculate_icc_from_dataframe(df)
            assert 0 <= icc <= 1
