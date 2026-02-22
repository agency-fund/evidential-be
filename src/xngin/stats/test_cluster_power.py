"""
Tests for cluster randomization power analysis using wide_dwh test data.
"""

from pathlib import Path

import pandas as pd
import pytest

from xngin.stats.cluster_power import (
    calculate_design_effect,
    calculate_effective_sample_size,
)


@pytest.fixture
def wide_dwh_data():
    """Load the wide_dwh test dataset."""
    data_path = Path(__file__).parent.parent / "apiserver" / "testdata" / "wide_dwh.csv"
    return pd.read_csv(data_path)


class TestDataExploration:
    """Explore and verify the wide_dwh test data structure."""

    def test_data_loads(self, wide_dwh_data):
        """Verify we can load the test data."""
        assert len(wide_dwh_data) == 1000
        assert "cluster_id" in wide_dwh_data.columns
        print(f"\n✓ Loaded {len(wide_dwh_data)} participants")

    def test_cluster_structure(self, wide_dwh_data):
        """Verify cluster structure."""
        n_clusters = wide_dwh_data["cluster_id"].nunique()
        cluster_sizes = wide_dwh_data.groupby("cluster_id").size()

        print(f"\n✓ Found {n_clusters} clusters")
        print(f"  Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, mean={cluster_sizes.mean():.1f}")

        assert n_clusters == 20
        assert cluster_sizes.min() > 0  # No empty clusters


class TestHelperFunctions:
    """Test the helper calculation functions."""

    def test_calculate_design_effect(self):
        """Test design effect calculation."""
        # ICC=0 → DEFF=1 (no clustering effect)
        assert calculate_design_effect(icc=0.0, avg_cluster_size=50) == 1.0

        # ICC=0.05, m=50 → DEFF = 1 + (50-1)*0.05 = 3.45
        deff = calculate_design_effect(icc=0.05, avg_cluster_size=50)
        assert deff == pytest.approx(3.45)

    def test_calculate_effective_sample_size(self):
        """Test effective sample size calculation."""
        # 1000 participants / DEFF of 2 = 500 effective
        effective_n = calculate_effective_sample_size(total_n=1000, deff=2.0)
        assert effective_n == 500


# TODO: Add more test classes for the main functions
