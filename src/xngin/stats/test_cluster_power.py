"""
Tests for cluster randomization power analysis using wide_dwh test data.
"""

import math
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
        deff_with_cv = calculate_design_effect(icc, avg_cluster_size, cv)

        # World Bank reports DEFT = 8.08, which means DEFF = 8.08² = 65.2864
        expected_deff_cv = 65.25
        assert deff_with_cv == pytest.approx(expected_deff_cv, abs=0.5)

        # Verify DEFT matches what World Bank reports
        deft_with_cv = math.sqrt(deff_with_cv)
        assert deft_with_cv == pytest.approx(8.08, abs=0.01)

        print("\n✅ World Bank verification:")
        print(f"  Standard DEFF: {deff_standard:.2f} (DEFT: {deft_standard:.2f})")
        print(f"  DEFF with CV: {deff_with_cv:.2f} (DEFT: {deft_with_cv:.2f})")

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

        print("\n📊 World Bank MDE verification:")
        print(f"  Standard DEFT: {deft_standard:.2f}")
        print(f"  DEFT with CV: {deft_with_cv:.2f}")
        print(f"  DEFT ratio: {deft_ratio:.2f}")
        print(f"  MDE ratio (from blog): {mde_ratio:.2f}")
        print(f"  Difference: {abs(deft_ratio - mde_ratio):.3f}")

        # The ratios should match
        assert deft_ratio == pytest.approx(mde_ratio, abs=0.05)

        print("  ✅ DEFT ratio matches MDE ratio!")


# TODO: Add more test classes for the main functions
