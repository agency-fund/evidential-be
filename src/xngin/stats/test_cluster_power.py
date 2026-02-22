"""
Tests for cluster randomization power analysis using wide_dwh test data.
"""

import math
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from xngin.apiserver.routers.common_api_types import DesignSpecMetric, MetricType
from xngin.stats.cluster_power import (
    calculate_design_effect,
    calculate_effective_sample_size,
    calculate_mde_cluster,  # ← Make sure this is here!
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


class TestICCComparison:
    """Calculate ICC from unbalanced cluster data."""

    def test_calculate_icc_statsmodels(self, wide_dwh_data):
        """Calculate ICC using statsmodels (works with unbalanced clusters)."""
        df = wide_dwh_data[["cluster_id", "revenue_30d"]].dropna()

        # Method 1: Statsmodels (recommended for unbalanced clusters)
        model = ols("revenue_30d ~ C(cluster_id)", data=df).fit()
        anova_table = anova_lm(model, typ=1)

        ms_between = anova_table.loc["C(cluster_id)", "mean_sq"]
        ms_within = anova_table.loc["Residual", "mean_sq"]

        # Calculate average cluster size
        n_per_cluster = len(df) / df["cluster_id"].nunique()

        # ICC formula: (MSB - MSW) / (MSB + (m-1)*MSW)
        icc_statsmodels = (ms_between - ms_within) / (ms_between + (n_per_cluster - 1) * ms_within)
        icc_statsmodels = max(0.0, min(1.0, icc_statsmodels))

        # Method 2: Manual ANOVA calculation (should match)
        grand_mean = df["revenue_30d"].mean()
        clusters = df.groupby("cluster_id")["revenue_30d"]

        # Between-cluster variance
        cluster_means = clusters.mean()
        cluster_sizes = clusters.size()
        ssb = sum(cluster_sizes * (cluster_means - grand_mean) ** 2)
        msb = ssb / (df["cluster_id"].nunique() - 1)

        # Within-cluster variance
        ssw = sum(clusters.apply(lambda x: sum((x - x.mean()) ** 2)))
        msw = ssw / (len(df) - df["cluster_id"].nunique())

        icc_manual = (msb - msw) / (msb + (n_per_cluster - 1) * msw)
        icc_manual = max(0.0, min(1.0, icc_manual))

        print("\n" + "=" * 60)
        print("ICC Calculation for revenue_30d (unbalanced clusters):")
        print("=" * 60)
        print(f"  Number of clusters: {df['cluster_id'].nunique()}")
        print(f"  Total observations: {len(df)}")
        print(f"  Avg cluster size:   {n_per_cluster:.1f}")
        print(f"  Cluster sizes:      {cluster_sizes.min()}-{cluster_sizes.max()}")
        print()
        print(f"  ICC (statsmodels):  {icc_statsmodels:.6f}")
        print(f"  ICC (manual):       {icc_manual:.6f}")
        print(f"  Difference:         {abs(icc_statsmodels - icc_manual):.6f}")
        print("=" * 60)

        # They should match very closely
        assert abs(icc_statsmodels - icc_manual) < 0.0001
        assert 0 <= icc_statsmodels <= 1

    def test_calculate_icc_converted(self, wide_dwh_data):
        """Calculate ICC for binary converted metric."""
        df = wide_dwh_data[["cluster_id", "converted"]].dropna()
        df["converted_numeric"] = df["converted"].astype(float)

        model = ols("converted_numeric ~ C(cluster_id)", data=df).fit()
        anova_table = anova_lm(model, typ=1)

        ms_between = anova_table.loc["C(cluster_id)", "mean_sq"]
        ms_within = anova_table.loc["Residual", "mean_sq"]
        n_per_cluster = len(df) / df["cluster_id"].nunique()

        icc = (ms_between - ms_within) / (ms_between + (n_per_cluster - 1) * ms_within)
        icc = max(0.0, min(1.0, icc))

        print(f"\nICC for conversion (binary): {icc:.6f}")

        assert 0 <= icc <= 1

    def test_diagnose_icc_calculation(self, wide_dwh_data):
        """Investigate why ICC is exactly zero."""

        df = wide_dwh_data[["cluster_id", "revenue_30d"]].dropna()

        # Calculate ANOVA
        model = ols("revenue_30d ~ C(cluster_id)", data=df).fit()
        anova_table = anova_lm(model, typ=1)

        print("\n" + "=" * 60)
        print("ANOVA Table Breakdown:")
        print("=" * 60)
        print(anova_table)
        print()

        ms_between = anova_table.loc["C(cluster_id)", "mean_sq"]
        ms_within = anova_table.loc["Residual", "mean_sq"]
        f_stat = anova_table.loc["C(cluster_id)", "F"]
        p_value = anova_table.loc["C(cluster_id)", "PR(>F)"]

        n_per_cluster = len(df) / df["cluster_id"].nunique()

        print(f"Mean Square Between (MSB): {ms_between:,.2f}")
        print(f"Mean Square Within (MSW):  {ms_within:,.2f}")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Average cluster size: {n_per_cluster:.1f}")
        print()

        # Calculate ICC without capping
        icc_raw = (ms_between - ms_within) / (ms_between + (n_per_cluster - 1) * ms_within)
        icc_capped = max(0.0, min(1.0, icc_raw))

        print(f"ICC (raw, before capping): {icc_raw:.6f}")
        print(f"ICC (capped to [0,1]):     {icc_capped:.6f}")
        print()

        # Check cluster means
        cluster_stats = df.groupby("cluster_id")["revenue_30d"].agg(["mean", "std", "count"])
        print("Cluster statistics:")
        print(cluster_stats)
        print()
        print(f"Range of cluster means: ${cluster_stats['mean'].min():.2f} - ${cluster_stats['mean'].max():.2f}")
        print(f"Std dev of cluster means: ${cluster_stats['mean'].std():.2f}")
        print(f"Grand mean: ${df['revenue_30d'].mean():.2f}")
        print("=" * 60)

        # If p-value is high, clusters don't differ significantly
        if p_value > 0.05:
            print("\n⚠️  P-value > 0.05: Cluster means are NOT significantly different!")
            print("    This explains why ICC ≈ 0")

    def test_search_for_clustering(self, wide_dwh_data):
        """Search through all numeric columns to find which have ICC > 0."""
        df = wide_dwh_data

        # Get all numeric columns (exclude cluster_id itself)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in {"id", "cluster_id", "participant_uuid"}]

        print("\n" + "=" * 80)
        print("Searching for columns with clustering effect (ICC > 0)...")
        print("=" * 80)

        results = []

        for col in numeric_cols[:50]:  # Test first 50 to keep it manageable
            try:
                data = df[["cluster_id", col]].dropna()
                if len(data) < 100:  # Skip if too little data
                    continue

                # Quick ANOVA
                model = ols(f"{col} ~ C(cluster_id)", data=data).fit()
                anova_table = anova_lm(model, typ=1)

                ms_between = anova_table.loc["C(cluster_id)", "mean_sq"]
                ms_within = anova_table.loc["Residual", "mean_sq"]
                p_value = anova_table.loc["C(cluster_id)", "PR(>F)"]

                n_per_cluster = len(data) / data["cluster_id"].nunique()
                icc = (ms_between - ms_within) / (ms_between + (n_per_cluster - 1) * ms_within)

                if icc > 0:  # Found one with positive ICC!
                    results.append({"column": col, "icc": icc, "p_value": p_value, "n": len(data)})
            except Exception:
                continue

        # Sort by ICC descending
        results.sort(key=itemgetter("icc"), reverse=True)
        print(f"\nFound {len(results)} columns with ICC > 0:")
        print("-" * 80)
        for r in results[:10]:  # Show top 10
            sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
            print(f"{r['column']:40s}  ICC={r['icc']:7.4f}  p={r['p_value']:.4f} {sig}")
        print("=" * 80)

        if len(results) > 0:
            print(f"\n✓ Best column for testing: {results[0]['column']} (ICC={results[0]['icc']:.4f})")
        else:
            print("\n✗ No columns found with ICC > 0 - all cluster assignments appear random")

    def test_check_geographic_clustering(self, wide_dwh_data):
        """Check if geographic columns show clustering."""
        df = wide_dwh_data

        # Try geographic groupings instead of cluster_id
        geo_cols = ["country_code", "region", "city"]

        print("\n" + "=" * 80)
        print("Checking if GEOGRAPHIC columns create natural clusters:")
        print("=" * 80)

        for geo_col in geo_cols:
            data = df[[geo_col, "revenue_30d"]].dropna()

            try:
                model = ols(f"revenue_30d ~ C({geo_col})", data=data).fit()
                anova_table = anova_lm(model, typ=1)

                ms_between = anova_table.loc[f"C({geo_col})", "mean_sq"]
                ms_within = anova_table.loc["Residual", "mean_sq"]
                p_value = anova_table.loc[f"C({geo_col})", "PR(>F)"]

                n_groups = data[geo_col].nunique()
                n_per_group = len(data) / n_groups

                icc = (ms_between - ms_within) / (ms_between + (n_per_group - 1) * ms_within)
                icc = max(0.0, icc)

                print(f"\n{geo_col}:")
                print(f"  Number of groups: {n_groups}")
                print(f"  Avg group size: {n_per_group:.1f}")
                print(f"  ICC: {icc:.6f}")
                print(f"  P-value: {p_value:.4f}")

                if icc > 0 and p_value < 0.05:
                    print("  ✓ This creates REAL clusters!")
                else:
                    print("  ✗ No clustering effect")

            except Exception as e:
                print(f"\n{geo_col}: Error - {e}")

        print("=" * 80)

    def test_cluster_power_with_real_clustering(self, wide_dwh_data):
        """Test cluster power functions using num_refunds (has real ICC=0.02)."""
        df = wide_dwh_data[["cluster_id", "num_refunds"]].dropna()

        # Calculate real ICC
        model = ols("num_refunds ~ C(cluster_id)", data=df).fit()
        anova_table = anova_lm(model, typ=1)
        ms_between = anova_table.loc["C(cluster_id)", "mean_sq"]
        ms_within = anova_table.loc["Residual", "mean_sq"]
        n_per_cluster = len(df) / df["cluster_id"].nunique()
        icc = (ms_between - ms_within) / (ms_between + (n_per_cluster - 1) * ms_within)

        # Get baseline and variance
        baseline = df["num_refunds"].mean()
        variance = df["num_refunds"].var()
        avg_cluster_size = n_per_cluster

        print("\n" + "=" * 70)
        print("Testing with REAL clustered data (num_refunds):")
        print("=" * 70)
        print(f"  N: {len(df)} participants, {df['cluster_id'].nunique()} clusters")
        print(f"  Baseline refunds: {baseline:.2f}")
        print(f"  Variance: {variance:.2f}")
        print(f"  ICC: {icc:.4f} (real clustering effect!)")
        print(f"  Avg cluster size: {avg_cluster_size:.1f}")
        print()

        # Create metric
        metric = DesignSpecMetric(
            field_name="num_refunds",
            metric_type=MetricType.NUMERIC,
            metric_baseline=baseline,
            metric_target=baseline * 0.8,  # 20% reduction in refunds
            metric_stddev=math.sqrt(variance),  # Convert variance to stddev!
            available_nonnull_n=len(df),
            available_n=len(df),
        )

        # Test MDE calculation
        target_value, pct_change = calculate_mde_cluster(
            available_n=1000,
            metric=metric,
            n_arms=2,
            icc=icc,
            avg_cluster_size=avg_cluster_size,
        )

        deff = calculate_design_effect(icc, avg_cluster_size)
        effective_n = int(1000 / deff)

        print(f"Design Effect (DEFF): {deff:.3f}")
        print(f"Effective sample size: {effective_n} (vs 1000 actual)")
        print(f"Power loss: {(1000 - effective_n) / 1000 * 100:.1f}%")
        print()
        print("MDE with 1000 participants:")
        print(f"  Target: {target_value:.2f} refunds")
        print(f"  Change: {pct_change:.1f}%")
        print("=" * 70)

        # Verify the calculations make sense
        assert deff > 1.0  # Should have design effect with ICC > 0
        assert effective_n < 1000  # Effective N should be less than actual N
        assert target_value != baseline  # MDE should differ from baseline


# TODO: Add more test classes for the main functions
