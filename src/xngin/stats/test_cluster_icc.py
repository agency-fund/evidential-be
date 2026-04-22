"""Tests for ICC and CV calculations from dataframes or databases."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from xngin.stats.cluster_icc import calculate_icc_from_dataframe


class TestGeneratedClusteredData:
    """Test ICC/CV calculation on generated clustered_dwh data."""

    @pytest.fixture(name="clustered_dwh_session")
    def fixture_clustered_dwh_session(self):
        """Load clustered_dwh data into SQLite for testing."""

        data_path = Path(__file__).parent.parent / "apiserver/testdata/clustered_dwh.csv.zst"
        df = pd.read_csv(data_path)

        engine = create_engine("sqlite:///:memory:")
        df.to_sql("clustered_dwh", engine, index=False)

        with Session(engine) as session:
            yield session

    def test_cluster_cvs(self, clustered_dwh_session):
        """Verify cluster schemes have correct coefficient of variation."""

        df = pd.read_sql("SELECT * FROM clustered_dwh", clustered_dwh_session.bind)

        equal_sizes = df.groupby("cluster_equal").size()
        equal_cv = equal_sizes.std() / equal_sizes.mean()
        assert pytest.approx(equal_cv) == 0.0

        moderate_sizes = df.groupby("cluster_moderate").size()
        moderate_cv = moderate_sizes.std() / moderate_sizes.mean()
        assert pytest.approx(moderate_cv, abs=1e-3) == 0.347

        powerlaw_sizes = df.groupby("cluster_powerlaw").size()
        powerlaw_cv = powerlaw_sizes.std() / powerlaw_sizes.mean()
        assert pytest.approx(powerlaw_cv, abs=1e-3) == 11.872

        print("\nCluster size CVs:")
        print(f"  Equal: {equal_cv:.3f} (range: [{equal_sizes.min()}, {equal_sizes.max()}])")
        print(f"  Moderate: {moderate_cv:.3f} (range: [{moderate_sizes.min()}, {moderate_sizes.max()}])")
        print(f"  Power-law: {powerlaw_cv:.3f} (range: [{powerlaw_sizes.min()}, {powerlaw_sizes.max()}])")


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
