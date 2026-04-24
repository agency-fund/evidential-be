"""Test our shim between DWH queries and cluster ICC/power stats."""

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


class TestGeneratedClusteredData:
    """Test ICC/CV calculation on generated clustered_dwh data."""

    @pytest.fixture(name="clustered_dwh_session")
    def fixture_clustered_dwh_session(self):
        """Load clustered_dwh data into SQLite for testing."""

        data_path = Path(__file__).parent.parent / "testdata/clustered_dwh.csv.zst"
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
