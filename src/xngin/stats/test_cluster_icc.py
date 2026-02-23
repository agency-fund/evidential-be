from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xngin.stats.cluster_icc import (
    calculate_cluster_sizes,
    calculate_icc_and_cv_from_database,
    calculate_icc_from_database,
)

"""
Tests for ICC and CV calculation from database.
"""


@pytest.fixture
def test_db_session() -> Generator[Session, None, None]:
    """Create a test database session connected to our PostgreSQL instance."""
    engine = create_engine("postgresql://localhost/evidential_test")
    session_local = sessionmaker(bind=engine)  # Lowercase
    session = session_local()
    yield session
    session.close()


@pytest.mark.skip(reason="Requires local PostgreSQL - for development only")
class TestICCCalculation:
    """Test ICC calculation from database."""

    def test_calculate_icc_from_database(self, test_db_session):
        """Test ICC calculation with real database data."""
        icc = calculate_icc_from_database(
            session=test_db_session,
            table_name="wide_dwh",
            cluster_column="cluster_id",
            outcome_column="num_refunds",
        )

        print(f"\nICC for num_refunds: {icc:.6f}")

        # Should match what we calculated before
        assert 0 <= icc <= 1
        assert icc > 0  # num_refunds has clustering
        assert icc == pytest.approx(0.0212, abs=0.001)


@pytest.mark.skip(reason="Requires local PostgreSQL - for development only")
class TestClusterSizeCalculation:
    """Test cluster size and CV calculation."""

    def test_calculate_cluster_sizes(self, test_db_session):
        """Test cluster size calculation."""
        stats = calculate_cluster_sizes(
            session=test_db_session,
            table_name="wide_dwh",
            cluster_column="cluster_id",
        )

        print("\nCluster statistics:")
        print(f"  Num clusters: {stats['num_clusters']}")
        print(f"  Avg size: {stats['avg_cluster_size']:.1f}")
        print(f"  Min size: {stats['min_cluster_size']}")
        print(f"  Max size: {stats['max_cluster_size']}")
        print(f"  CV: {stats['cv']:.4f}")

        assert stats["num_clusters"] == 20
        assert stats["avg_cluster_size"] > 0
        assert stats["cv"] > 0  # Should have some variation


@pytest.mark.skip(reason="Requires local PostgreSQL - for development only")
class TestCombinedCalculation:
    """Test combined ICC and CV calculation."""

    def test_calculate_icc_and_cv_from_database(self, test_db_session):
        """Test combined calculation."""
        result = calculate_icc_and_cv_from_database(
            session=test_db_session,
            table_name="wide_dwh",
            cluster_column="cluster_id",
            outcome_column="num_refunds",
        )

        print("\nCombined results:")
        print(f"  ICC: {result['icc']:.4f}")
        print(f"  CV: {result['cv']:.4f}")
        print(f"  Avg cluster size: {result['avg_cluster_size']:.1f}")
        print(f"  Num clusters: {result['num_clusters']}")

        # All values should be positive
        assert result["icc"] > 0
        assert result["cv"] > 0
        assert result["avg_cluster_size"] > 0
        assert result["num_clusters"] == 20
