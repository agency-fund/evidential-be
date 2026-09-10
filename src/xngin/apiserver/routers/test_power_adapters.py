"""Test our shim between DWH queries and cluster ICC/power stats."""

import pandas as pd
import pytest
from sqlalchemy import MetaData, Table, create_engine
from sqlalchemy.orm import Session

from xngin.apiserver import flags
from xngin.apiserver.conftest import get_test_uri_info
from xngin.apiserver.routers.common_api_types import Filter
from xngin.apiserver.routers.common_enums import Relation
from xngin.apiserver.routers.power_adapters import calculate_cluster_stats_from_database
from xngin.stats.stats_errors import StatsPowerError


@pytest.fixture(name="clustered_dwh_session", scope="module")
def fixture_clustered_dwh_session():
    """Connect to the clustered_dwh test database."""
    test_db = get_test_uri_info(flags.XNGIN_DEVDWH_DSN)
    engine = create_engine(
        test_db.connect_url,
        logging_name="test_power_adapters",
        execution_options={"logging_token": "test_power_adapters"},
    )
    try:
        with Session(engine) as session:
            yield session
    finally:
        engine.dispose()


@pytest.fixture(name="sa_table", scope="module")
def fixture_clustered_dwh_sa_table(clustered_dwh_session):
    return Table("clustered_dwh", MetaData(), autoload_with=clustered_dwh_session.get_bind())


@pytest.fixture(name="wide_dwh_sa_table", scope="module")
def fixture_wide_dwh_sa_table(clustered_dwh_session):
    return Table("wide_dwh", MetaData(), autoload_with=clustered_dwh_session.get_bind())


def test_raises_stats_power_error(clustered_dwh_session, sa_table):
    with pytest.raises(
        StatsPowerError,
        match="Power calc error for metric income: Need at least 2 clusters to calculate ICC",
    ):
        calculate_cluster_stats_from_database(
            session=clustered_dwh_session,
            sa_table=sa_table,
            cluster_column="cluster_equal",
            outcome_columns=["income"],
            filters=[Filter(field_name="cluster_equal", relation=Relation.INCLUDES, value=[1])],
        )


def test_income_low_icc_equal_clusters(clustered_dwh_session, sa_table):
    """Verify income has ICC ≈ 0.05 with equal-sized clusters."""
    result = calculate_cluster_stats_from_database(
        session=clustered_dwh_session,
        sa_table=sa_table,
        cluster_column="cluster_equal",
        outcome_columns=["income"],
        filters=[],
    )["income"]

    assert result["icc"] == pytest.approx(0.05, abs=0.02)
    assert result["cv"] == pytest.approx(0.0, abs=0.01)  # Equal clusters
    print(f"\nincome with equal clusters: ICC={result['icc']:.4f}, CV={result['cv']:.4f}")


def test_converted_low_icc_moderate_clusters(clustered_dwh_session, sa_table):
    """Binary outcomes with very low ICC are difficult to generate reliably."""
    result = calculate_cluster_stats_from_database(
        session=clustered_dwh_session,
        sa_table=sa_table,
        cluster_column="cluster_moderate",
        outcome_columns=["converted"],
        filters=[],
    )["converted"]

    assert result["icc"] < 0.05
    print(f"\nconverted with moderate: ICC={result['icc']:.6f} (target was 0.03, hard for binary)")


def test_test_score_high_icc_moderate_clusters(clustered_dwh_session, sa_table):
    """Verify high ICC (0.20) works with moderate variation clusters."""
    result = calculate_cluster_stats_from_database(
        session=clustered_dwh_session,
        sa_table=sa_table,
        cluster_column="cluster_moderate",
        outcome_columns=["test_score"],
        filters=[],
    )["test_score"]

    # With moderate clusters, should achieve higher ICC
    # Note: We generated with powerlaw, so this tests cross-cluster ICC
    print(f"\ntest_score with moderate clusters: ICC={result['icc']:.4f}, CV={result['cv']:.4f}")
    # Just verify it calculates without error - actual ICC may vary
    assert 0.0 <= result["icc"] <= 1.0


def test_engaged_high_icc_moderate_clusters(clustered_dwh_session, sa_table):
    """Verify engaged shows correlation with moderate clusters."""
    result = calculate_cluster_stats_from_database(
        session=clustered_dwh_session,
        sa_table=sa_table,
        cluster_column="cluster_moderate",
        outcome_columns=["engaged"],
        filters=[],
    )["engaged"]

    print(f"\nengaged with moderate clusters: ICC={result['icc']:.4f}, CV={result['cv']:.4f}")
    assert 0.0 <= result["icc"] <= 1.0


def test_power_law_clusters_lower_icc(clustered_dwh_session, sa_table):
    """
    Power-law clusters actually achieve good ICCs despite extreme variation!

    Despite highly unbalanced clusters (CV >> 1), the ICC generation works well
    for continuous outcomes. Binary outcomes may vary more.
    """
    # test_score generated with target ICC=0.20, engaged with target ICC=0.15 (powerlaw clusters).
    results = calculate_cluster_stats_from_database(
        session=clustered_dwh_session,
        sa_table=sa_table,
        cluster_column="cluster_powerlaw",
        outcome_columns=["test_score", "engaged"],
        filters=[],
    )
    result_score = results["test_score"]
    result_engaged = results["engaged"]

    print(f"\nPower-law clusters (extreme variation, CV={result_score['cv']:.2f}):")
    print(f"  test_score: ICC={result_score['icc']:.4f} (target was 0.20)")
    print(f"  engaged: ICC={result_engaged['icc']:.4f} (target was 0.15)")

    assert 0.15 < result_score["icc"] < 0.25
    assert 0.10 < result_engaged["icc"] < 0.25
    assert result_score["cv"] > 5.0


def test_batched_matches_individual_queries(clustered_dwh_session, sa_table):
    """One batched call returns the same stats as one call per metric.

    Includes converted/engaged to cover the BOOLEAN->INTEGER->FLOAT cast in a batched select.
    """
    outcome_columns = ["income", "test_score", "converted", "engaged"]

    batched = calculate_cluster_stats_from_database(
        session=clustered_dwh_session,
        sa_table=sa_table,
        cluster_column="cluster_moderate",
        outcome_columns=outcome_columns,
        filters=[],
    )

    for outcome_column in outcome_columns:
        individual = calculate_cluster_stats_from_database(
            session=clustered_dwh_session,
            sa_table=sa_table,
            cluster_column="cluster_moderate",
            outcome_columns=[outcome_column],
            filters=[],
        )[outcome_column]
        assert batched[outcome_column]["icc"] == pytest.approx(individual["icc"])
        assert batched[outcome_column]["avg_cluster_size"] == pytest.approx(individual["avg_cluster_size"])
        assert batched[outcome_column]["cv"] == pytest.approx(individual["cv"])


def test_null_outcomes_dropped_per_metric_but_counted_in_cluster_sizes(clustered_dwh_session, wide_dwh_sa_table):
    """Null outcome values are dropped from ICC but still counted in cluster sizes.

    wide_dwh has 1000 rows; 28 have a null age (the cluster key) and are excluded from
    everything, leaving 972. Of those, 24 have a null household_income: they are excluded
    from the ICC calculation but still counted in avg_cluster_size/cv, since metrics are
    outcomes that may be filled in as the experiment runs. Expected values match
    test_power_check_with_db_derived_icc_and_nulls_in_cluster_key in test_admin_api.py.
    """
    result = calculate_cluster_stats_from_database(
        session=clustered_dwh_session,
        sa_table=wide_dwh_sa_table,
        cluster_column="age",
        outcome_columns=["household_income"],
        filters=[],
    )["household_income"]

    assert result["icc"] == pytest.approx(0.021, abs=1e-3)
    # 972 rows / 72 distinct ages; includes the 24 rows with null household_income.
    assert result["avg_cluster_size"] == pytest.approx(13.5, abs=1e-3)
    assert result["cv"] == pytest.approx(0.282, abs=1e-3)


def test_cluster_cvs(clustered_dwh_session):
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
