"""
Calculate ICC and CV (Coefficient of Variation) from database tables using Mixed Models.
"""

import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM


def calculate_icc_from_database(
    session,
    table_name: str,
    cluster_column: str,
    outcome_column: str,
) -> float:
    """
    Calculate ICC using Linear Mixed Model (more robust than ANOVA).
    Uses statsmodels MixedLM for better variance component estimation,
    especially with unbalanced clusters.

    Args:
        session: SQLAlchemy session
        table_name: Name of the table to query
        cluster_column: Column containing cluster IDs
        outcome_column: Column containing outcome values

    Returns:
        ICC value between 0 and 1
    """
    query = f"""
        SELECT {cluster_column}, {outcome_column}
        FROM {table_name}
        WHERE {outcome_column} IS NOT NULL
          AND {cluster_column} IS NOT NULL
    """

    df = pd.read_sql(query, session.bind)

    if len(df) == 0:
        raise ValueError(f"No data found in {table_name}.{outcome_column}")

    model = MixedLM.from_formula(
        f"{outcome_column} ~ 1",  # Fixed effects: just intercept
        data=df,
        groups=df[cluster_column],  # Random effect: cluster
    )

    result = model.fit(method="lbfgs", reml=True)  # REML = Restricted Maximum Likelihood

    variance_between = float(result.cov_re.iloc[0, 0])  # Between-cluster variance: random effect variance

    variance_within = float(result.scale)  # Within-cluster variance: residual variance

    icc = variance_between / (variance_between + variance_within)

    return max(0.0, min(1.0, icc))


def calculate_cluster_sizes(
    session,
    table_name: str,
    cluster_column: str,
) -> dict:
    """
    Calculate cluster size statistics including CV.

    Args:
        session: SQLAlchemy session
        table_name: Name of the table to query
        cluster_column: Column containing cluster IDs

    Returns:
        dict with:
            - avg_cluster_size: Mean cluster size
            - min_cluster_size: Smallest cluster
            - max_cluster_size: Largest cluster
            - cv: Coefficient of variation (std/mean)
            - num_clusters: Number of clusters

    """
    query = f"""
        SELECT {cluster_column}, COUNT(*) as size
        FROM {table_name}
        WHERE {cluster_column} IS NOT NULL
        GROUP BY {cluster_column}
    """

    df = pd.read_sql(query, session.bind)

    if len(df) == 0:
        raise ValueError(f"No clusters found in {table_name}.{cluster_column}")

    sizes = df["size"]

    return {
        "avg_cluster_size": float(sizes.mean()),
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "cv": float(sizes.std() / sizes.mean()),  # Coefficient of variation
        "num_clusters": len(df),
    }


def calculate_icc_and_cv_from_database(
    session,
    table_name: str,
    cluster_column: str,
    outcome_column: str,
) -> dict:
    """
    Calculate both ICC and CV in one call.

    Convenience function that calculates both statistics needed
    for cluster power analysis.

    Returns:
        dict with icc, cv, avg_cluster_size, num_clusters
    """
    icc = calculate_icc_from_database(session, table_name, cluster_column, outcome_column)
    cluster_stats = calculate_cluster_sizes(session, table_name, cluster_column)

    return {
        "icc": icc,
        "cv": cluster_stats["cv"],
        "avg_cluster_size": cluster_stats["avg_cluster_size"],
        "num_clusters": cluster_stats["num_clusters"],
        "min_cluster_size": cluster_stats["min_cluster_size"],
        "max_cluster_size": cluster_stats["max_cluster_size"],
    }
