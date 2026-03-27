"""
Statistical functions for calculating intracluster correlation (ICC).

These functions accept dataframes rather than database sessions, following the pattern
established in analysis.py. The API layer is responsible for fetching data from the DWH.
"""

import numpy as np
import pandas as pd


def _icc_one_way_random_intercept(df: pd.DataFrame, *, cluster_column: str, outcome_column: str) -> float:
    """
    ICC for a single-intercept random-cluster model (outcome ~ 1 + (1|cluster)).

    Estimated from one-way random-effects ANOVA mean squares (Method of Moments):

        ICC = σ²_between / (σ²_between + σ²_within) = (MSB - MSW) / (MSB + (n0 - 1) * MSW)

    where n0 is the adjustment for unequal cluster sizes. ICC estimates are clamped to [0, 1].
    """
    # 1. cluster stats
    n_total = len(df)
    y = df[outcome_column].to_numpy(dtype=np.float64)
    codes, _ = pd.factorize(df[cluster_column])  # indices of cluster membership
    cluster_sizes = np.bincount(codes).astype(np.float64)
    k = len(cluster_sizes)

    # 2. ANOVA components for ICC
    df_b = k - 1
    df_w = n_total - k
    if df_w < 1:
        raise ValueError("Insufficient within-cluster data (need N > clusters)")

    cluster_sums = np.bincount(codes, weights=y)
    cluster_means = cluster_sums / cluster_sizes
    grand_mean = np.mean(y)
    # Sum of Squares Between (SSB)
    ssb = np.sum(cluster_sizes * (cluster_means - grand_mean) ** 2)
    # Sum of Squares Within (SSW) - subtract the mean of each cluster from each observation
    ssw = np.sum((y - cluster_means[codes]) ** 2)
    # Derive the mean squares
    msb = ssb / df_b
    msw = ssw / df_w

    # 3. n0 calculation; can think of it as an "effective cluster size" when we have unequal sizes
    # for an unbiased estimate of ICC.  n0 equals the common cluster size when sizes are balanced.
    sum_n2 = np.sum(cluster_sizes**2)
    n0 = (n_total - (sum_n2 / n_total)) / df_b

    # 4. Final calculation of ICC as the ratio of between-cluster variance to total variance.
    # If MSB < MSW, the point estimate for sigma_b is 0 (prevents negative ICC)
    var_between = max(0.0, (msb - msw) / n0)
    var_within = msw
    total_var = var_between + var_within
    if total_var == 0:
        return 0.0

    return float(np.clip(var_between / total_var, 0.0, 1.0))


def calculate_icc_from_dataframe(
    df: pd.DataFrame,
    *,
    cluster_column: str,
    outcome_column: str,
) -> float:
    """
    Calculate intraclass correlation (ICC) from a dataframe.

    Uses a one-way random-effects ANOVA estimator, equivalent to the variance-component
    ICC from a linear mixed model with a random intercept per cluster and no fixed
    effects beyond the grand mean.

    Args:
        df: DataFrame containing cluster and outcome columns
        cluster_column: Name of the column with cluster identifiers
        outcome_column: Name of the column with numeric outcomes

    Returns:
        ICC value between 0 and 1

    Raises:
        ValueError: If dataframe is empty or has insufficient data
    """
    if len(df) == 0:
        raise ValueError("Cannot calculate ICC from empty dataframe")

    missing = [name for name in (cluster_column, outcome_column) if name not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns: {', '.join(missing)}")

    if df[cluster_column].nunique() < 2:
        raise ValueError("Need at least 2 clusters to calculate ICC")

    if df[outcome_column].isna().any():
        raise ValueError(f"outcome column '{outcome_column}' contains NaN values")

    # Note: If we need to add covariates, we can use a mixed model ICC here if the dataset is large enough.
    return _icc_one_way_random_intercept(df, cluster_column=cluster_column, outcome_column=outcome_column)
