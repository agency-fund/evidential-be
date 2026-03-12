"""
Statistical functions for calculating intracluster correlation (ICC).

These functions accept dataframes rather than database sessions, following the pattern
established in analysis.py. The API layer is responsible for fetching data from the DWH.
"""

import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM


def calculate_icc_from_dataframe(df: pd.DataFrame) -> float:
    """
    Calculate ICC using Linear Mixed Model from a dataframe.

    Args:
        df: DataFrame with 'cluster_id' and 'outcome' columns

    Returns:
        ICC value between 0 and 1

    Raises:
        ValueError: If dataframe is empty or has insufficient data
    """
    if len(df) == 0:
        raise ValueError("Cannot calculate ICC from empty dataframe")

    if df["cluster_id"].nunique() < 2:
        raise ValueError("Need at least 2 clusters to calculate ICC")

    # Fit mixed-effects model: outcome ~ 1 + (1|cluster)
    try:
        model = MixedLM.from_formula(
            "outcome ~ 1",  # Fixed effects: just intercept
            data=df,
            groups=df["cluster_id"],  # Random effect: cluster
        )
        result = model.fit(method="lbfgs", reml=True)

        # Extract variance components
        variance_between = float(result.cov_re.iloc[0, 0])  # Between-cluster variance
        variance_within = float(result.scale)  # Within-cluster variance (residual)

        # Calculate ICC = σ²_between / (σ²_between + σ²_within)
        total_variance = variance_between + variance_within

        if total_variance == 0:
            return 0.0

        icc = variance_between / total_variance

        # Ensure ICC is in valid range [0, 1]
        return max(0.0, min(1.0, icc))

    except Exception as e:
        raise ValueError(f"Failed to calculate ICC: {e}") from e
