"""
Statistical functions for calculating intracluster correlation (ICC).

These functions accept dataframes rather than database sessions, following the pattern
established in analysis.py. The API layer is responsible for fetching data from the DWH.
"""

import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM


def calculate_icc_from_dataframe(df: pd.DataFrame, *, cluster_column: str, outcome_column: str) -> float:
    """
    Calculate ICC using Linear Mixed Model from a dataframe.

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

    # Fit mixed-effects model: outcome ~ 1 + (1|cluster)
    # Use array API (intercept design matrix) so outcome column names need not be Patsy-safe.
    try:
        endog = df[outcome_column].to_numpy(dtype=np.float64)
        exog = np.ones((len(endog), 1), dtype=np.float64)
        groups = df[cluster_column]
        result = MixedLM(endog, exog, groups=groups).fit(method="lbfgs", reml=True)

        # Extract variance components
        variance_between = float(result.cov_re[0, 0])  # Between-cluster variance
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
