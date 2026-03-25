"""
Statistical functions for calculating intracluster correlation (ICC).

These functions accept dataframes rather than database sessions, following the pattern
established in analysis.py. The API layer is responsible for fetching data from the DWH.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM


def _fit_random_intercept_mixedlm(
    outcomes: pd.Series,
    clusters: pd.Series,
    *,
    fit_kwargs: dict[str, Any] | None = None,
):
    """Fit mixed-effects model: outcome ~ 1 + (1|cluster); raise if the optimizer did not report convergence."""
    fit_kwargs = fit_kwargs or {}
    # Use array API (intercept design matrix) so outcome column names don't have to be Patsy-safe.
    endog = outcomes.to_numpy(dtype=np.float64)
    exog = np.ones((len(endog), 1))
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always")
        result = MixedLM(endog, exog, groups=clusters).fit(method="lbfgs", reml=True, **fit_kwargs)

    if not result.converged:
        msgs = list(dict.fromkeys(str(w.message) for w in fit_warnings))
        detail = "; ".join(msgs) if msgs else "no warnings were recorded during fit"
        raise ValueError(f"Mixed-effects ICC fit did not converge: {detail}")

    return result


def calculate_icc_from_dataframe(
    df: pd.DataFrame,
    *,
    cluster_column: str,
    outcome_column: str,
    mixedlm_fit_kwargs: dict[str, Any] | None = None,
) -> float:
    """
    Calculate ICC using Linear Mixed Model from a dataframe.

    Args:
        df: DataFrame containing cluster and outcome columns
        cluster_column: Name of the column with cluster identifiers
        outcome_column: Name of the column with numeric outcomes
        mixedlm_fit_kwargs: Optional extra keyword arguments forwarded to ``MixedLM.fit`` after
            ``method`` and ``reml`` (e.g. ``{"maxiter": 0}`` in tests to force non-convergence).

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

    cluster_ids = df[cluster_column]
    if cluster_ids.nunique() < 2:
        raise ValueError("Need at least 2 clusters to calculate ICC")

    # Early check of no clustering to avoid mixedlm numerical error when variance is zero, by
    # computing the between-cluster sum of squares as used in one-way ANOVA for ICC:
    #     icc_anova = ss_between / ss_total
    # where ss_total = ((y - grand_mean)**2).sum()
    y = df[outcome_column]
    cluster_sizes = cluster_ids.value_counts()
    cluster_means = y.groupby(cluster_ids).mean()
    grand_mean = y.mean()
    ss_between = ((cluster_means - grand_mean) ** 2 * cluster_sizes).sum()
    if np.isclose(ss_between, 0.0):
        return 0.0

    try:
        result = _fit_random_intercept_mixedlm(y, cluster_ids, fit_kwargs=mixedlm_fit_kwargs)
        # Extract variance components
        variance_between = float(result.cov_re[0, 0])  # Between-cluster variance
        variance_within = float(result.scale)  # Within-cluster variance (residual)
        # Calculate ICC = σ²_between / (σ²_between + σ²_within)
        total_variance = variance_between + variance_within
        icc = variance_between / total_variance

        # Ensure ICC is in valid range [0, 1]
        return max(0.0, min(1.0, icc))
    except Exception as e:
        # Note: could fall back to the one-way ANOVA ICC calculation as an alternative.
        raise ValueError(f"Failed to calculate ICC: {e}") from e
