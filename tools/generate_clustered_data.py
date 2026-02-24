#!/usr/bin/env python3
"""
Generate realistic clustered test data for cluster randomization testing.

Creates data with MULTIPLE clustering schemes (equal, moderate, power-law)
and MULTIPLE outcomes (continuous/binary, low/high ICC) for comprehensive testing.

Usage:
    python tools/generate_clustered_data.py --output data.csv
    python tools/generate_clustered_data.py --n-participants 10000 --n-clusters 1000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_cluster_sizes(
    n_clusters: int,
    distribution: str = "power_law",
    target_cv: float = 3.0,
    min_size: int = 1,
    max_size: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate cluster sizes following specified distribution.

    Args:
        n_clusters: Number of clusters to generate
        distribution: "power_law", "equal", or "moderate"
        target_cv: Target coefficient of variation (for power_law/moderate)
        min_size: Minimum cluster size
        max_size: Maximum cluster size
        seed: Random seed for reproducibility

    Returns:
        Array of cluster sizes (integers)
    """
    rng = np.random.RandomState(seed)

    if distribution == "equal":
        # All clusters same size
        size = (min_size + max_size) // 2
        sizes = np.full(n_clusters, size)

    elif distribution == "moderate":
        # Normal distribution around midpoint
        mean_size = (min_size + max_size) // 2
        std_size = mean_size * target_cv  # CV = std/mean
        sizes = rng.normal(mean_size, std_size, n_clusters)
        sizes = np.clip(sizes, min_size, max_size).astype(int)

    elif distribution == "power_law":
        # Power-law: many small, few large
        # Generate from power distribution [0, 1]
        raw_sizes = rng.power(a=2.0, size=n_clusters)
        # Scale to [min_size, max_size]
        sizes = (raw_sizes * (max_size - min_size) + min_size).astype(int)

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return sizes


def generate_continuous_outcome_with_icc(
    cluster_sizes: np.ndarray,
    icc: float,
    mean: float = 50000,
    total_std: float = 20000,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate continuous outcome with specified ICC.

    Model: Y[i,j] = μ + u[j] + ε[i,j]
    Where:
    - μ = overall mean
    - u[j] ~ N(0, sigma²_between) = cluster effect
    - ε[i,j] ~ N(0, sigma²_within) = individual effect
    - ICC = sigma²_between / (sigma²_between + sigma²_within)

    Args:
        cluster_sizes: Array of sizes for each cluster
        icc: Intracluster correlation coefficient (0 to 1)
        mean: Overall mean of outcome
        total_std: Total standard deviation
        seed: Random seed

    Returns:
        Array of outcomes (length = sum of cluster_sizes)
    """
    rng = np.random.RandomState(seed)

    # Decompose variance into between and within components
    total_var = total_std**2
    between_var = icc * total_var
    within_var = (1 - icc) * total_var

    between_std = np.sqrt(between_var)
    within_std = np.sqrt(within_var)

    # Generate cluster effects
    n_clusters = len(cluster_sizes)
    cluster_effects = rng.normal(0, between_std, n_clusters)

    # Generate individual outcomes
    outcomes = []
    for cluster_idx, size in enumerate(cluster_sizes):
        cluster_effect = cluster_effects[cluster_idx]
        individual_effects = rng.normal(0, within_std, size)
        cluster_outcomes = mean + cluster_effect + individual_effects
        outcomes.extend(cluster_outcomes)

    return np.array(outcomes)


def generate_binary_outcome_with_icc(
    cluster_sizes: np.ndarray,
    icc: float,
    base_prob: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate binary outcome with specified ICC.

    Uses varying cluster probabilities to induce correlation.

    Args:
        cluster_sizes: Array of sizes for each cluster
        icc: Intracluster correlation coefficient
        base_prob: Overall probability of outcome=1
        seed: Random seed

    Returns:
        Array of 0/1 outcomes
    """
    rng = np.random.RandomState(seed)

    # For binary outcomes, ICC relates to variance of cluster probabilities
    # We'll use beta distribution to generate cluster-specific probabilities

    if icc == 0:
        # No clustering - everyone has same probability
        total_n = sum(cluster_sizes)
        return rng.binomial(1, base_prob, total_n)

    # Use beta distribution to generate cluster probabilities
    # that have specified variance while maintaining mean = base_prob

    # Beta parameters for desired mean and variance
    # Mean = alpha/(alpha+beta) = base_prob
    # Variance is controlled by alpha+beta (larger = less variance)

    # For ICC, we want: variance of cluster means = icc * base_prob * (1-base_prob)
    target_var = icc * base_prob * (1 - base_prob)

    # Beta variance = (alpha*beta) / [(alpha+beta)²(alpha+beta+1)]
    # Solving for alpha and beta given mean and variance:
    if target_var >= base_prob * (1 - base_prob):
        target_var = 0.99 * base_prob * (1 - base_prob)  # Cap at maximum possible

    common = (base_prob * (1 - base_prob) / target_var) - 1
    alpha = base_prob * common
    beta = (1 - base_prob) * common

    # Generate cluster-specific probabilities
    n_clusters = len(cluster_sizes)
    cluster_probs = rng.beta(alpha, beta, n_clusters)

    # Generate outcomes for each cluster
    outcomes = []
    for cluster_idx, size in enumerate(cluster_sizes):
        prob = cluster_probs[cluster_idx]
        cluster_outcomes = rng.binomial(1, prob, size)
        outcomes.extend(cluster_outcomes)

    return np.array(outcomes)


def generate_multi_cluster_data(
    total_n: int = 10000,
    n_clusters_per_scheme: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate test data with multiple clustering schemes."""
    rng = np.random.RandomState(seed)

    avg_size = total_n / n_clusters_per_scheme
    print(f"Generating {n_clusters_per_scheme} clusters...")
    print(f"Target average cluster size: {avg_size:.1f}")

    # =================================================================
    # Scheme 1: Equal clusters (CV = 0)
    # =================================================================
    equal_sizes = np.full(n_clusters_per_scheme, int(avg_size))
    equal_sizes[0] += total_n - equal_sizes.sum()

    # =================================================================
    # Scheme 2: Moderate (CV ≈ 0.3) - Multinomial with similar probs
    # =================================================================
    # Create probabilities with small variance
    base_prob = 1.0 / n_clusters_per_scheme
    noise = rng.normal(0, base_prob * 0.15, n_clusters_per_scheme)
    probs_moderate = base_prob + noise
    probs_moderate = np.maximum(probs_moderate, 0.0001)  # Keep positive
    probs_moderate /= probs_moderate.sum()  # Normalize

    # Use multinomial - guaranteed to sum to total_n
    moderate_sizes = rng.multinomial(total_n, probs_moderate)
    moderate_sizes = np.maximum(moderate_sizes, 1)  # Ensure >= 1

    # If we added 1s, we might exceed total_n, so fix it
    if moderate_sizes.sum() > total_n:
        excess = moderate_sizes.sum() - total_n
        # Remove from largest clusters
        for _ in range(excess):
            idx = np.argmax(moderate_sizes)
            if moderate_sizes[idx] > 1:
                moderate_sizes[idx] -= 1

    # =================================================================
    # Scheme 3: Power-law (CV >> 1) - Multinomial with skewed probs
    # =================================================================
    # Create power-law probabilities
    ranks = np.arange(1, n_clusters_per_scheme + 1)
    probs_powerlaw = 1.0 / (ranks**1.5)  # Power law
    probs_powerlaw /= probs_powerlaw.sum()

    # Use multinomial
    powerlaw_sizes = rng.multinomial(total_n, probs_powerlaw)
    powerlaw_sizes = np.maximum(powerlaw_sizes, 1)

    # Fix if needed
    if powerlaw_sizes.sum() > total_n:
        excess = powerlaw_sizes.sum() - total_n
        for _ in range(excess):
            idx = np.argmax(powerlaw_sizes)
            if powerlaw_sizes[idx] > 1:
                powerlaw_sizes[idx] -= 1

    # Shuffle powerlaw
    rng.shuffle(powerlaw_sizes)

    # Calculate CVs
    cv_equal = equal_sizes.std() / equal_sizes.mean() if equal_sizes.mean() > 0 else 0
    cv_moderate = moderate_sizes.std() / moderate_sizes.mean()
    cv_powerlaw = powerlaw_sizes.std() / powerlaw_sizes.mean()

    print("\nActual CVs achieved:")
    print(f"  Equal: {cv_equal:.3f} (target: 0.000)")
    print(f"  Moderate: {cv_moderate:.3f} (target: ~0.300)")
    print(f"  Power-law: {cv_powerlaw:.3f} (target: >1.000)")

    print("\nCluster size ranges:")
    print(f"  Equal: [{equal_sizes.min()}, {equal_sizes.max()}]")
    print(f"  Moderate: [{moderate_sizes.min()}, {moderate_sizes.max()}], median={np.median(moderate_sizes):.0f}")
    print(f"  Power-law: [{powerlaw_sizes.min()}, {powerlaw_sizes.max()}], median={np.median(powerlaw_sizes):.0f}")

    # Verify
    print("\nVerifying sums:")
    print(f"  Equal: {equal_sizes.sum()} (target: {total_n})")
    print(f"  Moderate: {moderate_sizes.sum()} (target: {total_n})")
    print(f"  Power-law: {powerlaw_sizes.sum()} (target: {total_n})")

    assert equal_sizes.sum() == total_n
    assert moderate_sizes.sum() == total_n
    assert powerlaw_sizes.sum() == total_n

    # Create cluster assignments
    cluster_equal = np.repeat(range(n_clusters_per_scheme), equal_sizes)
    cluster_moderate = np.repeat(range(n_clusters_per_scheme), moderate_sizes)
    cluster_powerlaw = np.repeat(range(n_clusters_per_scheme), powerlaw_sizes)

    # Shuffle
    rng.shuffle(cluster_moderate)
    rng.shuffle(cluster_powerlaw)

    print("\nGenerating outcomes with different ICCs...")

    # Generate outcomes
    continuous_low_icc = generate_continuous_outcome_with_icc(
        equal_sizes, icc=0.05, mean=50000, total_std=20000, seed=seed + 10
    )
    continuous_high_icc = generate_continuous_outcome_with_icc(
        powerlaw_sizes, icc=0.20, mean=75000, total_std=15000, seed=seed + 11
    )
    binary_low_icc = generate_binary_outcome_with_icc(moderate_sizes, icc=0.03, base_prob=0.25, seed=seed + 20)
    binary_high_icc = generate_binary_outcome_with_icc(powerlaw_sizes, icc=0.15, base_prob=0.35, seed=seed + 21)

    # Create DataFrame
    df = pd.DataFrame({
        "participant_id": range(total_n),
        "cluster_equal": cluster_equal,
        "cluster_moderate": cluster_moderate,
        "cluster_powerlaw": cluster_powerlaw,
        "income": continuous_low_icc,
        "test_score": continuous_high_icc,
        "converted": binary_low_icc,
        "engaged": binary_high_icc,
    })

    print(f"\nGenerated {len(df)} participants")

    return df


def generate_ddl(output_path: Path) -> str:
    """
    Generate PostgreSQL DDL file for the clustered data table.

    Args:
        output_path: Path where DDL will be saved

    Returns:
        DDL content as string
    """
    ddl = """-- Clustered test data schema for cluster randomization testing
-- Generated by tools/generate_clustered_data.py

CREATE TABLE {{table_name}} (
    participant_id BIGINT PRIMARY KEY NOT NULL,

    -- Three independent clustering schemes
    cluster_equal INTEGER NOT NULL,      -- Equal-sized clusters (CV ≈ 0)
    cluster_moderate INTEGER NOT NULL,   -- Moderate variation (CV ≈ 0.3)
    cluster_powerlaw INTEGER NOT NULL,   -- Power-law distribution (CV ≈ 3)

    -- Continuous outcomes
    income DECIMAL,                      -- Low ICC (0.05) with cluster_equal
    test_score DECIMAL,                  -- High ICC (0.20) with cluster_powerlaw

    -- Binary outcomes
    converted BOOLEAN,                   -- Low ICC (0.03) with cluster_moderate
    engaged BOOLEAN                      -- High ICC (0.15) with cluster_powerlaw
);

-- Indexes for efficient querying
CREATE INDEX idx_cluster_equal ON {{table_name}}(cluster_equal);
CREATE INDEX idx_cluster_moderate ON {{table_name}}(cluster_moderate);
CREATE INDEX idx_cluster_powerlaw ON {{table_name}}(cluster_powerlaw);
"""

    # Save DDL file
    ddl_path = output_path.with_suffix(".postgres.ddl")
    ddl_path.write_text(ddl)
    print(f"Saved DDL to: {ddl_path}")

    return ddl


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Generate clustered test data for power analysis")
    parser.add_argument("--output", "-o", default="clustered_test_data.csv", help="Output CSV file path")
    parser.add_argument("--n-participants", "-n", type=int, default=10000, help="Total number of participants")
    parser.add_argument("--n-clusters", "-c", type=int, default=100, help="Number of clusters per scheme")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Generate data
    print(f"Generating clustered data with seed={args.seed}...")
    df = generate_multi_cluster_data(
        total_n=args.n_participants,
        n_clusters_per_scheme=args.n_clusters,
        seed=args.seed,
    )

    # Save
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    generate_ddl(output_path)

    # Show summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    for col in ["cluster_equal", "cluster_moderate", "cluster_powerlaw"]:
        sizes = df.groupby(col).size()
        cv = sizes.std() / sizes.mean()
        print(f"\n{col}:")
        print(f"  Clusters: {len(sizes)}")
        print(f"  Sizes: min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.1f}")
        print(f"  CV: {cv:.3f}")

    print("\nOutcomes:")
    for col in ["income", "test_score", "converted", "engaged"]:
        print(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")


if __name__ == "__main__":
    main()
