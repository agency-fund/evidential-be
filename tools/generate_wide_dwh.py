#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==2.2.3",
#     "typer==0.15.1",
#     "zstandard==0.23.0",
# ]
# ///
"""Generates a wide (500-column) test data warehouse for testing UI behavior with many columns.

Outputs:
  - src/xngin/apiserver/testdata/wide_dwh.csv.zst  (compressed CSV data)
  - src/xngin/apiserver/testdata/wide_dwh.postgres.ddl
  - src/xngin/apiserver/testdata/wide_dwh.redshift.ddl
  - src/xngin/apiserver/testing/wide_dwh_def.py     (ParticipantsDef with FieldDescriptors)
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer
import zstandard

# ---------------------------------------------------------------------------
# Type system: maps our column types to Postgres, Redshift, and DataType enum
# ---------------------------------------------------------------------------

PG_TYPES = {
    "BIGINT": "BIGINT",
    "INTEGER": "INTEGER",
    "BOOLEAN": "BOOLEAN",
    "VARCHAR": "VARCHAR(255)",
    "DECIMAL": "DECIMAL",
    "DOUBLE": "DOUBLE PRECISION",
    "DATE": "DATE",
    "TIMESTAMP": "TIMESTAMP",
    "TIMESTAMPTZ": "TIMESTAMPTZ",
    "UUID": "UUID",
}

RS_TYPES = {
    **PG_TYPES,
    "UUID": "CHAR(36)",  # Redshift has no native UUID type
}

# Maps our shorthand to the DataType enum value used in wide_dwh_def.py
DATATYPE_ENUM = {
    "BIGINT": "DataType.BIGINT",
    "INTEGER": "DataType.INTEGER",
    "BOOLEAN": "DataType.BOOLEAN",
    "VARCHAR": "DataType.CHARACTER_VARYING",
    "DECIMAL": "DataType.NUMERIC",
    "DOUBLE": "DataType.DOUBLE_PRECISION",
    "DATE": "DataType.DATE",
    "TIMESTAMP": "DataType.TIMESTAMP_WITHOUT_TIMEZONE",
    "TIMESTAMPTZ": "DataType.TIMESTAMP_WITH_TIMEZONE",
    "UUID": "DataType.UUID",
}


@dataclass
class ColSpec:
    name: str
    pg_type: str  # key into PG_TYPES / RS_TYPES / DATATYPE_ENUM
    nullable: bool
    is_unique_id: bool = False
    is_filter: bool = False
    is_metric: bool = False
    is_strata: bool = False
    generator: Callable[[np.random.Generator, int], np.ndarray] | None = None


# ---------------------------------------------------------------------------
# Generator helpers
# ---------------------------------------------------------------------------


def gen_sequential_bigint(_rng: np.random.Generator, n: int) -> np.ndarray:
    return np.arange(1, n + 1)


def gen_uuid(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate deterministic UUID-like strings from the RNG."""
    results = []
    for _ in range(n):
        b = rng.integers(0, 256, size=16, dtype=np.uint8)
        h = bytes(b).hex()
        results.append(f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}")
    return np.array(results, dtype=object)


def gen_int(lo: int, hi: int):
    def _gen(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.integers(lo, hi, size=n)

    return _gen


def gen_float(lo: float, hi: float):
    def _gen(rng: np.random.Generator, n: int) -> np.ndarray:
        return np.round(rng.uniform(lo, hi, size=n), 2)

    return _gen


def gen_normal(mean: float, std: float):
    def _gen(rng: np.random.Generator, n: int) -> np.ndarray:
        return np.round(rng.normal(mean, std, size=n), 4)

    return _gen


def gen_bool(p_true: float = 0.5):
    def _gen(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.random(n) < p_true

    return _gen


def gen_choice(choices: list[str]):
    def _gen(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice(choices, size=n)

    return _gen


def gen_date(start_year: int = 2020, end_year: int = 2025):
    def _gen(rng: np.random.Generator, n: int) -> np.ndarray:
        start = np.datetime64(f"{start_year}-01-01")
        end = np.datetime64(f"{end_year}-12-31")
        days = (end - start).astype(int)
        offsets = rng.integers(0, days, size=n)
        dates = start + offsets.astype("timedelta64[D]")
        return np.array([str(d) for d in dates], dtype=object)

    return _gen


def gen_timestamp(start_year: int = 2020, end_year: int = 2025):
    def _gen(rng: np.random.Generator, n: int) -> np.ndarray:
        start = np.datetime64(f"{start_year}-01-01T00:00:00")
        end = np.datetime64(f"{end_year}-12-31T23:59:59")
        seconds = (end - start).astype("timedelta64[s]").astype(int)
        offsets = rng.integers(0, seconds, size=n)
        timestamps = start + offsets.astype("timedelta64[s]")
        return np.array([str(t) for t in timestamps], dtype=object)

    return _gen


def gen_timestamptz(start_year: int = 2020, end_year: int = 2025):
    def _gen(rng: np.random.Generator, n: int) -> np.ndarray:
        start = np.datetime64(f"{start_year}-01-01T00:00:00")
        end = np.datetime64(f"{end_year}-12-31T23:59:59")
        seconds = (end - start).astype("timedelta64[s]").astype(int)
        offsets = rng.integers(0, seconds, size=n)
        timestamps = start + offsets.astype("timedelta64[s]")
        return np.array([str(t) + "+00" for t in timestamps], dtype=object)

    return _gen


# ---------------------------------------------------------------------------
# Cross-column correlation post-processing
# ---------------------------------------------------------------------------


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Rank data without scipy dependency. Returns 0-based float ranks."""
    return np.argsort(np.argsort(arr)).astype(float)


def _induce_rank_correlation(
    data: dict[str, np.ndarray],
    anchor_col: str,
    target_col: str,
    strength: float,
    rng: np.random.Generator,
) -> None:
    """Reorder target_col values so they rank-correlate with anchor_col at ~strength.

    Preserves the marginal distribution of target_col exactly (same sorted values).
    strength=1.0 means perfect rank correlation, strength=0.0 means no change.
    Use negative strength for inverse correlation.
    """
    original_dtype = data[target_col].dtype
    anchor = data[anchor_col].astype(float)
    target = data[target_col].astype(float)
    anchor_ranks = _rankdata(anchor)
    target_ranks = _rankdata(target)
    noise = rng.random(len(anchor_ranks)) * 0.01  # tiebreaker
    blended = abs(strength) * anchor_ranks + (1 - abs(strength)) * target_ranks + noise
    if strength < 0:
        blended = -blended
    reorder = np.argsort(blended)
    result = np.empty_like(target)
    result[reorder] = np.sort(target)
    data[target_col] = result.astype(original_dtype)


def _gen_ar_series(
    rng: np.random.Generator,
    nrows: int,
    n_steps: int,
    base_mean: float,
    base_std: float,
    ar_coeff: float,
    lo: float,
    hi: float,
) -> list[np.ndarray]:
    """Generate n_steps autoregressive columns where each step correlates with the previous."""
    columns: list[np.ndarray] = []
    prev = rng.normal(base_mean, base_std, nrows)
    prev = np.clip(prev, lo, hi)
    for _ in range(n_steps):
        noise = rng.normal(0, base_std * (1 - ar_coeff), nrows)
        prev = ar_coeff * prev + (1 - ar_coeff) * base_mean + noise
        prev = np.clip(prev, lo, hi)
        columns.append(np.round(prev, 2))
    return columns


def _make_retention_monotonic(data: dict[str, np.ndarray], prefix: str, n_steps: int) -> None:
    """Ensure boolean retention columns are per-row monotonic: once False, stays False."""
    for i in range(2, n_steps + 1):
        prev_col = f"{prefix}{i - 1:02d}"
        curr_col = f"{prefix}{i:02d}"
        prev_vals = data[prev_col].astype(bool)
        curr_vals = data[curr_col].astype(bool)
        # If previous week was False, current must also be False
        data[curr_col] = np.where(prev_vals, curr_vals, False)


def apply_correlations(data: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    """Apply cross-column correlations as a post-processing step.

    Modifies data in-place. Called after independent generation but before null injection.
    """

    # --- Cluster 1: Financial health ---
    # Higher income → higher savings, checking, investments, spend, LTV, credit score
    financial_targets = [
        ("savings_balance", 0.7),
        ("checking_balance", 0.6),
        ("investment_portfolio_value", 0.65),
        ("annual_spend", 0.6),
        ("total_lifetime_value", 0.6),
        ("credit_score", 0.6),
    ]
    for target, strength in financial_targets:
        _induce_rank_correlation(data, "household_income", target, strength, rng)
    # Higher income → lower debt ratios
    _induce_rank_correlation(data, "household_income", "debt_to_income_ratio", -0.5, rng)
    _induce_rank_correlation(data, "household_income", "total_debt", -0.3, rng)

    # --- Cluster 2: Engagement intensity ---
    engagement_targets = [
        ("num_sessions_30d", 0.7),
        ("num_pages_viewed_30d", 0.7),
        ("num_actions_30d", 0.7),
        ("total_session_duration_30d", 0.65),
        ("num_logins_90d", 0.6),
    ]
    for target, strength in engagement_targets:
        _induce_rank_correlation(data, "num_logins_30d", target, strength, rng)

    # --- Cluster 3: Revenue cascade ---
    # revenue_90d is the largest, then 30d, then 7d
    rev_90d = data["revenue_90d"].astype(float)
    rev_30d = rev_90d * rng.uniform(0.3, 0.8, len(rev_90d))
    rev_7d = rev_30d * rng.uniform(0.2, 0.6, len(rev_30d))
    data["revenue_30d"] = np.round(rev_30d, 2)
    data["revenue_7d"] = np.round(rev_7d, 2)

    # --- Cluster 4: Conversion → outcomes ---
    converted = data["converted"].astype(bool)
    # Non-converters get near-zero revenue
    for rev_col in ("revenue_7d", "revenue_30d", "revenue_90d"):
        vals = data[rev_col].astype(float)
        vals[~converted] *= 0.05
        data[rev_col] = np.round(vals, 2)
    # Non-converters get fewer clicks
    clicks = data["num_clicks"].astype(float)
    clicks[~converted] *= 0.2
    data["num_clicks"] = clicks.astype(int)
    # Non-converters get fewer total conversions
    convs = data["num_conversions_total"].astype(float)
    convs[~converted] = 0
    data["num_conversions_total"] = convs.astype(int)

    # --- Cluster 5: Time-series autocorrelation ---
    # monthly_spend_m01..m12
    ar_monthly = _gen_ar_series(rng, len(data["id"]), 12, 3000, 2000, 0.7, 50, 10000)
    for i, arr in enumerate(ar_monthly, 1):
        data[f"monthly_spend_m{i:02d}"] = arr

    # ts_value_daily_d01..d10
    ar_daily = _gen_ar_series(rng, len(data["id"]), 10, 500, 200, 0.6, 0, 1000)
    for i, arr in enumerate(ar_daily, 1):
        data[f"ts_value_daily_d{i:02d}"] = arr

    # engagement_score_d01..d10
    ar_engage = _gen_ar_series(rng, len(data["id"]), 10, 50, 20, 0.5, 0, 100)
    for i, arr in enumerate(ar_engage, 1):
        data[f"engagement_score_d{i:02d}"] = arr

    # metric_retention_w01..w10 — make per-row monotonic
    _make_retention_monotonic(data, "metric_retention_w", 10)

    # --- Cluster 6: Propensity/risk model coherence ---
    # High engagement → low churn propensity
    _induce_rank_correlation(data, "num_logins_30d", "propensity_score_churn", -0.5, rng)
    # Bad credit score → high credit risk
    _induce_rank_correlation(data, "credit_score", "risk_score_credit", -0.5, rng)


# ---------------------------------------------------------------------------
# Column definitions — 500 columns across 8 categories
# ---------------------------------------------------------------------------


def build_column_specs() -> list[ColSpec]:
    cols: list[ColSpec] = []

    # ---- Category 1: Identity & Demographics (40 columns) ----
    cols.extend((
        ColSpec("id", "BIGINT", nullable=False, is_unique_id=True, generator=gen_sequential_bigint),
        ColSpec("participant_uuid", "UUID", nullable=False, generator=gen_uuid),
        ColSpec("age", "INTEGER", nullable=True, is_filter=True, is_strata=True, generator=gen_int(18, 90)),
        ColSpec(
            "gender",
            "VARCHAR",
            nullable=True,
            is_filter=True,
            is_strata=True,
            generator=gen_choice(["male", "female", "non_binary", "prefer_not_to_say"]),
        ),
        ColSpec(
            "country_code",
            "VARCHAR",
            nullable=True,
            is_filter=True,
            is_strata=True,
            generator=gen_choice(["US", "GB", "CA", "DE", "FR", "AU", "JP", "BR", "IN", "MX"]),
        ),
        ColSpec(
            "region",
            "VARCHAR",
            nullable=True,
            is_filter=True,
            generator=gen_choice([
                "northeast",
                "southeast",
                "midwest",
                "southwest",
                "west",
                "pacific",
                "mountain",
                "central",
            ]),
        ),
        ColSpec(
            "city",
            "VARCHAR",
            nullable=True,
            generator=gen_choice([
                "new_york",
                "los_angeles",
                "chicago",
                "houston",
                "phoenix",
                "philadelphia",
                "san_antonio",
                "san_diego",
                "dallas",
                "austin",
                "london",
                "toronto",
                "berlin",
                "paris",
                "tokyo",
            ]),
        ),
        ColSpec(
            "postal_code", "VARCHAR", nullable=True, generator=gen_choice([f"{i:05d}" for i in range(10001, 10051)])
        ),
        ColSpec("registration_date", "DATE", nullable=True, is_filter=True, generator=gen_date(2018, 2024)),
        ColSpec("registration_timestamp", "TIMESTAMPTZ", nullable=True, generator=gen_timestamptz(2018, 2024)),
        ColSpec(
            "language_code",
            "VARCHAR",
            nullable=True,
            is_filter=True,
            generator=gen_choice(["en", "es", "fr", "de", "ja", "pt", "zh", "ko"]),
        ),
        ColSpec(
            "education_level",
            "VARCHAR",
            nullable=True,
            is_filter=True,
            is_strata=True,
            generator=gen_choice(["high_school", "bachelors", "masters", "doctorate", "other"]),
        ),
        ColSpec(
            "marital_status",
            "VARCHAR",
            nullable=True,
            is_filter=True,
            generator=gen_choice(["single", "married", "divorced", "widowed"]),
        ),
        ColSpec("birth_date", "DATE", nullable=True, generator=gen_date(1940, 2005)),
        ColSpec("is_active", "BOOLEAN", nullable=True, is_filter=True, generator=gen_bool(0.8)),
        ColSpec("is_verified", "BOOLEAN", nullable=True, is_filter=True, generator=gen_bool(0.7)),
        ColSpec("is_premium", "BOOLEAN", nullable=True, is_filter=True, is_strata=True, generator=gen_bool(0.15)),
        ColSpec("is_internal", "BOOLEAN", nullable=True, generator=gen_bool(0.02)),
        ColSpec(
            "account_type",
            "VARCHAR",
            nullable=True,
            is_filter=True,
            generator=gen_choice(["free", "basic", "pro", "enterprise"]),
        ),
        ColSpec(
            "acquisition_channel",
            "VARCHAR",
            nullable=True,
            is_filter=True,
            is_strata=True,
            generator=gen_choice(["organic", "paid_search", "social", "referral", "email", "direct"]),
        ),
        ColSpec(
            "platform",
            "VARCHAR",
            nullable=True,
            is_filter=True,
            generator=gen_choice(["ios", "android", "web", "desktop"]),
        ),
        ColSpec(
            "device_type",
            "VARCHAR",
            nullable=True,
            is_filter=True,
            generator=gen_choice(["mobile", "tablet", "desktop"]),
        ),
        ColSpec(
            "browser_family",
            "VARCHAR",
            nullable=True,
            generator=gen_choice(["chrome", "safari", "firefox", "edge", "other"]),
        ),
        ColSpec(
            "os_family", "VARCHAR", nullable=True, generator=gen_choice(["windows", "macos", "linux", "ios", "android"])
        ),
        ColSpec("household_size", "INTEGER", nullable=True, generator=gen_int(1, 8)),
        ColSpec("num_children", "INTEGER", nullable=True, generator=gen_int(0, 5)),
        ColSpec(
            "employment_status",
            "VARCHAR",
            nullable=True,
            generator=gen_choice(["employed", "self_employed", "unemployed", "student", "retired"]),
        ),
        ColSpec(
            "industry_sector",
            "VARCHAR",
            nullable=True,
            generator=gen_choice([
                "tech",
                "finance",
                "healthcare",
                "education",
                "retail",
                "manufacturing",
                "government",
            ]),
        ),
        ColSpec("years_as_customer", "INTEGER", nullable=True, generator=gen_int(0, 15)),
        ColSpec("referral_source_id", "BIGINT", nullable=True, generator=gen_int(1, 100000)),
    ))
    # Fill to 40 with more demographics
    for i in range(1, 13):
        cols.append(ColSpec(f"demographic_flag_{i:02d}", "BOOLEAN", nullable=True, generator=gen_bool(0.3 + i * 0.03)))

    # ---- Category 2: Financial / Economic (50 columns) ----
    cols.extend((
        ColSpec(
            "household_income",
            "DECIMAL",
            nullable=True,
            is_filter=True,
            is_strata=True,
            generator=gen_float(15000, 250000),
        ),
        ColSpec("credit_score", "INTEGER", nullable=True, is_filter=True, is_strata=True, generator=gen_int(300, 850)),
        ColSpec("savings_balance", "DECIMAL", nullable=True, is_metric=True, generator=gen_float(0, 500000)),
        ColSpec("checking_balance", "DECIMAL", nullable=True, generator=gen_float(0, 50000)),
        ColSpec("total_debt", "DECIMAL", nullable=True, generator=gen_float(0, 300000)),
        ColSpec("debt_to_income_ratio", "DECIMAL", nullable=True, is_filter=True, generator=gen_float(0, 2.5)),
        ColSpec("annual_spend", "DECIMAL", nullable=True, is_metric=True, generator=gen_float(500, 100000)),
        ColSpec("avg_transaction_value", "DECIMAL", nullable=True, is_metric=True, generator=gen_float(5, 500)),
        ColSpec("num_transactions_30d", "INTEGER", nullable=True, is_metric=True, generator=gen_int(0, 200)),
        ColSpec("num_transactions_90d", "INTEGER", nullable=True, generator=gen_int(0, 600)),
    ))
    for m in range(1, 13):
        cols.append(
            ColSpec(
                f"monthly_spend_m{m:02d}", "DECIMAL", nullable=True, is_metric=(m <= 3), generator=gen_float(50, 10000)
            )
        )
    cols.extend((
        ColSpec("total_lifetime_value", "DECIMAL", nullable=True, is_metric=True, generator=gen_float(100, 500000)),
        ColSpec("avg_order_value", "DECIMAL", nullable=True, generator=gen_float(10, 1000)),
        ColSpec("num_refunds", "INTEGER", nullable=True, generator=gen_int(0, 20)),
        ColSpec("refund_amount_total", "DECIMAL", nullable=True, generator=gen_float(0, 5000)),
        ColSpec(
            "payment_method_primary",
            "VARCHAR",
            nullable=True,
            is_filter=True,
            generator=gen_choice(["credit_card", "debit_card", "paypal", "bank_transfer", "crypto"]),
        ),
        ColSpec("has_subscription", "BOOLEAN", nullable=True, is_filter=True, generator=gen_bool(0.4)),
        ColSpec(
            "subscription_tier",
            "VARCHAR",
            nullable=True,
            generator=gen_choice(["none", "basic", "standard", "premium"]),
        ),
        ColSpec("subscription_monthly_fee", "DECIMAL", nullable=True, generator=gen_float(0, 99.99)),
        ColSpec("first_purchase_date", "DATE", nullable=True, generator=gen_date(2018, 2024)),
        ColSpec("last_purchase_date", "DATE", nullable=True, generator=gen_date(2023, 2025)),
        ColSpec("days_since_last_purchase", "INTEGER", nullable=True, generator=gen_int(0, 365)),
        ColSpec("investment_portfolio_value", "DECIMAL", nullable=True, generator=gen_float(0, 1000000)),
        ColSpec("mortgage_balance", "DECIMAL", nullable=True, generator=gen_float(0, 500000)),
        ColSpec("auto_loan_balance", "DECIMAL", nullable=True, generator=gen_float(0, 60000)),
        ColSpec("student_loan_balance", "DECIMAL", nullable=True, generator=gen_float(0, 200000)),
        ColSpec("insurance_premium_monthly", "DECIMAL", nullable=True, generator=gen_float(50, 2000)),
        ColSpec(
            "tax_bracket",
            "VARCHAR",
            nullable=True,
            generator=gen_choice(["10pct", "12pct", "22pct", "24pct", "32pct", "35pct", "37pct"]),
        ),
        ColSpec("num_credit_cards", "INTEGER", nullable=True, generator=gen_int(0, 10)),
        ColSpec("credit_utilization_ratio", "DECIMAL", nullable=True, generator=gen_float(0, 1)),
    ))
    # Correlated financial columns
    for i in range(1, 11):
        cols.append(ColSpec(f"financial_metric_{i:02d}", "DECIMAL", nullable=True, generator=gen_float(0, 10000)))

    # ---- Category 3: Behavioral / Engagement (100 columns) ----
    cols.extend((
        ColSpec("num_logins_30d", "INTEGER", nullable=True, is_metric=True, generator=gen_int(0, 100)),
        ColSpec("num_logins_90d", "INTEGER", nullable=True, generator=gen_int(0, 300)),
        ColSpec("days_since_last_login", "INTEGER", nullable=True, is_filter=True, generator=gen_int(0, 365)),
        ColSpec("total_session_duration_30d", "DECIMAL", nullable=True, is_metric=True, generator=gen_float(0, 36000)),
        ColSpec("avg_session_duration_30d", "DECIMAL", nullable=True, generator=gen_float(0, 3600)),
        ColSpec("num_sessions_30d", "INTEGER", nullable=True, generator=gen_int(0, 200)),
        ColSpec("num_pages_viewed_30d", "INTEGER", nullable=True, is_metric=True, generator=gen_int(0, 5000)),
        ColSpec("num_actions_30d", "INTEGER", nullable=True, generator=gen_int(0, 2000)),
        ColSpec("last_login_timestamp", "TIMESTAMPTZ", nullable=True, generator=gen_timestamptz(2024, 2025)),
        ColSpec("first_login_timestamp", "TIMESTAMPTZ", nullable=True, generator=gen_timestamptz(2018, 2024)),
    ))
    features = [
        "search",
        "checkout",
        "profile",
        "settings",
        "help",
        "notifications",
        "dashboard",
        "reports",
        "export",
        "share",
    ]
    for feat in features:
        cols.append(
            ColSpec(f"pct_sessions_with_feature_{feat}_30d", "DECIMAL", nullable=True, generator=gen_float(0, 1))
        )
    for feat in features:
        cols.append(ColSpec(f"num_uses_feature_{feat}_30d", "INTEGER", nullable=True, generator=gen_int(0, 500)))
    for w in range(1, 11):
        cols.append(
            ColSpec(f"weekly_active_w{w:02d}", "BOOLEAN", nullable=True, is_metric=(w <= 2), generator=gen_bool(0.6))
        )
    cols.extend((
        ColSpec("bounce_rate_30d", "DECIMAL", nullable=True, generator=gen_float(0, 1)),
        ColSpec("click_through_rate_30d", "DECIMAL", nullable=True, is_metric=True, generator=gen_float(0, 0.5)),
        ColSpec("scroll_depth_avg", "DECIMAL", nullable=True, generator=gen_float(0, 1)),
        ColSpec("num_errors_encountered_30d", "INTEGER", nullable=True, generator=gen_int(0, 50)),
        ColSpec("num_support_tickets_30d", "INTEGER", nullable=True, generator=gen_int(0, 10)),
        ColSpec("num_feedback_submitted", "INTEGER", nullable=True, generator=gen_int(0, 20)),
        ColSpec("nps_score", "INTEGER", nullable=True, is_metric=True, generator=gen_int(0, 10)),
        ColSpec("satisfaction_score", "DECIMAL", nullable=True, is_metric=True, generator=gen_float(1, 5)),
    ))
    # Engagement time-series
    for d in range(1, 11):
        cols.append(ColSpec(f"engagement_score_d{d:02d}", "DECIMAL", nullable=True, generator=gen_float(0, 100)))
    for w in range(1, 11):
        cols.append(ColSpec(f"engagement_count_w{w:02d}", "INTEGER", nullable=True, generator=gen_int(0, 200)))
    # Additional engagement metrics
    cols.extend((
        ColSpec("avg_time_between_sessions_hrs", "DECIMAL", nullable=True, generator=gen_float(0, 720)),
        ColSpec("max_session_duration_30d", "DECIMAL", nullable=True, generator=gen_float(0, 7200)),
        ColSpec("num_unique_pages_30d", "INTEGER", nullable=True, generator=gen_int(0, 500)),
        ColSpec("num_downloads_30d", "INTEGER", nullable=True, generator=gen_int(0, 50)),
        ColSpec("num_uploads_30d", "INTEGER", nullable=True, generator=gen_int(0, 30)),
        ColSpec("num_shares_30d", "INTEGER", nullable=True, generator=gen_int(0, 100)),
        ColSpec("num_comments_30d", "INTEGER", nullable=True, generator=gen_int(0, 200)),
        ColSpec("num_likes_30d", "INTEGER", nullable=True, generator=gen_int(0, 500)),
        ColSpec("num_bookmarks_30d", "INTEGER", nullable=True, generator=gen_int(0, 100)),
        ColSpec("has_completed_onboarding", "BOOLEAN", nullable=True, is_filter=True, generator=gen_bool(0.75)),
        ColSpec("onboarding_completion_date", "DATE", nullable=True, generator=gen_date(2020, 2024)),
        ColSpec("num_notifications_sent_30d", "INTEGER", nullable=True, generator=gen_int(0, 100)),
        ColSpec("num_notifications_opened_30d", "INTEGER", nullable=True, generator=gen_int(0, 80)),
        ColSpec("notification_click_rate", "DECIMAL", nullable=True, generator=gen_float(0, 1)),
        ColSpec("email_open_rate_30d", "DECIMAL", nullable=True, generator=gen_float(0, 1)),
        ColSpec("email_click_rate_30d", "DECIMAL", nullable=True, generator=gen_float(0, 0.5)),
        ColSpec("num_social_connections", "INTEGER", nullable=True, generator=gen_int(0, 1000)),
        ColSpec("has_profile_photo", "BOOLEAN", nullable=True, generator=gen_bool(0.6)),
        ColSpec("profile_completeness_pct", "DECIMAL", nullable=True, generator=gen_float(0, 1)),
        ColSpec("num_saved_items", "INTEGER", nullable=True, generator=gen_int(0, 200)),
        ColSpec("num_wishlists", "INTEGER", nullable=True, generator=gen_int(0, 10)),
    ))
    # Fill remaining to reach 100
    for i in range(1, 9):
        cols.append(ColSpec(f"behavior_flag_{i:02d}", "BOOLEAN", nullable=True, generator=gen_bool(0.4)))

    # ---- Category 4: A/B Test Outcomes (80 columns) ----
    cols.extend((
        ColSpec("converted", "BOOLEAN", nullable=True, is_metric=True, generator=gen_bool(0.12)),
        ColSpec("num_clicks", "INTEGER", nullable=True, is_metric=True, generator=gen_int(0, 500)),
        ColSpec("num_impressions", "INTEGER", nullable=True, generator=gen_int(0, 10000)),
        ColSpec("revenue_7d", "DECIMAL", nullable=True, is_metric=True, generator=gen_float(0, 5000)),
        ColSpec("revenue_30d", "DECIMAL", nullable=True, is_metric=True, generator=gen_float(0, 20000)),
        ColSpec("revenue_90d", "DECIMAL", nullable=True, generator=gen_float(0, 50000)),
        ColSpec("predicted_retention_days", "DECIMAL", nullable=True, generator=gen_float(0, 365)),
        ColSpec("churn_probability", "DECIMAL", nullable=True, is_metric=True, generator=gen_float(0, 1)),
        ColSpec("conversion_timestamp", "TIMESTAMPTZ", nullable=True, generator=gen_timestamptz(2024, 2025)),
        ColSpec("conversion_date", "DATE", nullable=True, generator=gen_date(2024, 2025)),
    ))
    for w in range(1, 11):
        cols.append(
            ColSpec(
                f"metric_retention_w{w:02d}",
                "BOOLEAN",
                nullable=True,
                is_metric=(w <= 4),
                generator=gen_bool(0.7 - w * 0.04),
            )
        )
    for w in range(1, 11):
        cols.append(ColSpec(f"metric_revenue_w{w:02d}", "DECIMAL", nullable=True, generator=gen_float(0, 2000)))
    cols.extend((
        ColSpec("first_conversion_date", "DATE", nullable=True, generator=gen_date(2023, 2025)),
        ColSpec("num_conversions_total", "INTEGER", nullable=True, generator=gen_int(0, 50)),
        ColSpec("conversion_value_total", "DECIMAL", nullable=True, generator=gen_float(0, 100000)),
        ColSpec("time_to_convert_hours", "DECIMAL", nullable=True, generator=gen_float(0, 720)),
        ColSpec("goal_completion_rate", "DECIMAL", nullable=True, generator=gen_float(0, 1)),
    ))
    # Outcome metrics per variant
    for v in range(1, 6):
        for metric in ["clicks", "views", "conversions", "revenue"]:
            cols.append(
                ColSpec(f"outcome_{metric}_variant_{v}", "DECIMAL", nullable=True, generator=gen_float(0, 1000))
            )
    # Outcome flags
    for i in range(1, 16):
        cols.append(ColSpec(f"outcome_flag_{i:02d}", "BOOLEAN", nullable=True, generator=gen_bool(0.25)))
    # Additional outcome metrics
    cols.extend((
        ColSpec("engagement_lift_pct", "DECIMAL", nullable=True, generator=gen_float(-50, 200)),
        ColSpec("revenue_lift_pct", "DECIMAL", nullable=True, generator=gen_float(-30, 100)),
        ColSpec("retention_lift_pct", "DECIMAL", nullable=True, generator=gen_float(-20, 50)),
        ColSpec("activation_score", "DECIMAL", nullable=True, generator=gen_float(0, 100)),
        ColSpec("days_to_first_action", "INTEGER", nullable=True, generator=gen_int(0, 90)),
        ColSpec("num_goal_completions", "INTEGER", nullable=True, generator=gen_int(0, 100)),
        ColSpec("avg_order_frequency_30d", "DECIMAL", nullable=True, generator=gen_float(0, 30)),
        ColSpec("customer_effort_score", "DECIMAL", nullable=True, generator=gen_float(1, 7)),
        ColSpec("page_load_time_avg_ms", "INTEGER", nullable=True, generator=gen_int(100, 5000)),
        ColSpec("interaction_depth_score", "DECIMAL", nullable=True, generator=gen_float(0, 10)),
        ColSpec("repeat_visit_rate", "DECIMAL", nullable=True, generator=gen_float(0, 1)),
        ColSpec("viral_coefficient", "DECIMAL", nullable=True, generator=gen_float(0, 3)),
    ))

    # ---- Category 5: Experiment Assignments (50 columns) ----
    cols.extend((
        ColSpec(
            "treatment_group_primary",
            "VARCHAR",
            nullable=True,
            is_filter=True,
            is_strata=True,
            generator=gen_choice(["control", "treatment_a", "treatment_b"]),
        ),
        ColSpec(
            "treatment_group_secondary",
            "VARCHAR",
            nullable=True,
            is_filter=True,
            generator=gen_choice(["control", "variant_1", "variant_2", "variant_3"]),
        ),
        ColSpec("assignment_date", "DATE", nullable=True, is_filter=True, generator=gen_date(2024, 2025)),
        ColSpec("assignment_timestamp", "TIMESTAMPTZ", nullable=True, generator=gen_timestamptz(2024, 2025)),
        ColSpec(
            "assignment_method",
            "VARCHAR",
            nullable=True,
            generator=gen_choice(["random", "stratified", "adaptive", "preassigned"]),
        ),
        ColSpec("randomization_seed", "BIGINT", nullable=True, generator=gen_int(1, 2**31)),
    ))
    for exp_num in range(1, 11):
        exp_id = f"exp_{exp_num:03d}"
        cols.extend((
            ColSpec(
                f"experiment_arm_{exp_id}",
                "VARCHAR",
                nullable=True,
                generator=gen_choice(["control", "treatment_a", "treatment_b", "holdout"]),
            ),
            ColSpec(f"experiment_enrolled_{exp_id}", "BOOLEAN", nullable=True, generator=gen_bool(0.5)),
            ColSpec(f"experiment_enrolled_date_{exp_id}", "DATE", nullable=True, generator=gen_date(2024, 2025)),
        ))
    # More assignment metadata
    cols.extend((
        ColSpec("is_holdout", "BOOLEAN", nullable=True, is_filter=True, generator=gen_bool(0.05)),
        ColSpec("holdout_group_id", "INTEGER", nullable=True, generator=gen_int(0, 10)),
        ColSpec("assignment_weight", "DECIMAL", nullable=True, generator=gen_float(0, 1)),
        ColSpec("stratification_bucket", "INTEGER", nullable=True, generator=gen_int(1, 20)),
        ColSpec("consent_to_experiment", "BOOLEAN", nullable=True, generator=gen_bool(0.9)),
        ColSpec(
            "experiment_cohort",
            "VARCHAR",
            nullable=True,
            generator=gen_choice(["cohort_a", "cohort_b", "cohort_c", "cohort_d"]),
        ),
        ColSpec("experiment_start_date", "DATE", nullable=True, generator=gen_date(2024, 2025)),
        ColSpec("experiment_end_date", "DATE", nullable=True, generator=gen_date(2024, 2025)),
        ColSpec("experiment_duration_days", "INTEGER", nullable=True, generator=gen_int(7, 90)),
        ColSpec("assignment_confidence", "DECIMAL", nullable=True, generator=gen_float(0, 1)),
        ColSpec("is_eligible", "BOOLEAN", nullable=True, is_filter=True, generator=gen_bool(0.85)),
        ColSpec(
            "eligibility_reason",
            "VARCHAR",
            nullable=True,
            generator=gen_choice(["qualified", "age_range", "geo_match", "behavior_match", "random"]),
        ),
        ColSpec("experiment_priority", "INTEGER", nullable=True, generator=gen_int(1, 5)),
        ColSpec("assignment_version", "INTEGER", nullable=True, generator=gen_int(1, 3)),
    ))

    # ---- Category 6: Predictive Model Features (80 columns) ----
    propensity_targets = [
        "purchase",
        "churn",
        "upgrade",
        "downgrade",
        "referral",
        "support_contact",
        "return_visit",
        "high_value",
    ]
    for target in propensity_targets:
        cols.append(
            ColSpec(
                f"propensity_score_{target}",
                "DECIMAL",
                nullable=True,
                is_metric=(target in {"purchase", "churn"}),
                generator=gen_float(0, 1),
            )
        )
    risk_targets = ["credit", "fraud", "attrition", "default", "late_payment"]
    for target in risk_targets:
        cols.append(ColSpec(f"risk_score_{target}", "DECIMAL", nullable=True, generator=gen_float(0, 1)))
    for i in range(1, 21):
        cols.append(ColSpec(f"feature_numeric_{i:03d}", "DECIMAL", nullable=True, generator=gen_normal(0, 1)))
    for i in range(1, 11):
        cols.append(
            ColSpec(
                f"feature_categorical_{i:03d}",
                "VARCHAR",
                nullable=True,
                generator=gen_choice([f"cat_{j}" for j in range(1, 8)]),
            )
        )
    for i in range(1, 11):
        cols.append(ColSpec(f"feature_binary_{i:03d}", "BOOLEAN", nullable=True, generator=gen_bool(0.5)))
    cols.extend((
        ColSpec(
            "model_version", "VARCHAR", nullable=True, generator=gen_choice(["v1.0", "v1.1", "v2.0", "v2.1", "v3.0"])
        ),
        ColSpec("model_score_timestamp", "TIMESTAMPTZ", nullable=True, generator=gen_timestamptz(2024, 2025)),
        ColSpec("model_confidence", "DECIMAL", nullable=True, generator=gen_float(0, 1)),
    ))
    # Embeddings / dense features
    for i in range(1, 11):
        cols.append(ColSpec(f"embedding_dim_{i:03d}", "DOUBLE", nullable=True, generator=gen_normal(0, 1)))
    # Additional model features
    for i in range(1, 6):
        cols.append(ColSpec(f"model_feature_interaction_{i:02d}", "DECIMAL", nullable=True, generator=gen_normal(0, 2)))
    cols.extend((
        ColSpec("prediction_date", "DATE", nullable=True, generator=gen_date(2024, 2025)),
        ColSpec(
            "model_segment",
            "VARCHAR",
            nullable=True,
            generator=gen_choice(["high_value", "medium_value", "low_value", "at_risk", "new"]),
        ),
        ColSpec("anomaly_score", "DECIMAL", nullable=True, generator=gen_float(0, 1)),
        ColSpec("cluster_id", "INTEGER", nullable=True, generator=gen_int(0, 20)),
        ColSpec("nearest_neighbor_distance", "DOUBLE", nullable=True, generator=gen_float(0, 10)),
        ColSpec("feature_importance_rank", "INTEGER", nullable=True, generator=gen_int(1, 100)),
        ColSpec("model_calibration_score", "DECIMAL", nullable=True, generator=gen_float(0, 1)),
    ))

    # ---- Category 7: Temporal / Time-Series (80 columns) ----
    for d in range(1, 11):
        cols.append(ColSpec(f"ts_value_daily_d{d:02d}", "DECIMAL", nullable=True, generator=gen_float(0, 1000)))
    for d in range(1, 11):
        cols.append(ColSpec(f"ts_count_daily_d{d:02d}", "INTEGER", nullable=True, generator=gen_int(0, 500)))
    for w in range(1, 11):
        cols.append(ColSpec(f"ts_count_weekly_w{w:02d}", "INTEGER", nullable=True, generator=gen_int(0, 2000)))
    for w in range(1, 11):
        cols.append(ColSpec(f"ts_value_weekly_w{w:02d}", "DECIMAL", nullable=True, generator=gen_float(0, 5000)))
    for m in range(1, 11):
        cols.append(ColSpec(f"trend_indicator_m{m:02d}", "BOOLEAN", nullable=True, generator=gen_bool(0.5)))
    for m in range(1, 11):
        cols.append(ColSpec(f"ts_avg_monthly_m{m:02d}", "DECIMAL", nullable=True, generator=gen_float(0, 10000)))
    # Seasonal and rolling metrics
    for q in range(1, 5):
        cols.append(ColSpec(f"seasonal_index_q{q}", "DECIMAL", nullable=True, generator=gen_float(0.5, 1.5)))
    for w in range(1, 7):
        cols.append(ColSpec(f"rolling_avg_{w:02d}w", "DECIMAL", nullable=True, generator=gen_float(0, 5000)))
    # Year-over-year and momentum
    for m in range(1, 7):
        cols.append(ColSpec(f"yoy_growth_m{m:02d}", "DECIMAL", nullable=True, generator=gen_float(-1, 3)))
    for m in range(1, 5):
        cols.append(ColSpec(f"momentum_indicator_m{m:02d}", "DECIMAL", nullable=True, generator=gen_float(-5, 5)))

    # ---- Category 8: Metadata (20 columns) ----
    cols.extend((
        ColSpec(
            "source_system",
            "VARCHAR",
            nullable=True,
            generator=gen_choice(["crm", "web_analytics", "mobile_sdk", "data_pipeline", "manual_import"]),
        ),
        ColSpec("audit_trail_uuid", "UUID", nullable=True, generator=gen_uuid),
        ColSpec("last_updated_timestamp", "TIMESTAMPTZ", nullable=False, generator=gen_timestamptz(2024, 2025)),
        ColSpec("created_timestamp", "TIMESTAMPTZ", nullable=True, generator=gen_timestamptz(2020, 2024)),
        ColSpec("etl_batch_id", "BIGINT", nullable=True, generator=gen_int(1, 10000)),
        ColSpec("data_quality_score", "DECIMAL", nullable=True, generator=gen_float(0, 1)),
        ColSpec("metadata_version", "INTEGER", nullable=True, generator=gen_int(1, 5)),
        ColSpec("is_test_record", "BOOLEAN", nullable=True, generator=gen_bool(0.01)),
        ColSpec(
            "record_status",
            "VARCHAR",
            nullable=True,
            generator=gen_choice(["active", "inactive", "deleted", "archived"]),
        ),
        ColSpec("consent_given", "BOOLEAN", nullable=True, generator=gen_bool(0.95)),
        ColSpec("consent_date", "DATE", nullable=True, generator=gen_date(2020, 2024)),
        ColSpec(
            "privacy_level", "VARCHAR", nullable=True, generator=gen_choice(["public", "restricted", "confidential"])
        ),
        ColSpec("data_source_version", "VARCHAR", nullable=True, generator=gen_choice(["v1", "v2", "v3"])),
        ColSpec("ingestion_timestamp", "TIMESTAMPTZ", nullable=True, generator=gen_timestamptz(2024, 2025)),
        ColSpec("row_hash", "VARCHAR", nullable=True, generator=gen_choice([f"hash_{i:04d}" for i in range(100)])),
        ColSpec("partition_key", "INTEGER", nullable=True, generator=gen_int(0, 99)),
        ColSpec("processing_flag", "BOOLEAN", nullable=True, generator=gen_bool(0.1)),
        ColSpec(
            "notes",
            "VARCHAR",
            nullable=True,
            generator=gen_choice(["", "flagged", "reviewed", "needs_attention", "ok"]),
        ),
        ColSpec(
            "tag_primary",
            "VARCHAR",
            nullable=True,
            generator=gen_choice(["alpha", "beta", "gamma", "delta", "epsilon"]),
        ),
        ColSpec(
            "tag_secondary",
            "VARCHAR",
            nullable=True,
            generator=gen_choice(["red", "blue", "green", "yellow", "orange"]),
        ),
    ))

    # Apply defaults: 75% of eligible columns (excluding timestamps, UUIDs, and
    # the primary key) get is_metric, is_filter, and is_strata.
    excluded_types = {"TIMESTAMP", "TIMESTAMPTZ", "UUID"}
    eligible = [c for c in cols if c.pg_type not in excluded_types and not c.is_unique_id]
    target_count = int(len(eligible) * 0.75)
    for flag in ("is_metric", "is_filter", "is_strata"):
        already = sum(1 for c in eligible if getattr(c, flag))
        remaining = [c for c in eligible if not getattr(c, flag)]
        need = target_count - already
        for col in remaining[:need]:
            setattr(col, flag, True)

    assert sum(c.is_filter for c in cols) >= 300, f"Expected >=300 filters, got {sum(c.is_filter for c in cols)}"
    assert sum(c.is_metric for c in cols) >= 300, f"Expected >=300 metrics, got {sum(c.is_metric for c in cols)}"
    assert sum(c.is_strata for c in cols) >= 200, f"Expected >=200 strata, got {sum(c.is_strata for c in cols)}"

    return cols


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTDATA_DIR = REPO_ROOT / "src" / "xngin" / "apiserver" / "testdata"
TESTING_DIR = REPO_ROOT / "src" / "xngin" / "apiserver" / "testing"

app = typer.Typer()


@app.command()
def main(
    nrows: int = typer.Option(1000, help="Number of rows to generate."),
    seed: int = typer.Option(42, help="Random seed for reproducibility."),
):
    cols = build_column_specs()
    print(f"Defined {len(cols)} columns.")
    assert len(cols) == 500, f"Expected 500 columns, got {len(cols)}"

    # Verify unique names
    names = [c.name for c in cols]
    assert len(names) == len(set(names)), f"Duplicate column names: {[n for n in names if names.count(n) > 1]}"

    rng = np.random.default_rng(seed)

    # Phase 1: Generate all columns independently
    print(f"Generating {nrows} rows...")
    data: dict[str, np.ndarray] = {}
    for col in cols:
        if col.generator is None:
            raise ValueError(f"No generator for column {col.name}")
        data[col.name] = col.generator(rng, nrows)

    # Phase 2: Apply cross-column correlations
    print("Applying cross-column correlations...")
    apply_correlations(data, rng)

    # Phase 3: Inject nulls
    null_rate = 0.03  # ~3% nulls for nullable columns
    for col in cols:
        if col.nullable:
            values = data[col.name]
            null_mask = rng.random(nrows) < null_rate
            if values.dtype.kind in {"U", "O"}:  # string/object arrays
                values = values.copy()
                values[null_mask] = None
            else:
                values = values.astype(object)
                values[null_mask] = None
            data[col.name] = values

    # Write CSV → zstd
    csv_path = TESTDATA_DIR / "wide_dwh.csv.zst"
    print(f"Writing {csv_path}...")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(names)
    for i in range(nrows):
        row = []
        for col in cols:
            val = data[col.name][i]
            if val is None:
                row.append("")
            elif isinstance(val, (bool, np.bool_)):
                row.append("true" if val else "false")
            elif isinstance(val, (np.integer, int)):
                row.append(str(int(val)))
            elif isinstance(val, (np.floating, float)):
                row.append(str(float(val)))
            else:
                row.append(str(val))
        writer.writerow(row)

    csv_bytes = buf.getvalue().encode("utf-8")
    cctx = zstandard.ZstdCompressor(level=19)
    compressed = cctx.compress(csv_bytes)
    csv_path.write_bytes(compressed)
    print(f"  Uncompressed: {len(csv_bytes):,} bytes")
    print(f"  Compressed:   {len(compressed):,} bytes")

    # Write Postgres DDL
    pg_ddl_path = TESTDATA_DIR / "wide_dwh.postgres.ddl"
    print(f"Writing {pg_ddl_path}...")
    pg_lines = []
    for col in cols:
        pg_type = PG_TYPES[col.pg_type]
        extras = []
        if col.is_unique_id:
            extras.append("PRIMARY KEY")
        if not col.nullable:
            extras.append("NOT NULL")
        line = f"  {col.name} {pg_type}"
        if extras:
            line += " " + " ".join(extras)
        pg_lines.append(line)
    pg_ddl = "CREATE TABLE {{table_name}} (\n" + ",\n".join(pg_lines) + "\n);\n"
    pg_ddl_path.write_text(pg_ddl)

    # Write Redshift DDL
    rs_ddl_path = TESTDATA_DIR / "wide_dwh.redshift.ddl"
    print(f"Writing {rs_ddl_path}...")
    rs_lines = []
    for col in cols:
        rs_type = RS_TYPES[col.pg_type]
        extras = []
        # Redshift: no PRIMARY KEY
        if not col.nullable:
            extras.append("NOT NULL")
        line = f"  {col.name} {rs_type}"
        if extras:
            line += " " + " ".join(extras)
        rs_lines.append(line)
    rs_ddl = "CREATE TABLE {{table_name}} (\n" + ",\n".join(rs_lines) + "\n);\n"
    rs_ddl_path.write_text(rs_ddl)

    # Write wide_dwh_def.py
    def_path = TESTING_DIR / "wide_dwh_def.py"
    print(f"Writing {def_path}...")
    field_lines = []
    for col in cols:
        dt_enum = DATATYPE_ENUM[col.pg_type]
        kwargs = [
            f'field_name="{col.name}"',
            f"data_type={dt_enum}",
        ]
        if col.is_unique_id:
            kwargs.append("is_unique_id=True")
        if col.is_filter:
            kwargs.append("is_filter=True")
        if col.is_metric:
            kwargs.append("is_metric=True")
        if col.is_strata:
            kwargs.append("is_strata=True")
        field_lines.append(f"        FieldDescriptor({', '.join(kwargs)}),")

    def_content = textwrap.dedent('''\
        # This file was generated by tools/generate_wide_dwh.py
        """Definitions for the wide (500-column) testing data warehouse.

        WIDE_DWH_RAW_DATA is the path to the compressed raw wide dwh data.
        WIDE_DWH_PARTICIPANT_DEF is defined with respect to these contents and has
        fields that map to our wide test dwh, covering all supported data types.
        """

        from pathlib import Path

        from xngin.apiserver.dwh.inspection_types import FieldDescriptor
        from xngin.apiserver.routers.common_enums import DataType
        from xngin.apiserver.settings import ParticipantsDef

        WIDE_DWH_RAW_DATA = Path(__file__).resolve().parent.parent / "testdata/wide_dwh.csv.zst"

        WIDE_DWH_PARTICIPANT_DEF = ParticipantsDef(
            type="schema",
            participant_type="wide_test_participant_type",
            table_name="wide_dwh",
            fields=[
    ''')
    def_content += "\n".join(field_lines) + "\n"
    def_content += "    ],\n)\n"

    def_path.write_text(def_content)
    try:
        subprocess.run(
            [(shutil.which("uv")), "run", "--project", str(REPO_ROOT), "ruff", "format", str(def_path)],
            check=True,
            env={"VIRTUAL_ENV": ""},
        )
    except subprocess.SubprocessError as err:
        print(f"Failed to format: {err}")

    print(f"\nDone! Generated {len(cols)} columns x {nrows} rows.")


if __name__ == "__main__":
    app()
