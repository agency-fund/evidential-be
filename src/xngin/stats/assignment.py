
import numpy as np
import pandas as pd
from stochatreat import stochatreat

from xngin.stats.balance import (
    BalanceResult,
    check_balance_of_preprocessed_df,
    preprocess_for_balance_and_stratification,
    restore_original_numeric_columns,
)

STOCHATREAT_STRATUM_ID_NAME = "stratum_id"
STOCHATREAT_TREAT_NAME = "treat"


def assign_treatment_and_check_balance(
    df: pd.DataFrame,
    stratum_cols: list[str],
    id_col: str,
    n_arms: int,
    quantiles: int = 4,
    random_state: int | None = None,
) -> tuple[list[int], list[int] | None, BalanceResult | None, list[str]]:
    """
    Core assignment logic that operates on a pandas DataFrame.

    Note: Python Decimal types must be converted to float before calling this function.

    Args:
        df: pandas DataFrame containing the data (Decimal types should be converted to float)
        stratum_cols: List of column names to stratify on
        id_col: Name of column containing unit identifiers
        n_arms: Number of arms in your experiment
        quantiles: number of buckets to use for stratification of numerics
        random_state: Random seed for reproducibility

    Returns:
        tuple of (treatment_ids, stratum_ids, balance_result, orig_stratum_cols)

        stratum_ids - list of stratum ids, one for each row in the dataframe if you wish to do any
            post-hoc analyses by stratum
        orig_stratum_cols - deduplicated and sorted list of stratum_cols
    """
    if len(stratum_cols) == 0:
        # No stratification, so use simple random assignment
        treatment_ids = simple_random_assignment(df, n_arms, random_state)
        return treatment_ids, None, None, []

    # Create copy for analysis while attempting to convert any numeric "object" types that pandas
    # didn't originally recognize when creating the dataframe.
    df = df.infer_objects()

    # Dedupe the strata names and then sort them for a stable output ordering
    orig_stratum_cols = sorted(set(stratum_cols))

    orig_data_to_stratify = df[[id_col, *orig_stratum_cols]]
    df_cleaned, exclude_cols_set, numeric_notnull_set = (
        preprocess_for_balance_and_stratification(
            data=orig_data_to_stratify,
            exclude_cols=[id_col],
            quantiles=quantiles,
        )
    )
    # Our original target of columns to stratify on may have gotten smaller:
    post_stratum_cols = sorted(set(orig_stratum_cols) - exclude_cols_set)

    if len(post_stratum_cols) == 0:
        # No stratification, so use simple random assignment while still outputting strata, even
        # though they're either all the same value or all unique values.
        treatment_ids = simple_random_assignment(df, n_arms, random_state)
        return treatment_ids, None, None, orig_stratum_cols

    # Do stratified random assignment
    # TODO: when we support unequal arm assignments, be careful about ensuring the right treatment
    # assignment id is mapped to the right arm_name.
    treatment_status = stochatreat(
        data=df_cleaned,
        idx_col=id_col,
        stratum_cols=post_stratum_cols,
        treats=n_arms,
        probs=[1 / n_arms] * n_arms,
        # internally uses legacy np.random.RandomState which can take None
        random_state=random_state,  # type: ignore[arg-type]
    )
    df_cleaned = df_cleaned.merge(treatment_status, on=id_col)
    stratum_ids = df_cleaned[STOCHATREAT_STRATUM_ID_NAME]
    treatment_ids = df_cleaned[STOCHATREAT_TREAT_NAME]

    # Put back non-null numeric columns for a more robust balance check.
    df_cleaned.drop(columns=[STOCHATREAT_STRATUM_ID_NAME], inplace=True)
    df_cleaned_for_balance_check = restore_original_numeric_columns(
        df_orig=orig_data_to_stratify,
        df_cleaned=df_cleaned,
        numeric_notnull_set=numeric_notnull_set,
    )
    # Explicitly delete to avoid accidental reuse and free memory. Could gc.collect() if needed.
    del orig_data_to_stratify
    del df_cleaned
    # Do balance check with treatment assignments as the dependent var using preprocessed data.
    balance_check_cols = [*post_stratum_cols, STOCHATREAT_TREAT_NAME]
    balance_result = check_balance_of_preprocessed_df(
        df_cleaned_for_balance_check[balance_check_cols],
        treatment_col=STOCHATREAT_TREAT_NAME,
        exclude_col_set=exclude_cols_set,
    )
    del df_cleaned_for_balance_check

    return list(treatment_ids), list(stratum_ids), balance_result, orig_stratum_cols


def simple_random_assignment(
    df: pd.DataFrame,
    n_arms: int,
    random_state: int | None = None,
) -> list[int]:
    """
    Perform simple random assignment of DataFrame rows into the given arms.

    Args:
        df: pandas DataFrame containing the data
        n_arms: Number of arms in your experiment
        random_state: Random seed for reproducibility

    Returns:
        List of treatment ids
    """
    rng = np.random.default_rng(random_state)
    # Create an equal number of treatment ids for each arm and shuffle to ensure arms are as balanced as possible.
    treatment_ids = list(range(n_arms))
    treatment_mask = np.repeat(treatment_ids, np.ceil(len(df) / n_arms))
    rng.shuffle(treatment_mask)
    return [int(x) for x in treatment_mask[: len(df)]]
