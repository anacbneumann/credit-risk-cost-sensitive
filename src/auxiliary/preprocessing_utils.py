# ==============================
# preprocessing_utils.py
# ==============================
# Summary:
#   Utility functions for the credit risk preprocessing pipeline.
#   Covers logging, column renaming, deduplication, missingness flags,
#   outlier capping, log transforms, and rule-based anomaly detection.
#   All functions here are pure data-cleaning operations — no analytical
#   feature engineering. Derived feature builders live in feature_utils.py.
#
#   Dataset context: GiveMeSomeCredit (Kaggle) — binary classification for
#   predicting serious delinquency within two years.
#
# Typical usage:
#   from preprocessing_utils import (
#       rename_credit_columns, create_missing_flags,
#       flag_util_outliers, flag_and_clean_dpd,
#       median_impute, cap_dependents,
#   )
# ==============================

# ── Standard library ──────────────────────────────────────────────────────────
import logging

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
_logger = logging.getLogger(__name__)


# ==============================================================================
# Logging helpers
# ==============================================================================

def log_info(msg: str) -> None:
    """Log an informational message with a timestamp.

    Summary:
        Thin wrapper around the standard library logger that emits an INFO-level
        message.

    Args:
        msg: Human-readable message to be logged.

    Returns:
        None
    """
    _logger.info(msg)


def log_warn(msg: str) -> None:
    """Log a warning message with a timestamp.

    Summary:
        Thin wrapper around the standard library logger that emits a WARNING-level
        message.

    Args:
        msg: Human-readable warning message to be logged.

    Returns:
        None
    """
    _logger.warning(msg)


# ==============================================================================
# Column renaming utilities
# ==============================================================================

def safe_rename(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    """Rename DataFrame columns using a mapping, silently skipping missing keys.

    Summary:
        Only columns present in df are renamed; columns absent from the mapping
        are left untouched and no error is raised.

    Args:
        df: Input DataFrame.
        rename_map: Dictionary mapping old column names to new column names,
            e.g. {"OldName": "new_name"}.

    Returns:
        DataFrame with applicable columns renamed. Original is not modified.
    """
    present = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=present)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has an id column (best-effort).

    Summary:
        If no column named "id" already exists, the first column is renamed
        to "id". Mirrors the R implementation's fallback behaviour.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame where df.columns[0] is renamed to "id" when an "id" column
        was not already present.
    """
    if "id" not in df.columns:
        new_cols = list(df.columns)
        new_cols[0] = "id"
        df = df.copy()
        df.columns = new_cols
    return df


def rename_credit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw GiveMeSomeCredit Kaggle column names to snake_case equivalents.

    Summary:
        Applies a fixed mapping from the original Kaggle column names (PascalCase
        / mixed) to concise, consistent snake_case names. Emits a warning listing
        any expected columns absent after the rename.

    Args:
        df: Input DataFrame containing original Kaggle column names.

    Returns:
        DataFrame with standardized snake_case column names where applicable.
        Columns not present in the mapping are left unchanged.
    """
    rename_map: dict[str, str] = {
        "SeriousDlqin2yrs":                      "default_2y",
        "RevolvingUtilizationOfUnsecuredLines":   "util_unsecured",
        "age":                                    "age_years",
        "DebtRatio":                              "debt_ratio",
        "MonthlyIncome":                          "monthly_income",
        "NumberOfOpenCreditLinesAndLoans":        "open_credit_cnt",
        "NumberOfTimes90DaysLate":                "dpd_90p_cnt",
        "NumberRealEstateLoansOrLines":           "real_estate_cnt",
        "NumberOfDependents":                     "dependents_cnt",
        "NumberOfTime30-59DaysPastDueNotWorse":   "dpd_30_59_cnt",
        "NumberOfTime60-89DaysPastDueNotWorse":   "dpd_60_89_cnt",
    }

    df = safe_rename(df, rename_map)

    expected = [
        "id", "default_2y", "util_unsecured", "age_years",
        "dpd_30_59_cnt", "dpd_60_89_cnt", "dpd_90p_cnt",
        "debt_ratio", "monthly_income", "open_credit_cnt",
        "real_estate_cnt", "dependents_cnt",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        log_warn(
            f"Some expected columns are missing after renaming: {', '.join(missing)}"
        )

    return df


# ==============================================================================
# Deduplication
# ==============================================================================

def deduplicate_rows_ignoring_id(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows, ignoring the id column when comparing.

    Summary:
        Identifies rows that are identical across all columns except "id",
        keeps the first occurrence, and logs the count and percentage removed.

    Args:
        df: Input DataFrame. Must contain an "id" column.

    Returns:
        Tuple (df_dedup, dup_n) where:
            - df_dedup: deduplicated DataFrame.
            - dup_n: number of duplicate rows removed.
    """
    key_cols = [c for c in df.columns if c != "id"]
    dup_mask = df.duplicated(subset=key_cols, keep="first")

    dup_n   = int(dup_mask.sum())
    dup_pct = 100 * dup_n / len(df)

    log_info(
        f"Repeated rows (ignoring id): {dup_n} ({dup_pct:.2f}% of dataset). "
        "Keeping first occurrence."
    )

    return df[~dup_mask].reset_index(drop=True), dup_n


# ==============================================================================
# Missing value helpers
# ==============================================================================

def create_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create boolean missingness indicator columns for selected features.

    Summary:
        Adds two columns indicating whether monthly_income and dependents_cnt
        are missing. These flags are informative since missingness may be
        non-random in this dataset.
        Note: dependents_missing is dropped at the end of the features pipeline
        (post-EDA decision). income_missing is retained as a model feature.

    Args:
        df: Input DataFrame containing monthly_income and dependents_cnt.

    Returns:
        DataFrame with two additional boolean columns:
            - income_missing: True where monthly_income is NaN.
            - dependents_missing: True where dependents_cnt is NaN.
    """
    df = df.copy()
    df["income_missing"]     = df["monthly_income"].isna()
    df["dependents_missing"] = df["dependents_cnt"].isna()
    return df


def median_impute(
    x: pd.Series,
    round_to_int: bool = False,
) -> pd.Series:
    """Replace NaN values with the column median.

    Summary:
        Computes the median of x (ignoring NaNs) and fills missing entries.
        Optionally rounds the result and casts to Int64 (nullable integer),
        which preserves integer semantics of count-type features.

    Args:
        x: Numeric pandas Series to impute.
        round_to_int: If True, imputed values are rounded and cast to Int64.

    Returns:
        Series with NaN values replaced by the column median.
    """
    med = x.median(skipna=True)
    x = x.fillna(med)
    if round_to_int:
        x = x.round().astype("Int64")
    return x


# ==============================================================================
# Outlier capping and transforms
# ==============================================================================

def cap_and_log1p(x: pd.Series, cap_value: float) -> pd.Series:
    """Cap a numeric Series at cap_value and apply a log1p transform.

    Summary:
        Clips values above cap_value, then applies log1p to compress the right
        tail. Equivalent to log1p(pmin(x, cap_value)) in R.

    Args:
        x: Numeric pandas Series to transform.
        cap_value: Upper threshold; values above this are clipped before log.

    Returns:
        Series equal to np.log1p(x.clip(upper=cap_value)).
    """
    return np.log1p(x.clip(upper=cap_value))


def flag_util_outliers(
    df: pd.DataFrame,
    util_cap: float = 10.0,
) -> pd.DataFrame:
    """Flag utilization outliers and apply a capped log1p transform.

    Summary:
        Creates two boolean flag columns from the raw util_unsecured values
        before any transformation, then caps and log-transforms the column
        in-place. Flags are built first to preserve the original signal.
        Note: util_gt10 is dropped at the end of the features pipeline
        (post-EDA decision). util_gt1 is retained.

    Args:
        df: Input DataFrame containing util_unsecured.
        util_cap: Upper cap threshold for the transformation. Defaults to 10.

    Returns:
        DataFrame with three changes:
            - util_gt1: True where raw util_unsecured > 1.
            - util_gt10: True where raw util_unsecured > util_cap.
            - util_unsecured: replaced with log1p(clip(util_unsecured, util_cap)).
    """
    df = df.copy()
    df["util_gt1"]  = df["util_unsecured"].gt(1)        & df["util_unsecured"].notna()
    df["util_gt10"] = df["util_unsecured"].gt(util_cap) & df["util_unsecured"].notna()
    df["util_unsecured"] = cap_and_log1p(df["util_unsecured"], util_cap)
    return df


def cap_dependents(df: pd.DataFrame, cap_value: int = 10) -> pd.DataFrame:
    """Cap dependents_cnt at a maximum value, preserving NaN.

    Summary:
        Clips dependents_cnt to cap_value to reduce the influence of rare
        high-count observations. Missing values are left unchanged.

    Args:
        df: Input DataFrame containing dependents_cnt.
        cap_value: Maximum allowed value for dependents count. Defaults to 10.

    Returns:
        DataFrame with dependents_cnt clipped at cap_value; NaN preserved.
    """
    df = df.copy()
    df["dependents_cnt"] = df["dependents_cnt"].clip(upper=cap_value)
    return df


# ==============================================================================
# DPD (days past due) anomaly detection and cleaning
# ==============================================================================

def flag_and_clean_dpd(
    df: pd.DataFrame,
    dpd_cols: list[str] | None = None,
    special_values: list[float] | None = None,
    threshold: float = 90,
) -> pd.DataFrame:
    """Create a unified delinquency anomaly flag and sanitize DPD columns.

    Summary:
        Marks rows as anomalous when any DPD column contains a value >= threshold
        or matches a known sentinel value (e.g. 96, 98 are Kaggle artefacts).
        Anomalous values are then replaced with NaN to prevent distortion of
        downstream statistics and feature builders.
        Note: dpd_any_gt90 is dropped at the end of the features pipeline
        (post-EDA decision). The raw DPD count columns are consumed by
        feature_utils.build_dpd_severity and are dropped there.

    Args:
        df: Input DataFrame containing the DPD columns.
        dpd_cols: Column names to inspect. Defaults to
            ["dpd_30_59_cnt", "dpd_60_89_cnt", "dpd_90p_cnt"].
        special_values: Sentinel values indicating data anomalies. Defaults to
            [96, 98].
        threshold: Values >= this number are treated as anomalous. Defaults to 90.

    Returns:
        DataFrame with:
            - dpd_any_gt90: boolean row-level flag (True if any DPD column
              was anomalous for that row).
            - All columns in dpd_cols with anomalous values replaced by NaN.
    """
    if dpd_cols is None:
        dpd_cols = ["dpd_30_59_cnt", "dpd_60_89_cnt", "dpd_90p_cnt"]
    if special_values is None:
        special_values = [96, 98]

    df = df.copy()

    def is_anom(s: pd.Series) -> pd.Series:
        return s.notna() & (s.ge(threshold) | s.isin(special_values))

    flag = pd.Series(False, index=df.index)
    for col in dpd_cols:
        if col not in df.columns:
            continue
        anom = is_anom(df[col])
        flag = flag | anom
        df.loc[anom, col] = np.nan

    df["dpd_any_gt90"] = flag

    n_flagged   = int(flag.sum())
    pct_flagged = 100 * flag.mean()
    log_info(
        f"DPD anomaly flag created (dpd_any_gt90). "
        f"Marked rows: {n_flagged} ({pct_flagged:.2f}%)."
    )

    return df


# ==============================================================================
# Age cleaning
# ==============================================================================

def clean_age(
    df: pd.DataFrame,
    remove_zero: bool = True,
    min_age: int = 18,
    max_age: int = 120,
) -> pd.DataFrame:
    """Clean age_years: remove zero-age rows and null out-of-range values.

    Summary:
        Two-step cleaning: (1) optionally drops rows where age_years == 0,
        assumed to be data entry errors; (2) sets implausible ages (below
        min_age or above max_age) to NaN.

    Args:
        df: Input DataFrame containing age_years.
        remove_zero: If True (default), rows where age_years == 0 are dropped.
        min_age: Minimum plausible age; values below this are set to NaN.
            Defaults to 18.
        max_age: Maximum plausible age; values above this are set to NaN.
            Defaults to 120.

    Returns:
        DataFrame with cleaned age_years. Row count may decrease when
        remove_zero is True.
    """
    df = df.copy()

    if remove_zero:
        before  = len(df)
        df      = df[df["age_years"] != 0].reset_index(drop=True)
        removed = before - len(df)
        log_info(f"Removed rows where age_years == 0: {removed}")

    out_range = df["age_years"].notna() & (
        df["age_years"].lt(min_age) | df["age_years"].gt(max_age)
    )
    n_out = int(out_range.sum())
    df.loc[out_range, "age_years"] = np.nan
    log_info(
        f"Set age_years out of range [{min_age}, {max_age}] to NaN: {n_out}"
    )

    return df


# ==============================================================================
# Database stub
# ==============================================================================

def append_to_db_example(df: pd.DataFrame, table_name: str) -> None:
    """Placeholder for appending a DataFrame to a database table.

    Summary:
        Intentional stub — no credentials or side effects. Demonstrates how a
        SQLAlchemy / psycopg2 integration would look; credentials must be
        supplied via environment variables and never committed to version control.

    Args:
        df: DataFrame to be appended to the destination table.
        table_name: Name of the target database table.

    Returns:
        None

    Example::

        # Example only — DO NOT COMMIT CREDENTIALS
        # import os
        # from sqlalchemy import create_engine
        #
        # engine = create_engine(
        #     "postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}".format(
        #         user=os.environ["DB_USER"],
        #         pw=os.environ["DB_PASSWORD"],
        #         host=os.environ["DB_HOST"],
        #         port=os.environ["DB_PORT"],
        #         db=os.environ["DB_NAME"],
        #     )
        # )
        # df.to_sql(table_name, engine, if_exists="append", index=False)
    """
    return None