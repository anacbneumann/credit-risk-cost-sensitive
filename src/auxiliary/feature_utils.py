# ==============================
# feature_utils.py
# ==============================
# Summary:
#   Utility functions for the credit risk feature engineering pipeline.
#   Covers all analytical derived feature builders: DPD severity encoding,
#   debt ratio transformations, income transformations, and real estate
#   discretization. Each function receives a clean DataFrame (output of
#   01_preprocessing.py) and returns an augmented DataFrame.
#
#   This module contains no cleaning logic — that responsibility belongs to
#   preprocessing_utils.py. The separation ensures that each stage of the
#   pipeline has a single, well-defined responsibility.
#
# Typical usage:
#   from feature_utils import (
#       build_dpd_severity,
#       build_debt_ratio_features,
#       build_income_features,
#       build_real_estate_bucket,
#       build_feature_summary,
#       save_feature_summary_to_excel,
#   )
# ==============================

# ── Standard library ──────────────────────────────────────────────────────────
import logging
from datetime import date
from pathlib import Path

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
# DPD severity encoding
# ==============================================================================

def build_dpd_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Build an ordinal DPD severity score from the delinquency count columns.

    Summary:
        Encodes the worst delinquency bucket ever observed for each borrower
        as a single ordinal integer. Intermediate ever-flags are created
        internally and dropped before returning. The source DPD count columns
        (dpd_30_59_cnt, dpd_60_89_cnt, dpd_90p_cnt) are dropped via
        COLS_TO_DROP in 02_features.py after this function runs.

        Severity scale:
            0 = no delinquency ever
            1 = worst bucket is 30–59 days past due
            2 = worst bucket is 60–89 days past due
            3 = worst bucket is 90+ days past due

        Assignments are applied in ascending order so that a borrower with
        90+ DPD always ends at level 3, regardless of lower-bucket counts.

    Args:
        df: Input DataFrame containing dpd_30_59_cnt, dpd_60_89_cnt, and
            dpd_90p_cnt (after anomalous values have been set to NaN by
            preprocessing_utils.flag_and_clean_dpd).

    Returns:
        DataFrame with one new column:
            - dpd_severity: int64 ordinal in {0, 1, 2, 3}.
    """
    df = df.copy()

    df["dpd_30_59_ever"] = (df["dpd_30_59_cnt"].fillna(0) > 0).astype(int)
    df["dpd_60_89_ever"] = (df["dpd_60_89_cnt"].fillna(0) > 0).astype(int)
    df["dpd_90p_ever"]   = (df["dpd_90p_cnt"].fillna(0)   > 0).astype(int)

    df["dpd_severity"] = 0
    df.loc[df["dpd_30_59_ever"] == 1, "dpd_severity"] = 1
    df.loc[df["dpd_60_89_ever"] == 1, "dpd_severity"] = 2
    df.loc[df["dpd_90p_ever"]   == 1, "dpd_severity"] = 3

    df = df.drop(columns=["dpd_30_59_ever", "dpd_60_89_ever", "dpd_90p_ever"])

    n_nonzero = int((df["dpd_severity"] > 0).sum())
    log_info(
        f"dpd_severity created. Borrowers with any delinquency: "
        f"{n_nonzero} ({100 * n_nonzero / len(df):.2f}%). "
        f"Distribution: {df['dpd_severity'].value_counts().sort_index().to_dict()}"
    )

    return df


# ==============================================================================
# Debt ratio features
# ==============================================================================

def build_debt_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build derived features from debt_ratio.

    Summary:
        Creates an unreliability flag (extreme debt ratio combined with missing
        income), a four-bucket ordinal category, a capped version, and a
        log-transformed version. Bucket boundaries separate plausible from
        implausible debt-ratio values.

        Columns produced and their fate (see COLS_TO_DROP in 02_features.py):
            - dr_unreliable:         dropped (low predictive power post-EDA)
            - dr_bucket:             dropped (superseded by numeric version)
            - debt_ratio_cap100:     dropped (intermediate; log version retained)
            - debt_ratio_cap100_log: retained as model feature
            - debt_ratio (original): dropped

    Args:
        df: Input DataFrame containing debt_ratio and income_missing
            (income_missing is created by preprocessing_utils.create_missing_flags).

    Returns:
        DataFrame with four additional columns:
            - dr_unreliable: True when income_missing is True AND debt_ratio > 100.
            - dr_bucket: ordinal category in {0,1,2,3} based on debt_ratio.
            - debt_ratio_cap100: debt_ratio clipped at 100.
            - debt_ratio_cap100_log: log1p(debt_ratio_cap100).
    """
    df = df.copy()

    df["dr_unreliable"] = (
        df["income_missing"]
        & df["debt_ratio"].notna()
        & df["debt_ratio"].gt(100)
    )

    conditions = [
        df["debt_ratio"].isna(),
        df["debt_ratio"].le(100),
        df["debt_ratio"].gt(100)   & df["debt_ratio"].le(1_000),
        df["debt_ratio"].gt(1_000) & df["debt_ratio"].le(10_000),
        df["debt_ratio"].gt(10_000),
    ]
    choices = [np.nan, 0, 1, 2, 3]
    df["dr_bucket"] = pd.Categorical(
        np.select(conditions, choices, default=np.nan),
        categories=[0, 1, 2, 3],
        ordered=True,
    )

    df["debt_ratio_cap100"]     = df["debt_ratio"].clip(upper=100)
    df["debt_ratio_cap100_log"] = np.log1p(df["debt_ratio_cap100"])

    n_unreliable = int(df["dr_unreliable"].sum())
    log_info(
        f"Debt ratio features created. "
        f"dr_unreliable rows: {n_unreliable} ({100 * n_unreliable / len(df):.2f}%)."
    )

    return df


# ==============================================================================
# Income features
# ==============================================================================

def build_income_features(
    df: pd.DataFrame,
    cap_p: float = 0.995,
) -> pd.DataFrame:
    """Build derived features from monthly_income.

    Summary:
        Computes an upper cap at the cap_p percentile of the raw income
        distribution, flags rows above that cap as extreme, then creates capped
        and log-transformed versions. The extreme flag is derived from the raw
        (pre-imputation) values so that imputed observations are not falsely
        flagged.

        Columns produced and their fate (see COLS_TO_DROP in 02_features.py):
            - income_extreme_flag:    dropped (low predictive power post-EDA)
            - monthly_income_cap:     dropped (intermediate; log version retained)
            - monthly_income_cap_log: retained as model feature
            - monthly_income (original): dropped

    Args:
        df: Input DataFrame containing monthly_income.
        cap_p: Quantile level for the income cap (e.g. 0.995 = 99.5th percentile).
            Defaults to 0.995.

    Returns:
        DataFrame with three additional columns:
            - income_extreme_flag: True when raw monthly_income exceeds the cap.
            - monthly_income_cap: monthly_income clipped at the percentile cap.
            - monthly_income_cap_log: log1p(monthly_income_cap).
    """
    df = df.copy()

    cap = float(df["monthly_income"].quantile(cap_p))

    df["income_extreme_flag"]    = df["monthly_income"].notna() & df["monthly_income"].gt(cap)
    df["monthly_income_cap"]     = df["monthly_income"].clip(upper=cap)
    df["monthly_income_cap_log"] = np.log1p(df["monthly_income_cap"])

    n_extreme   = int(df["income_extreme_flag"].sum())
    pct_extreme = 100 * df["income_extreme_flag"].mean()
    log_info(
        f"Income features created. Cap set to p{cap_p:.3f} = {cap:.2f}. "
        f"Extreme flagged rows: {n_extreme} ({pct_extreme:.2f}%)."
    )

    return df


# ==============================================================================
# Real estate bucket
# ==============================================================================

def build_real_estate_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Discretize real_estate_cnt into an ordinal bucket feature.

    Summary:
        Groups the number of real estate loans/lines into six ordered buckets
        to reduce sparsity at the high end of the distribution. Bucket edges
        mirror the original R implementation exactly.

        Bucket mapping:
            0: count == 0
            1: count == 1
            2: count == 2
            3: count in [3, 5]
            4: count in [6, 10]
            5: count >= 11

        Columns produced and their fate (see COLS_TO_DROP in 02_features.py):
            - real_estate_bucket: dropped (low predictive power post-EDA)
            - real_estate_cnt (original): dropped

    Args:
        df: Input DataFrame containing real_estate_cnt.

    Returns:
        DataFrame with one additional column:
            - real_estate_bucket: ordered pd.Categorical with levels [0..5].
              Rows where real_estate_cnt is NaN receive NaN.
    """
    df = df.copy()

    bins   = [-np.inf, 0, 1, 2, 5, 10, np.inf]
    labels = [0, 1, 2, 3, 4, 5]

    df["real_estate_bucket"] = pd.cut(
        df["real_estate_cnt"],
        bins=bins,
        labels=labels,
        right=True,
    ).astype("Int64")

    df["real_estate_bucket"] = pd.Categorical(
        df["real_estate_bucket"],
        categories=labels,
        ordered=True,
    )

    log_info(
        f"real_estate_bucket created. "
        f"Distribution: {df['real_estate_bucket'].value_counts().sort_index().to_dict()}"
    )

    return df


# ==============================================================================
# Feature summary and export
# ==============================================================================

def build_feature_summary(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Build a summary statistics table for the final feature set.

    Summary:
        Computes dtype, count, null rate, mean, std, min, median, and max for
        each feature column. Adds a generated_on date column for traceability.
        The result is intended for Excel export and optionally for appending
        to a database feature statistics table for drift monitoring.

        NOTE: In a production environment this summary would be appended to a
        feature statistics table in the database.
        Example:
            summary_df.to_sql("schema.feature_stats", engine,
                               if_exists="append", index=False)

    Args:
        df: DataFrame containing the final feature columns.
        feature_cols: List of column names to summarise. Columns not present
            in df are skipped with a warning.

    Returns:
        DataFrame with one row per feature and columns:
            feature, dtype, count, null_rate, mean, std, min, median, max,
            generated_on.
    """
    valid_cols = [c for c in feature_cols if c in df.columns]
    skipped    = [c for c in feature_cols if c not in df.columns]

    if skipped:
        log_warn(f"Columns not found in DataFrame, skipped in summary: {skipped}")

    rows = []
    for col in valid_cols:
        s       = df[col]
        numeric = pd.to_numeric(s, errors="coerce")
        has_num = numeric.notna().any()

        rows.append({
            "feature":      col,
            "dtype":        str(s.dtype),
            "count":        int(s.notna().sum()),
            "null_rate":    round(float(s.isna().mean()), 6),
            "mean":         round(float(numeric.mean()),   6) if has_num else None,
            "std":          round(float(numeric.std()),    6) if has_num else None,
            "min":          round(float(numeric.min()),    6) if has_num else None,
            "median":       round(float(numeric.median()), 6) if has_num else None,
            "max":          round(float(numeric.max()),    6) if has_num else None,
            "generated_on": str(date.today()),
        })

    return pd.DataFrame(rows)


def save_feature_summary_to_excel(
    summary_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Append a feature summary DataFrame to an Excel file.

    Summary:
        If the file already exists, the new rows are appended below existing
        data to preserve history. If it does not exist, a new file is created.

        NOTE: In a production environment this function would be replaced by
        an append to a feature statistics tracking table in the database.
        Example:
            summary_df.to_sql("schema.feature_stats", engine,
                               if_exists="append", index=False)

    Args:
        summary_df: DataFrame produced by build_feature_summary.
        output_path: Path to the target Excel file (.xlsx).

    Returns:
        None
    """
    output_path = Path(output_path)

    if output_path.exists():
        existing = pd.read_excel(output_path)
        combined = pd.concat([existing, summary_df], ignore_index=True)
        log_info(
            f"Appending {len(summary_df)} row(s) to existing summary "
            f"({len(existing)} existing rows)."
        )
    else:
        combined = summary_df
        log_info(f"Creating new feature summary with {len(summary_df)} row(s).")

    combined.to_excel(output_path, index=False)
    log_info(f"Feature summary saved to: {output_path}")