# ==============================
# 01_preprocessing.py
# ==============================
# Summary:
#   Main preprocessing executor script.
#   Loads the raw training dataset, applies the rule-based preprocessing pipeline,
#   and saves two outputs: raw (as loaded) and processed (treated).
#   Converted from the original R implementation to integrate into a Python-based
#   pipeline. Parameters are centralized in a single PARAMS dict for easy tuning.
#
# Inputs:
#   - CSV file path (default: data/cs_train.csv)
#
# Outputs:
#   - Parquet raw dataset      (default: data/processed/cs_train_raw.parquet)
#   - Parquet processed dataset (default: data/processed/cs_train_processed.parquet)
#
# Typical usage:
#   python 01_preprocessing.py
#   python 01_preprocessing.py data/cs_train.csv data/processed
# ==============================

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import sys
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd

# ── Internal ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.auxiliary.preprocessing_utils import (
    log_info,
    standardize_column_names,
    rename_credit_columns,
    deduplicate_rows_ignoring_id,
    create_missing_flags,
    flag_and_clean_dpd,
    flag_util_outliers,
    clean_age,
    build_debt_ratio_features,
    build_income_features,
    median_impute,
    cap_dependents,
    build_real_estate_bucket,
    # append_to_db_example,  # Example when DB integration is needed
)


# ==============================================================================
# Centralized parameters
# ==============================================================================

PARAMS: dict = {
    "dpd_cols":           ["dpd_30_59_cnt", "dpd_60_89_cnt", "dpd_90p_cnt"],
    "dpd_special_values": [96, 98],
    "dpd_threshold":      90,

    "util_cap":           10,

    "income_cap_p":       0.995,

    "age_remove_zero":    True,
    "age_min":            18,
    "age_max":            120,

    "dependents_cap":     10,

    "impute_round_counts": True,
}


# ==============================================================================
# CLI argument parsing
# ==============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for input/output paths.

    Summary:
        Provides an ``argparse``-based interface that mirrors the positional
        argument convention of the original R script. Both arguments are optional
        and fall back to sensible defaults.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        ``argparse.Namespace`` with two attributes:
            - ``input_path`` (str): Path to the raw CSV file.
            - ``output_dir`` (str): Directory where processed outputs are saved.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess the GiveMeSomeCredit dataset."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="database/cs-training.csv",
        help="Path to the raw CSV input file (default: database/cs-training.csv).",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="database/processed",
        help="Directory for processed outputs (default: database/processed).",
    )
    return parser.parse_args(argv)


# ==============================================================================
# File system helpers
# ==============================================================================

def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it recursively if needed.

    Summary:
        Equivalent to R's ``dir.create(path, recursive = TRUE)``. Safe to call
        even if the directory already exists.

    Args:
        path: Directory path to create (string or :class:`pathlib.Path`).

    Returns:
        :class:`pathlib.Path` object pointing to the (now existing) directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ==============================================================================
# Pipeline
# ==============================================================================

def main(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Execute the end-to-end preprocessing pipeline.

    Summary:
        Orchestrates the full sequence of preprocessing steps — column renaming,
        deduplication, missing flags, DPD cleaning, outlier transforms, age
        cleaning, derived feature construction, median imputation, and bucketing —
        then persists both the raw and processed DataFrames as Parquet files.

    Args:
        argv: Optional argument list forwarded to :func:`parse_args`. When
            ``None``, arguments are read from ``sys.argv``.

    Returns:
        Dictionary with two keys:
            - ``"raw"``: DataFrame as loaded from disk (no transformations).
            - ``"processed"``: DataFrame after all preprocessing steps.
    """
    cfg = parse_args(argv)
    output_dir = ensure_dir(cfg.output_dir)

    log_info(f"Starting preprocessing. Input: {cfg.input_path}")

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    raw_df = pd.read_csv(cfg.input_path)
    raw_df = standardize_column_names(raw_df)

    log_info(f"Raw dataset loaded. Rows: {len(raw_df)} | Cols: {len(raw_df.columns)}")

    # Work copy — raw_df is kept untouched for the raw output
    df = raw_df.copy()

    # ------------------------------------------------------------------
    # 1) Rename columns
    # ------------------------------------------------------------------
    df = rename_credit_columns(df)
    log_info("Columns renamed/standardized.")

    # ------------------------------------------------------------------
    # 2) Deduplicate rows (ignoring id)
    # ------------------------------------------------------------------
    df, _ = deduplicate_rows_ignoring_id(df)
    log_info(f"After deduplication. Rows: {len(df)}")

    # ------------------------------------------------------------------
    # 3) Missing flags — must run BEFORE imputation
    # ------------------------------------------------------------------
    df = create_missing_flags(df)
    log_info("Missing flags created: income_missing, dependents_missing.")

    # ------------------------------------------------------------------
    # 4) DPD anomalies: unified flag + replace anomalous values with NaN
    # ------------------------------------------------------------------
    df = flag_and_clean_dpd(
        df,
        dpd_cols=PARAMS["dpd_cols"],
        special_values=PARAMS["dpd_special_values"],
        threshold=PARAMS["dpd_threshold"],
    )

    # ------------------------------------------------------------------
    # 5) util_unsecured: outlier flags + cap + log1p transform
    # ------------------------------------------------------------------
    df = flag_util_outliers(df, util_cap=PARAMS["util_cap"])
    log_info(
        "util_unsecured transformed (cap + log1p) and flags created "
        "(util_gt1, util_gt10)."
    )

    # ------------------------------------------------------------------
    # 6) Age cleaning
    # ------------------------------------------------------------------
    df = clean_age(
        df,
        remove_zero=PARAMS["age_remove_zero"],
        min_age=PARAMS["age_min"],
        max_age=PARAMS["age_max"],
    )

    # ------------------------------------------------------------------
    # 7) Debt ratio derived features
    # ------------------------------------------------------------------
    df = build_debt_ratio_features(df)
    log_info(
        "Debt ratio features created: dr_unreliable, dr_bucket, "
        "debt_ratio_cap100, debt_ratio_cap100_log."
    )

    # ------------------------------------------------------------------
    # 8) Income derived features (extreme flag + cap + log1p)
    # ------------------------------------------------------------------
    df = build_income_features(df, cap_p=PARAMS["income_cap_p"])

    # ------------------------------------------------------------------
    # 9) Median imputation — after flags and caps are in place
    # ------------------------------------------------------------------
    df["monthly_income"] = median_impute(df["monthly_income"], round_to_int=False)
    df["dependents_cnt"] = median_impute(
        df["dependents_cnt"], round_to_int=PARAMS["impute_round_counts"]
    )
    log_info("Missing values imputed using median (monthly_income, dependents_cnt).")

    # ------------------------------------------------------------------
    # 10) Dependents cap
    # ------------------------------------------------------------------
    df = cap_dependents(df, cap_value=PARAMS["dependents_cap"])
    log_info(f"dependents_cnt capped at {PARAMS['dependents_cap']}.")

    # ------------------------------------------------------------------
    # 11) Real estate bucket (discrete ordinal)
    # ------------------------------------------------------------------
    df = build_real_estate_bucket(df)
    log_info("real_estate_bucket created (discrete bins).")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    raw_out  = output_dir / "cs_train_raw.csv"
    proc_out = output_dir / "cs_train_processed.csv"

    raw_df.to_csv(raw_out, index=False)
    df.to_csv(proc_out, index=False)

    log_info(f"Saved raw dataset:       {raw_out}")
    log_info(f"Saved processed dataset: {proc_out}")

    # Example: DB append (uncomment when integration is ready)
    # append_to_db_example(df, table_name="schema.table_name")

    log_info(f"Final processed rows: {len(df)} (raw was {len(raw_df)}).")

    return {"raw": raw_df, "processed": df}


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    main(sys.argv[1:])