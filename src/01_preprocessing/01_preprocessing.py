# ==============================
# 01_preprocessing.py
# ==============================
# Summary:
#   Main preprocessing executor script.
#   Loads the raw training dataset, applies the rule-based cleaning pipeline,
#   and saves two outputs: raw (as loaded) and clean (treated, pre-features).
#   Feature engineering is handled downstream by 02_features.py.
#   Converted from the original R implementation to integrate into a Python-based
#   pipeline. Parameters are centralized in a single PARAMS dict for easy tuning.
#
# Inputs:
#   - CSV file path (default: database/cs-training.csv)
#
# Outputs:
#   - CSV raw dataset   (default: database/processed/cs_train_raw.csv)
#   - CSV clean dataset (default: database/processed/cs_train_clean.csv)
#
# Typical usage:
#   python 01_preprocessing.py
#   python 01_preprocessing.py database/cs-training.csv database/processed
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
    median_impute,
    cap_dependents,
    # append_to_db_example,  # Uncomment when DB integration is needed
)


# ==============================================================================
# Centralized parameters
# ==============================================================================

PARAMS: dict = {
    "dpd_cols":            ["dpd_30_59_cnt", "dpd_60_89_cnt", "dpd_90p_cnt"],
    "dpd_special_values":  [96, 98],
    "dpd_threshold":       90,

    "util_cap":            10,

    "age_remove_zero":     True,
    "age_min":             18,
    "age_max":             120,

    "dependents_cap":      10,

    "impute_round_counts": True,
}


# ==============================================================================
# CLI argument parsing
# ==============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for input/output paths.

    Summary:
        Provides an argparse-based interface with two optional positional
        arguments that fall back to sensible defaults.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:] when None.

    Returns:
        argparse.Namespace with:
            - input_path (str): Path to the raw CSV file.
            - output_dir (str): Directory where outputs are saved.
    """
    parser = argparse.ArgumentParser(
        description="Clean the GiveMeSomeCredit dataset (pre-feature-engineering)."
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
        Equivalent to R's dir.create(path, recursive = TRUE). Safe to call
        even if the directory already exists.

    Args:
        path: Directory path to create (string or Path).

    Returns:
        Path object pointing to the (now existing) directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ==============================================================================
# Pipeline
# ==============================================================================

def main(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Execute the end-to-end cleaning pipeline.

    Summary:
        Orchestrates the full sequence of cleaning steps: column renaming,
        deduplication, missing flags, DPD anomaly detection and sanitization,
        utilization outlier capping, age cleaning, median imputation, and
        dependents capping. Persists both the raw and clean DataFrames as CSV.
        Feature engineering (dpd_severity, debt ratio features, income features,
        real estate buckets) is handled downstream by 02_features.py.

    Args:
        argv: Optional argument list forwarded to parse_args. When None,
            arguments are read from sys.argv.

    Returns:
        Dictionary with two keys:
            - "raw": DataFrame as loaded from disk (no transformations).
            - "clean": DataFrame after all cleaning steps, before features.
    """
    cfg        = parse_args(argv)
    output_dir = ensure_dir(cfg.output_dir)

    log_info(f"Starting preprocessing pipeline. Input: {cfg.input_path}")

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    raw_df = pd.read_csv(cfg.input_path)
    raw_df = standardize_column_names(raw_df)
    log_info(f"Raw dataset loaded. Rows: {len(raw_df)} | Cols: {len(raw_df.columns)}")

    # Work copy — raw_df is kept untouched for the raw output
    df = raw_df.copy()

    # ------------------------------------------------------------------
    # 1) Rename columns to snake_case
    # ------------------------------------------------------------------
    df = rename_credit_columns(df)
    log_info("Step 1 complete: columns renamed.")

    # ------------------------------------------------------------------
    # 2) Deduplicate rows (ignoring id)
    # ------------------------------------------------------------------
    df, _ = deduplicate_rows_ignoring_id(df)
    log_info(f"Step 2 complete: deduplication. Rows remaining: {len(df)}")

    # ------------------------------------------------------------------
    # 3) Missing flags — must run BEFORE imputation
    # ------------------------------------------------------------------
    df = create_missing_flags(df)
    log_info("Step 3 complete: missing flags created (income_missing, dependents_missing).")

    # ------------------------------------------------------------------
    # 4) DPD anomaly flag + replace anomalous values with NaN
    # ------------------------------------------------------------------
    df = flag_and_clean_dpd(
        df,
        dpd_cols=PARAMS["dpd_cols"],
        special_values=PARAMS["dpd_special_values"],
        threshold=PARAMS["dpd_threshold"],
    )
    log_info("Step 4 complete: DPD anomalies flagged and cleaned.")

    # ------------------------------------------------------------------
    # 5) util_unsecured: outlier flags + cap + log1p transform
    # ------------------------------------------------------------------
    df = flag_util_outliers(df, util_cap=PARAMS["util_cap"])
    log_info("Step 5 complete: util_unsecured capped, log-transformed, flags created.")

    # ------------------------------------------------------------------
    # 6) Age cleaning
    # ------------------------------------------------------------------
    df = clean_age(
        df,
        remove_zero=PARAMS["age_remove_zero"],
        min_age=PARAMS["age_min"],
        max_age=PARAMS["age_max"],
    )
    log_info("Step 6 complete: age_years cleaned.")

    # ------------------------------------------------------------------
    # 7) Median imputation — after flags and caps are in place
    # ------------------------------------------------------------------
    df["monthly_income"] = median_impute(df["monthly_income"], round_to_int=False)
    df["dependents_cnt"] = median_impute(
        df["dependents_cnt"], round_to_int=PARAMS["impute_round_counts"]
    )
    log_info("Step 7 complete: median imputation applied (monthly_income, dependents_cnt).")

    # ------------------------------------------------------------------
    # 8) Dependents cap
    # ------------------------------------------------------------------
    df = cap_dependents(df, cap_value=PARAMS["dependents_cap"])
    log_info(f"Step 8 complete: dependents_cnt capped at {PARAMS['dependents_cap']}.")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    raw_out   = output_dir / "cs_train_raw.csv"
    clean_out = output_dir / "cs_train_clean.csv"

    raw_df.to_csv(raw_out,   index=False)
    df.to_csv(clean_out,     index=False)

    log_info(f"Saved raw dataset:   {raw_out}")
    log_info(f"Saved clean dataset: {clean_out}")

    # Example: DB append (uncomment when integration is ready)
    # append_to_db_example(df, table_name="schema.cs_train_clean")

    log_info(
        f"Preprocessing complete. "
        f"Final clean rows: {len(df)} (raw was {len(raw_df)})."
    )

    return {"raw": raw_df, "clean": df}


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    main(sys.argv[1:])