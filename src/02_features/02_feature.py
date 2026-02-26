# ==============================
# 02_features.py
# ==============================
# Summary:
#   Feature engineering executor script.
#   Loads the clean dataset produced by 01_preprocessing.py, applies all
#   analytical derived feature builders (DPD severity, debt ratio transforms,
#   income transforms, real estate discretization), drops intermediate and
#   low-value columns based on post-EDA decisions, and saves the final
#   modelling-ready dataset. Also exports a feature summary table to Excel
#   for audit and consumption in the modelling stage.
#
#   This script has no cleaning logic — all sanitization is handled upstream
#   by 01_preprocessing.py.
#
# Inputs:
#   - Clean CSV (default: database/processed/cs_train_clean.csv)
#
# Outputs:
#   - CSV modelling dataset  (default: database/processed/cs_train_features.csv)
#   - Excel feature summary  (default: reports/feature_summary.xlsx)
#
# Typical usage:
#   python 02_features.py
#   python 02_features.py database/processed/cs_train_clean.csv database/processed reports
# ==============================

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import sys
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd

# ── Internal ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.auxiliary.feature_utils import (
    log_info,
    build_dpd_severity,
    build_debt_ratio_features,
    build_income_features,
    build_real_estate_bucket,
    build_feature_summary,
    save_feature_summary_to_excel,
    # append_to_db_example (stub) — uncomment when DB integration is ready
)


# ==============================================================================
# Centralized parameters
# ==============================================================================

PARAMS: dict = {
    "income_cap_p": 0.995,
}

# Columns removed at the end of this pipeline based on post-EDA decisions:
#   - Intermediate columns superseded by a derived version
#     (e.g. monthly_income_cap is dropped; monthly_income_cap_log is kept)
#   - Columns with low predictive power confirmed by bivariate EDA
#   - Columns with high multicollinearity with retained features
#
# real_estate_bucket_grp is listed defensively — may not exist in all runs.
COLS_TO_DROP: list[str] = [
    # DPD source columns (consumed by build_dpd_severity; no longer needed)
    "dpd_30_59_cnt",
    "dpd_60_89_cnt",
    "dpd_90p_cnt",
    # DPD anomaly flag (low predictive power post-EDA)
    "dpd_any_gt90",
    # Debt ratio intermediates and original
    "dr_unreliable",
    "dr_bucket",
    "debt_ratio_cap100",
    "debt_ratio_cap100_log",
    "debt_ratio",
    # Income intermediates and original
    "income_extreme_flag",
    "monthly_income_cap",
    "monthly_income",
    # Real estate (low predictive power post-EDA)
    "real_estate_cnt",
    "real_estate_bucket",
    "real_estate_bucket_grp",   # defensive
    # Utilization outlier flag (low predictive power post-EDA)
    "util_gt10",
    # Missing indicator (low predictive power post-EDA)
    "dependents_missing",
]


# ==============================================================================
# CLI argument parsing
# ==============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for input, output, and report paths.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:] when None.

    Returns:
        argparse.Namespace with:
            - input_path (str): Path to the clean CSV.
            - output_dir (str): Directory for the modelling-ready CSV.
            - report_dir (str): Directory for the Excel feature summary.
    """
    parser = argparse.ArgumentParser(
        description="Feature engineering for the GiveMeSomeCredit dataset."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="database/processed/cs_train_clean.csv",
        help="Path to the clean CSV (default: database/processed/cs_train_clean.csv).",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="database/processed",
        help="Directory for the modelling-ready CSV (default: database/processed).",
    )
    parser.add_argument(
        "report_dir",
        nargs="?",
        default="reports",
        help="Directory for the Excel feature summary (default: reports).",
    )
    return parser.parse_args(argv)


# ==============================================================================
# File system helpers
# ==============================================================================

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist.

    Args:
        path: Target directory path.

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
    """Execute the end-to-end feature engineering pipeline.

    Summary:
        Loads the clean dataset, applies all derived feature builders in
        dependency order, drops intermediate and post-EDA-discarded columns,
        saves the modelling-ready CSV, and exports a feature summary table
        to Excel.

    Args:
        argv: Optional argument list forwarded to parse_args. When None,
            arguments are read from sys.argv.

    Returns:
        Dictionary with one key:
            - "features": modelling-ready DataFrame after all steps and drops.
    """
    cfg        = parse_args(argv)
    output_dir = ensure_dir(cfg.output_dir)
    report_dir = ensure_dir(cfg.report_dir)

    log_info(f"Starting feature engineering pipeline. Input: {cfg.input_path}")

    # ------------------------------------------------------------------
    # Load clean data
    # ------------------------------------------------------------------
    df = pd.read_csv(cfg.input_path)
    log_info(f"Clean dataset loaded. Rows: {len(df)} | Cols: {len(df.columns)}")

    # ------------------------------------------------------------------
    # 1) DPD severity — must run BEFORE DPD source columns are dropped
    # ------------------------------------------------------------------
    df = build_dpd_severity(df)
    log_info(
        "Step 1 complete: dpd_severity created "
        "(0=none, 1=30-59d, 2=60-89d, 3=90d+)."
    )

    # ------------------------------------------------------------------
    # 2) Debt ratio derived features
    # ------------------------------------------------------------------
    df = build_debt_ratio_features(df)
    log_info(
        "Step 2 complete: debt ratio features created "
        "(dr_unreliable, dr_bucket, debt_ratio_cap100, debt_ratio_cap100_log)."
    )

    # ------------------------------------------------------------------
    # 3) Income derived features
    # ------------------------------------------------------------------
    df = build_income_features(df, cap_p=PARAMS["income_cap_p"])
    log_info(
        "Step 3 complete: income features created "
        "(income_extreme_flag, monthly_income_cap, monthly_income_cap_log)."
    )

    # ------------------------------------------------------------------
    # 4) Real estate bucket
    # ------------------------------------------------------------------
    df = build_real_estate_bucket(df)
    log_info("Step 4 complete: real_estate_bucket created.")

    # ------------------------------------------------------------------
    # 5) Drop intermediate and post-EDA-discarded columns
    # ------------------------------------------------------------------
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    cols_missing = [c for c in COLS_TO_DROP if c not in df.columns]

    df = df.drop(columns=cols_present)

    if cols_missing:
        log_info(
            f"Columns listed in COLS_TO_DROP but not found (skipped): {cols_missing}"
        )
    log_info(
        f"Step 5 complete: dropped {len(cols_present)} columns. "
        f"Remaining columns ({len(df.columns)}): {list(df.columns)}"
    )

    # ------------------------------------------------------------------
    # 6) Build feature summary and export to Excel
    # ------------------------------------------------------------------
    summary_exclude = {"id", "default_2y"}
    feature_cols    = [c for c in df.columns if c not in summary_exclude]

    summary_df   = build_feature_summary(df, feature_cols)
    summary_path = report_dir / "feature_summary.xlsx"
    save_feature_summary_to_excel(summary_df, summary_path)
    log_info(
        f"Step 6 complete: feature summary exported "
        f"({len(feature_cols)} features → {summary_path})."
    )

    # ------------------------------------------------------------------
    # Save modelling-ready dataset
    # ------------------------------------------------------------------
    features_out = output_dir / "cs_train_features.csv"
    df.to_csv(features_out, index=False)
    log_info(f"Modelling-ready dataset saved: {features_out}")

    # Example: DB append (uncomment when integration is ready)
    # append_to_db_example(df, table_name="schema.cs_train_features")

    log_info(
        f"Feature engineering complete. "
        f"Final rows: {len(df)} | Final columns: {len(df.columns)}"
    )

    return {"features": df}


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    main(sys.argv[1:])