# ==============================
# 04_evaluate.py
# ==============================
# Summary:
#   Evaluates the trained XGBoost model on the held-out test set.
#   Loads the serialised model and the test split produced by 02_train.py,
#   generates predicted probabilities, finds the cost-optimal threshold
#   (CFP=1, CFN=5), computes a full set of metrics for both train and test,
#   logs a confusion matrix summary, and exports all metrics to Excel.
#
#   Cost rationale: misclassifying a defaulter (FN) is assumed to be 5x
#   more costly than denying credit to a good payer (FP). The theoretical
#   optimal threshold derived from this ratio is t* = 1/6 ≈ 0.1667.
#
# Inputs:
#   - Serialised model pipeline  (default: models/xgb_final.joblib)
#   - Test split CSV             (default: database/splits/test.csv)
#   - Train split CSV            (default: database/splits/train.csv)
#
# Outputs:
#   - Excel metrics file         (default: reports/model_metrics.xlsx)
#
# Typical usage:
#   python 04_evaluate.py
#   python 04_evaluate.py models/xgb_final.joblib database/splits/test.csv
# ==============================

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import sys
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd

# ── Internal ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.auxiliary.model_utils import (
    log_info,
    log_warn,
    prepare_features,
    find_optimal_threshold,
    compute_metrics,
    save_metrics_to_excel,
)


# ==============================================================================
# Centralized parameters
# ==============================================================================

PARAMS: dict = {
    # Cost matrix
    "c_fp": 1.0,
    "c_fn": 5.0,

    # Threshold sweep resolution
    "threshold_step": 0.01,

    # Column settings (must match 02_train.py)
    "target_col": "default_2y",
    "drop_cols":  [],   # id was already removed in split CSVs
}


# ==============================================================================
# CLI argument parsing
# ==============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for model, data, and output paths.

    Args:
        argv: Argument list. Defaults to sys.argv[1:] when None.

    Returns:
        Namespace with model_path, test_path, train_path, and output_path.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the trained XGBoost credit risk model."
    )
    parser.add_argument(
        "model_path", nargs="?",
        default="models/xgb_final.joblib",
        help="Path to serialised model (default: models/xgb_final.joblib).",
    )
    parser.add_argument(
        "test_path", nargs="?",
        default="database/splits/test.csv",
        help="Path to test split CSV (default: database/splits/test.csv).",
    )
    parser.add_argument(
        "train_path", nargs="?",
        default="database/splits/train.csv",
        help="Path to train split CSV (default: database/splits/train.csv).",
    )
    parser.add_argument(
        "output_path", nargs="?",
        default="reports/model_metrics.xlsx",
        help="Path to output Excel metrics file (default: reports/model_metrics.xlsx).",
    )
    return parser.parse_args(argv)


# ==============================================================================
# Helpers
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


def load_model(model_path: str | Path) -> tuple:
    """Load a serialised model artefact produced by 02_train.py.

    Args:
        model_path: Path to the .joblib file containing
            {"model": XGBClassifier, "feature_names": list[str]}.

    Returns:
        Tuple (model, feature_names).

    Raises:
        FileNotFoundError: If model_path does not exist.
        KeyError: If the artefact does not contain expected keys.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    artefact = joblib.load(model_path)
    log_info(f"Model loaded from: {model_path}")
    return artefact["model"], artefact["feature_names"]


def align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Align a feature DataFrame to match the columns seen during training.

    Summary:
        Adds missing columns as zeros and drops extra columns, then reorders
        to match the training column order. Logs any discrepancies as warnings.

    Args:
        X: Feature DataFrame to align.
        feature_names: Ordered list of column names from training.

    Returns:
        DataFrame with exactly the columns in feature_names, in that order.
    """
    missing = [c for c in feature_names if c not in X.columns]
    extra   = [c for c in X.columns if c not in feature_names]

    if missing:
        log_warn(f"Columns missing in input (filling with 0): {missing}")
        for col in missing:
            X[col] = 0

    if extra:
        log_warn(f"Extra columns in input (dropping): {extra}")
        X = X.drop(columns=extra)

    return X[feature_names]


def log_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    c_fp: float,
    c_fn: float,
    split_label: str = "test",
) -> None:
    """Log a formatted confusion matrix summary to stdout.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for class 1.
        threshold: Classification cutoff.
        c_fp: Cost per false positive.
        c_fn: Cost per false negative.
        split_label: Label for the dataset split being evaluated.

    Returns:
        None
    """
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    cost = c_fp * fp + c_fn * fn

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    log_info(
        f"[{split_label.upper()}] Confusion matrix at threshold={threshold:.4f} | "
        f"TP={tp} | TN={tn} | FP={fp} | FN={fn} | "
        f"Sensitivity={sensitivity:.4f} | Precision={precision:.4f} | "
        f"Cost (FP*{c_fp} + FN*{c_fn}) = {cost}"
    )


# ==============================================================================
# Pipeline
# ==============================================================================

def main(argv: list[str] | None = None) -> dict:
    """Execute the evaluation pipeline.

    Summary:
        Loads the serialised model and both splits, generates predicted
        probabilities, finds the cost-optimal threshold on the test set,
        computes metrics for train and test, logs the confusion matrix,
        and exports all results to Excel.

    Args:
        argv: Optional CLI argument list forwarded to parse_args.

    Returns:
        Dictionary with keys "train_metrics", "test_metrics", "best_threshold".
    """
    cfg = parse_args(argv)
    ensure_dir(Path(cfg.output_path).parent)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model, feature_names = load_model(cfg.model_path)

    # ------------------------------------------------------------------
    # Load splits
    # ------------------------------------------------------------------
    log_info(f"Loading test split:  {cfg.test_path}")
    df_test = pd.read_csv(cfg.test_path)

    log_info(f"Loading train split: {cfg.train_path}")
    df_train = pd.read_csv(cfg.train_path)

    log_info(
        f"Test rows: {len(df_test)} | Train rows: {len(df_train)}"
    )

    # ------------------------------------------------------------------
    # Prepare features — reuse prepare_features then align to training columns
    # ------------------------------------------------------------------
    X_test, y_test = prepare_features(
        df_test,
        target_col=PARAMS["target_col"],
        drop_cols=PARAMS["drop_cols"],
    )
    X_test = align_features(X_test, feature_names)

    X_train, y_train = prepare_features(
        df_train,
        target_col=PARAMS["target_col"],
        drop_cols=PARAMS["drop_cols"],
    )
    X_train = align_features(X_train, feature_names)

    # ------------------------------------------------------------------
    # Predict probabilities
    # ------------------------------------------------------------------
    log_info("Generating predicted probabilities on test set...")
    y_prob_test  = model.predict_proba(X_test)[:, 1]

    log_info("Generating predicted probabilities on train set...")
    y_prob_train = model.predict_proba(X_train)[:, 1]

    # ------------------------------------------------------------------
    # Find optimal threshold (on test set)
    # ------------------------------------------------------------------
    best_threshold, best_cost = find_optimal_threshold(
        y_true=y_test.values,
        y_prob=y_prob_test,
        c_fp=PARAMS["c_fp"],
        c_fn=PARAMS["c_fn"],
        step=PARAMS["threshold_step"],
    )
    log_info(
        f"Optimal threshold (test): {best_threshold:.4f} | "
        f"Minimum cost: {best_cost:.0f} | "
        f"Theoretical t* = {PARAMS['c_fp'] / (PARAMS['c_fp'] + PARAMS['c_fn']):.4f}"
    )

    # ------------------------------------------------------------------
    # Log confusion matrices
    # ------------------------------------------------------------------
    log_confusion_matrix(
        y_test.values, y_prob_test, best_threshold,
        PARAMS["c_fp"], PARAMS["c_fn"], split_label="test",
    )
    log_confusion_matrix(
        y_train.values, y_prob_train, best_threshold,
        PARAMS["c_fp"], PARAMS["c_fn"], split_label="train",
    )

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    test_metrics  = compute_metrics(
        y_test.values, y_prob_test, best_threshold,
        c_fp=PARAMS["c_fp"], c_fn=PARAMS["c_fn"],
        split_label="test",
    )
    train_metrics = compute_metrics(
        y_train.values, y_prob_train, best_threshold,
        c_fp=PARAMS["c_fp"], c_fn=PARAMS["c_fn"],
        split_label="train",
    )

    log_info(
        f"[TEST]  ROC AUC={test_metrics['roc_auc']} | "
        f"PR AUC={test_metrics['pr_auc']} | "
        f"Cost={test_metrics['cost']}"
    )
    log_info(
        f"[TRAIN] ROC AUC={train_metrics['roc_auc']} | "
        f"PR AUC={train_metrics['pr_auc']} | "
        f"Cost={train_metrics['cost']}"
    )

    # ------------------------------------------------------------------
    # Export metrics to Excel
    # ------------------------------------------------------------------
    save_metrics_to_excel(
        metrics=[train_metrics, test_metrics],
        output_path=cfg.output_path,
    )

    log_info("Evaluation complete.")

    return {
        "train_metrics":  train_metrics,
        "test_metrics":   test_metrics,
        "best_threshold": best_threshold,
    }


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    main(sys.argv[1:])