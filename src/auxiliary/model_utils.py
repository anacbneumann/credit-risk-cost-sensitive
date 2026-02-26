# ==============================
# model_utils.py
# ==============================
# Summary:
#   Shared utility functions for the credit risk modelling pipeline.
#   Covers feature preparation, cost-sensitive threshold optimisation,
#   metric computation, and results persistence. Used by both 02_train.py
#   and 03_evaluate.py to keep shared logic in a single place.
#
#   Cost assumption: CFP = 1 (false positive), CFN = 5 (false negative).
#   Theoretical optimal threshold: t* = CFP / (CFP + CFN) = 1/6 ≈ 0.1667.
#
# Typical usage:
#   from model_utils import prepare_features, find_optimal_threshold, compute_metrics
# ==============================

# ── Standard library ──────────────────────────────────────────────────────────
import logging
from datetime import date
from pathlib import Path
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

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
    """Emit an INFO-level log message.

    Args:
        msg: Human-readable message.

    Returns:
        None
    """
    _logger.info(msg)


def log_warn(msg: str) -> None:
    """Emit a WARNING-level log message.

    Args:
        msg: Human-readable warning message.

    Returns:
        None
    """
    _logger.warning(msg)


# ==============================================================================
# Feature preparation
# ==============================================================================

def prepare_features(
    df: pd.DataFrame,
    target_col: str = "default_2y",
    drop_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from target and apply dummy encoding and zero-variance removal.

    Summary:
        Mirrors the tidymodels recipe used in the notebook:
        step_dummy (one-hot encode all object/bool/category columns) +
        step_zv (drop constant columns). Boolean columns are cast to int
        before encoding to produce clean 0/1 columns compatible with XGBoost.

    Args:
        df: Input DataFrame containing both features and target.
        target_col: Name of the binary target column. Defaults to "default_2y".
        drop_cols: Additional columns to drop before processing (e.g. "id").
            Defaults to None.

    Returns:
        Tuple (X, y) where:
            - X: pd.DataFrame of encoded, non-constant features.
            - y: pd.Series of integer target values (0/1).
    """
    df = df.copy()

    # Drop ancillary columns
    cols_to_remove = [target_col] + (drop_cols or [])
    cols_to_remove = [c for c in cols_to_remove if c in df.columns]

    y = df[target_col].astype(int)
    X = df.drop(columns=cols_to_remove)

    # Cast booleans to int so get_dummies does not create redundant columns
    bool_cols = X.select_dtypes(include="bool").columns.tolist()
    X[bool_cols] = X[bool_cols].astype(int)

    # One-hot encode remaining categorical / object columns
    X = pd.get_dummies(X, drop_first=False)

    # Remove zero-variance (constant) columns
    zv_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if zv_cols:
        log_warn(f"Dropping zero-variance columns: {zv_cols}")
        X = X.drop(columns=zv_cols)

    log_info(
        f"Features prepared: {X.shape[1]} columns, {X.shape[0]} rows. "
        f"Target positive rate: {y.mean():.4f}."
    )
    return X, y


# ==============================================================================
# Cost-sensitive threshold optimisation
# ==============================================================================

def cost_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    c_fp: float = 1.0,
    c_fn: float = 5.0,
) -> float:
    """Compute total business cost at a given classification threshold.

    Summary:
        Applies the threshold to predicted probabilities, counts false positives
        and false negatives, then returns the weighted total cost.
        Cost = c_fp * FP + c_fn * FN.

    Args:
        y_true: 1-D array of true binary labels (0/1).
        y_prob: 1-D array of predicted probabilities for class 1.
        threshold: Classification cutoff; observations with y_prob >= threshold
            are predicted as positive (1).
        c_fp: Cost per false positive. Defaults to 1.0.
        c_fn: Cost per false negative. Defaults to 5.0.

    Returns:
        Total cost as a float.
    """
    y_pred = (y_prob >= threshold).astype(int)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return c_fp * fp + c_fn * fn


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    c_fp: float = 1.0,
    c_fn: float = 5.0,
    step: float = 0.01,
) -> tuple[float, float]:
    """Find the classification threshold that minimises total business cost.

    Summary:
        Sweeps thresholds from 0 to 1 in increments of *step*, evaluates
        cost_at_threshold at each point, and returns the threshold and cost
        at the minimum. Ties are broken by taking the first (lowest) threshold.

    Args:
        y_true: 1-D array of true binary labels (0/1).
        y_prob: 1-D array of predicted probabilities for class 1.
        c_fp: Cost per false positive. Defaults to 1.0.
        c_fn: Cost per false negative. Defaults to 5.0.
        step: Grid resolution for threshold sweep. Defaults to 0.01.

    Returns:
        Tuple (best_threshold, best_cost).
    """
    thresholds = np.arange(0.0, 1.0 + step, step)
    costs = np.array([
        cost_at_threshold(y_true, y_prob, t, c_fp, c_fn)
        for t in thresholds
    ])
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx])


# ==============================================================================
# Metric computation
# ==============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    c_fp: float = 1.0,
    c_fn: float = 5.0,
    split_label: str = "test",
) -> dict:
    """Compute a full set of evaluation metrics at a given threshold.

    Summary:
        Computes ROC AUC, PR AUC (average precision), confusion matrix
        components (TP, TN, FP, FN), sensitivity (recall), precision,
        and total business cost. Returns a flat dictionary suitable for
        appending to a results log.

    Args:
        y_true: 1-D array of true binary labels (0/1).
        y_prob: 1-D array of predicted probabilities for class 1.
        threshold: Classification cutoff used to derive binary predictions.
        c_fp: Cost per false positive. Defaults to 1.0.
        c_fn: Cost per false negative. Defaults to 5.0.
        split_label: Label identifying the dataset split (e.g. "train", "test").
            Defaults to "test".

    Returns:
        Dictionary with keys:
            split, roc_auc, pr_auc, threshold, cost,
            tp, tn, fp, fn, sensitivity, precision, evaluated_on.
    """
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    cost        = c_fp * fp + c_fn * fn

    return {
        "split":        split_label,
        "roc_auc":      round(roc_auc_score(y_true, y_prob), 6),
        "pr_auc":       round(average_precision_score(y_true, y_prob), 6),
        "threshold":    round(threshold, 4),
        "cost":         cost,
        "tp":           tp,
        "tn":           tn,
        "fp":           fp,
        "fn":           fn,
        "sensitivity":  round(sensitivity, 4),
        "precision":    round(precision, 4),
        "evaluated_on": str(date.today()),
    }


# ==============================================================================
# Results persistence
# ==============================================================================

def save_metrics_to_excel(
    metrics: list[dict],
    output_path: str | Path,
) -> None:
    """Append a list of metric dictionaries to an Excel file.

    Summary:
        If the file already exists, the new rows are appended below the
        existing data. If it does not exist, a new file is created.
        Each row represents one evaluation run identified by split and
        evaluated_on date.

        NOTE: In a production environment this function would be replaced by
        an append to a metrics tracking table in the database (e.g. via
        SQLAlchemy). The schema would be identical to the DataFrame columns.
        Example:
            metrics_df.to_sql("schema.model_metrics", engine,
                               if_exists="append", index=False)

    Args:
        metrics: List of metric dictionaries, each produced by compute_metrics.
        output_path: Path to the target Excel file (.xlsx).

    Returns:
        None
    """
    output_path = Path(output_path)
    new_df = pd.DataFrame(metrics)

    if output_path.exists():
        existing = pd.read_excel(output_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        log_info(
            f"Appending {len(new_df)} row(s) to existing metrics file "
            f"({len(existing)} existing rows)."
        )
    else:
        combined = new_df
        log_info(f"Creating new metrics file with {len(new_df)} row(s).")

    combined.to_excel(output_path, index=False)
    log_info(f"Metrics saved to: {output_path}")