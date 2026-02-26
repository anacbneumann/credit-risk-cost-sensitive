# ==============================
# 03_train.py
# ==============================
# Summary:
#   Trains the final XGBoost model with fixed hyperparameters selected
#   during EDA/tuning (tree_depth=3, min_n=15, learn_rate=0.1, trees=500).
#   Loads the preprocessed dataset, performs a stratified train/test split,
#   fits the model on the training portion, and serialises both the trained
#   pipeline and the split indices for reproducible evaluation downstream.
#
#   The test set produced here is held out and used exclusively in 03_evaluate.py.
#   Re-running this script with the same seed reproduces the exact same split.
#
# Inputs:
#   - Processed CSV (default: database/processed/cs_train_processed.csv)
#
# Outputs:
#   - Serialised model pipeline  (default: models/xgb_final.joblib)
#   - Train CSV                  (default: database/splits/train.csv)
#   - Test CSV                   (default: database/splits/test.csv)
#
# Typical usage:
#   python 03_train.py
#   python 03_train.py database/processed/cs_train_processed.csv models database/splits
# ==============================

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import sys
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

# ── Internal ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.auxiliary.model_utils import log_info, prepare_features


# ==============================================================================
# Centralized parameters
# ==============================================================================

PARAMS: dict = {
    # XGBoost hyperparameters (selected via CV cost-minimisation in EDA)
    "n_estimators":  500,
    "max_depth":     3,
    "min_child_weight": 15,   # equivalent to tidymodels min_n
    "learning_rate": 0.10,

    # XGBoost runtime settings
    "eval_metric":   "logloss",
    "use_label_encoder": False,
    "random_state":  51,

    # Split settings
    "test_size":     0.20,
    "split_seed":    51,

    # Column settings
    "target_col":    "default_2y",
    "drop_cols":     ["id"],
}


# ==============================================================================
# CLI argument parsing
# ==============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for input/output paths.

    Args:
        argv: Argument list. Defaults to sys.argv[1:] when None.

    Returns:
        Namespace with input_path, model_dir, and splits_dir attributes.
    """
    parser = argparse.ArgumentParser(
        description="Train the final XGBoost credit risk model."
    )
    parser.add_argument(
        "input_path", nargs="?",
        default="database/processed/cs_train_features.csv",
        help="Path to the preprocessed CSV (default: database/processed/cs_train_features.csv).",
    )
    parser.add_argument(
        "model_dir", nargs="?",
        default="models",
        help="Directory to save the serialised model (default: models).",
    )
    parser.add_argument(
        "splits_dir", nargs="?",
        default="database/splits",
        help="Directory to save train/test split CSVs (default: database/splits).",
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


# ==============================================================================
# Pipeline
# ==============================================================================

def main(argv: list[str] | None = None) -> dict:
    """Execute the training pipeline.

    Summary:
        Loads the processed dataset, separates features from target,
        performs a stratified 80/20 split, fits an XGBClassifier with
        fixed hyperparameters on the training set, and persists the model
        and split DataFrames to disk.

    Args:
        argv: Optional CLI argument list forwarded to parse_args.

    Returns:
        Dictionary with keys "model", "X_train", "X_test", "y_train", "y_test".
    """
    cfg = parse_args(argv)
    model_dir  = ensure_dir(cfg.model_dir)
    splits_dir = ensure_dir(cfg.splits_dir)

    # ------------------------------------------------------------------
    # Load processed data
    # ------------------------------------------------------------------
    log_info(f"Loading processed dataset: {cfg.input_path}")
    df = pd.read_csv(cfg.input_path)
    log_info(f"Dataset loaded. Rows: {len(df)} | Cols: {len(df.columns)}")

    # ------------------------------------------------------------------
    # Prepare features
    # ------------------------------------------------------------------
    X, y = prepare_features(
        df,
        target_col=PARAMS["target_col"],
        drop_cols=PARAMS["drop_cols"],
    )
    log_info(f"Feature matrix shape: {X.shape}")

    # ------------------------------------------------------------------
    # Stratified train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=PARAMS["test_size"],
        stratify=y,
        random_state=PARAMS["split_seed"],
    )
    log_info(
        f"Split complete. Train: {len(X_train)} rows | Test: {len(X_test)} rows. "
        f"Train default rate: {y_train.mean():.4f} | "
        f"Test default rate: {y_test.mean():.4f}."
    )

    # ------------------------------------------------------------------
    # Save splits (for reproducible evaluation in 03_evaluate.py)
    # ------------------------------------------------------------------
    train_out = splits_dir / "train.csv"
    test_out  = splits_dir / "test.csv"

    X_train.assign(**{PARAMS["target_col"]: y_train.values}).to_csv(train_out, index=False)
    X_test.assign(**{PARAMS["target_col"]: y_test.values}).to_csv(test_out, index=False)
    log_info(f"Train split saved: {train_out}")
    log_info(f"Test split saved:  {test_out}")

    # ------------------------------------------------------------------
    # Build and fit XGBoost model
    # ------------------------------------------------------------------
    log_info(
        f"Training XGBClassifier with params: "
        f"n_estimators={PARAMS['n_estimators']}, "
        f"max_depth={PARAMS['max_depth']}, "
        f"min_child_weight={PARAMS['min_child_weight']}, "
        f"learning_rate={PARAMS['learning_rate']}."
    )

    model = XGBClassifier(
        n_estimators=PARAMS["n_estimators"],
        max_depth=PARAMS["max_depth"],
        min_child_weight=PARAMS["min_child_weight"],
        learning_rate=PARAMS["learning_rate"],
        eval_metric=PARAMS["eval_metric"],
        random_state=PARAMS["random_state"],
        verbosity=0,
    )

    model.fit(X_train, y_train)
    log_info("Model training complete.")

    # Quick sanity check on train set
    train_proba = model.predict_proba(X_train)[:, 1]
    log_info(
        f"Train set predicted probability — "
        f"mean: {train_proba.mean():.4f} | "
        f"min: {train_proba.min():.4f} | "
        f"max: {train_proba.max():.4f}."
    )

    # ------------------------------------------------------------------
    # Serialise model
    # ------------------------------------------------------------------
    model_path = model_dir / "xgb_final.joblib"
    joblib.dump({"model": model, "feature_names": list(X_train.columns)}, model_path)
    log_info(f"Model serialised to: {model_path}")

    return {
        "model":   model,
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
    }


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    main(sys.argv[1:])