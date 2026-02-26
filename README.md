# Credit Risk Cost-Sensitive Classification (Kaggle / GiveMeSomeCredit)

Credit risk classification project focused on **cost-sensitive decision making**.  
Goal: predict default (`target = default_2y`) and choose a **classification threshold** that minimises business cost.

## Business cost definition

We model the operational impact of classification errors using:

- **C_FP = 1** → Reject a good payer (false positive)
- **C_FN = 5** → Approve a defaulter (false negative, higher business loss)

Total cost:


Cost = C_FP * FP + C_FN*FN


A theoretical reference threshold from the cost ratio is:


t = C_FP / (C_FP+C_FN) 
t = 1/6 
~ approx 0.1667


> Note: the pipeline still searches the **empirical best threshold** on the held-out test set by sweeping thresholds (step = 0.01).

## Final model

The final pipeline trains an **XGBoost classifier** with fixed hyperparameters selected during experimentation / tuning:

- `n_estimators = 500`
- `max_depth = 3`
- `min_child_weight = 15`
- `learning_rate = 0.10`
- `random_state = 51`

The train/test split is stratified and reproducible (seed = 51).

---

## Repository structure

Current structure (expected):

```text
credit-risk-cost-sensitive/
├─ database/
│  ├─ cs-training.csv
│  ├─ cs-test.csv
│  ├─ data_dict.csv
│  ├─ processed/
│  │  ├─ cs_train_raw.csv
│  │  ├─ cs_train_clean.csv
│  │  └─ cs_train_features.csv
│  └─ splits/
│     ├─ train.csv
│     └─ test.csv
├─ models/
│  └─ xgb_final.joblib
├─ reports/
│  ├─ feature_summary.xlsx
│  └─ model_metrics.xlsx
├─ notebooks/
│  └─ (EDA / experiments in R)
├─ src/
│  ├─ 01_preprocessing/
│  │  └─ 01_preprocessing.py
│  ├─ 02_features/
│  │  └─ 02_feature.py
│  ├─ 03_train/
│  │  └─ 03_train.py
│  ├─ 04_output/
│  │  └─ 04_evaluate.py
│  └─ auxiliary/
│     ├─ preprocessing_utils.py
│     ├─ feature_utils.py
│     └─ model_utils.py
├─ requirements.txt
└─ README.md
```

---

## Requirements

- **Python**: 3.10+ recommended
- Main libraries:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `xgboost`
  - `joblib`
  - `openpyxl` (Excel outputs - only for github purposes)

### Setup (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

```

---

## Data

This project expects the Kaggle “GiveMeSomeCredit” dataset format.

https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset/data

Place the files in:

- `database/cs-training.csv` (required)
- `database/cs-test.csv` (optional; not required for the 01→04 pipeline)
- `database/data_dict.csv` (optional; for reference)

**Recommendation:** do not version raw datasets in Git.

---

## Pipeline overview (01 → 04)

High-level flow:

1) **01_preprocessing**  
   Load raw CSV → rule-based cleaning → save `raw` + `clean`

2) **02_feature**  
   Load clean CSV → build analytical features → drop EDA-decided columns → save final modelling dataset + Excel feature summary

3) **03_train**  
   Load features CSV → stratified split → train final XGBoost → persist model + save split CSVs

4) **04_evaluate**  
   Load model + splits → predict probabilities → find cost-optimal threshold on test → compute metrics (train & test) → export to Excel

---

## Quickstart (run from project root)

> Important: Run commands from the repository root so default relative paths work.

### 1) Preprocessing

Default:

```bash
python src/01_preprocessing/01_preprocessing.py
```

With explicit paths:

```bash
python src/01_preprocessing/01_preprocessing.py database/cs-training.csv database/processed
```

**Outputs:**
- `database/processed/cs_train_raw.csv`
- `database/processed/cs_train_clean.csv`

---

### 2) Feature engineering

Default:

```bash
python src/02_features/02_feature.py
```

With explicit paths:

```bash
python src/02_features/02_feature.py database/processed/cs_train_clean.csv database/processed reports
```

**Outputs:**
- `database/processed/cs_train_features.csv`
- `reports/feature_summary.xlsx`

---

### 3) Train model (XGBoost)

Default:

```bash
python src/03_train/03_train.py
```

With explicit paths:

```bash
python src/03_train/03_train.py database/processed/cs_train_features.csv models database/splits
```

**Outputs:**
- `models/xgb_final.joblib`
- `database/splits/train.csv`
- `database/splits/test.csv`

> Note: the saved artefact is a `joblib` object containing:
> - the trained model pipeline
> - the list of feature names used during training (for safe evaluation)

---

### 4) Evaluate (cost-sensitive)

Default:

```bash
python src/04_output/04_evaluate.py
```

With explicit paths:

```bash
python src/04_output/04_evaluate.py models/xgb_final.joblib database/splits/test.csv database/splits/train.csv reports/model_metrics.xlsx
```

**Output:**
- `reports/model_metrics.xlsx`

This step:
- finds the **cost-optimal threshold** via grid search (0 → 1, step 0.01)
- computes metrics for both **train** and **test**
- logs a confusion matrix summary and the best threshold/cost

---

## Outputs & artefacts

### CSV artefacts
- `database/processed/cs_train_raw.csv`  
- `database/processed/cs_train_clean.csv`  
- `database/processed/cs_train_features.csv`  
- `database/splits/train.csv`  
- `database/splits/test.csv`

### Model artefact
- `models/xgb_final.joblib`

### Excel reports
- `reports/feature_summary.xlsx`  
  Feature-level summary for audit and modelling traceability.

- `reports/model_metrics.xlsx`  
  Metrics table including (per split):
  - ROC AUC
  - PR AUC (Average Precision)
  - threshold used
  - TP, TN, FP, FN
  - sensitivity (recall)
  - precision
  - total business cost
  - run date

---

## Logging

All scripts implement **step-by-step logs**, including:
- data loading/saving
- pipeline stage start/finish
- key shapes / target rate checks
- threshold search result (best threshold + cost)
- failures with stack traces (`logger.exception`)

---

## Notebooks (EDA / experiments)

The `notebooks/` folder contains exploratory analysis and model experimentation (in **R**), including:
- missingness and outlier checks
- baseline models (LR / RF / XGB)
- cost-vs-threshold plots
- lightweight tuning experiments

These notebooks are **not part of the production pipeline** (01→04).

---

## Reproducibility notes

- Train/test split is deterministic (seed = 51).
- Model hyperparameters are fixed in the training script.
- Metrics may still vary slightly across environments due to version differences.


---

## Author

Developed by **Ana C B Neumann**.  
Suggestions and improvements: open an issue.
