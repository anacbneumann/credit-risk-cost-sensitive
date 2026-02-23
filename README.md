# Credit Risk Cost-Sensitive Modeling (Give Me Some Credit)

This repository contains an end-to-end, **cost-sensitive** credit risk modeling project based on the classic *Give Me Some Credit* dataset.  
The goal is to explore the data, clean and validate it, engineer features, train models, and evaluate decisions using a **business-aware cost perspective** (not only accuracy).

Link Kaggle: https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset?resource=download

## Project Goals
- Build a solid **EDA** with strong data quality checks (missing values, outliers, impossible values, duplicates).
- Prepare a clean and reproducible preprocessing pipeline.
- Train and compare baseline and stronger models (e.g., logistic regression, tree-based models).
- Evaluate models using metrics suitable for imbalanced classification (ROC-AUC, PR-AUC, sensitivity/recall, etc.).
- Add a **cost-sensitive layer** to support decision thresholds aligned with real-world risk trade-offs.
