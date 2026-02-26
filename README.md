# Credit Risk Cost-Sensitive Classification (Kaggle / GiveMeSomeCredit)

Credit risk classification project focused on **cost-sensitive decision making**.  
Goal: predict default (`target = default_2y`) and choose a **classification threshold** that minimises business cost.

## Business cost definition

We model the operational impact of classification errors using:

- **C_FP = 1** → Reject a good payer (false positive)
- **C_FN = 5** → Approve a defaulter (false negative, higher business loss)

Total cost:

\[
\text{Cost} = C_{FP}\cdot FP + C_{FN}\cdot FN
\]

A theoretical reference threshold from the cost ratio is:

\[
t^* = \frac{C_{FP}}{C_{FP}+C_{FN}} = \frac{1}{6} \approx 0.1667
\]

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
