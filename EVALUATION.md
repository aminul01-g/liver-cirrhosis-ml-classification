# Reproducible evaluation — Scenario A

Measured on **2026-11-26** with seed 42, Python 3.14, scikit-learn 1.9.1.
Script: [`eval/eval_liver.py`](eval/eval_liver.py) · Raw output: [`eval_results_scenario_a.json`](eval_results_scenario_a.json)

This reproduces the notebook's Scenario A (severity columns `Fibrosis_Stage`,
`MELD_Score`, `Child_Pugh_Class` excluded), ordinal encoding, seed 42, 80/20
stratified holdout, and 5-fold cross-validation. The notebook uses 5 repeats;
this script uses 3 so it completes in seconds. All numbers below were produced by
running the script — none are copied from the notebook.

## Setup

- 100 patients, 32 features, 80 train / 20 holdout
- Class counts: No Cirrhosis 36, Compensated 22, Decompensated 42
- `RepeatedStratifiedKFold(5 splits × 3 repeats)`, `StandardScaler` inside each `Pipeline`

## Results

| Model | CV accuracy | CV macro F1 | CV ROC-AUC (OvR) | Holdout accuracy | Holdout macro F1 |
|---|---:|---:|---:|---:|---:|
| Extra Trees | 0.996 ± 0.016 | 0.995 | 1.000 | 1.00 | 1.00 |
| Random Forest | 0.988 ± 0.025 | 0.986 | 1.000 | 1.00 | 1.00 |
| Gradient Boosting | 0.979 ± 0.029 | 0.974 | 1.000 | 1.00 | 1.00 |
| Decision Tree | 0.954 ± 0.062 | 0.947 | 0.963 | 0.95 | 0.94 |
| Logistic Regression | 0.946 ± 0.055 | 0.931 | 0.997 | 0.95 | 0.94 |
| SVM (RBF) | 0.942 ± 0.053 | 0.923 | 0.998 | 1.00 | 1.00 |
| Gaussian Naive Bayes | 0.929 ± 0.064 | 0.905 | 0.970 | 0.90 | 0.88 |
| LDA | 0.688 ± 0.072 | 0.537 | 0.759 | 1.00 | 1.00 |

## Interpretation and limitations

- The holdout has only 20 patients: one patient is 5 percentage points of accuracy.
  LDA's 1.00 holdout accuracy alongside 0.69 CV accuracy is the clearest illustration
  of that instability and should not be presented as a good result.
- Near-perfect scores across several families are consistent with the dataset being
  small and likely synthetic. This is a methodological case study, not a clinical
  model and not a medical device.
- The exact values depend on scikit-learn version (evaluated with 1.9.1); the notebook
  targeted ≥1.3.
- No external validation set was used.

## Reproduce

```bash
python -m pip install numpy pandas scikit-learn
python eval/eval_liver.py
```

The script writes `eval_results_scenario_a.json`; output is deterministic for the
pinned seed within a given scikit-learn version.
