# 🩺 liver-cirrhosis-ml-classification

> Multi-class prediction of liver cirrhosis status (*No Cirrhosis*, *Compensated*, *Decompensated*) using 13 machine learning models, with leak-free preprocessing, repeated cross-validation, and a leakage sensitivity analysis.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)
![Models](https://img.shields.io/badge/models-13-green)

## 📌 Overview

This project predicts a patient's cirrhosis status from demographics, lifestyle factors, laboratory values, and clinical signs. It benchmarks 13 classifiers inside scikit-learn `Pipeline`s and compares them with repeated stratified cross-validation plus a held-out test set.

**Target classes:** `No_Cirrhosis` · `Compensated` · `Decompensated`

## 📂 Repository structure

```
.
├── liver_cirrhosis_ml.ipynb        # Full notebook: EDA, preprocessing, models, evaluation
├── liver_cirrhosis_dataset.csv     # Dataset (100 patients, 37 columns)
├── requirements.txt                # Python dependencies
└── README.md
```

## 📊 Dataset

- **100 patients**, 37 columns (35 candidate features after dropping `Patient_ID` and the target)
- **Demographics & lifestyle:** age, gender, BMI, alcohol consumption, smoking status
- **History:** hepatitis B/C, diabetes, hypertension, family history
- **Labs:** albumin, bilirubin, prothrombin time, platelets, AST, ALT, ALP, GGT, sodium, creatinine, INR, copper, triglycerides, cholesterol
- **Clinical signs:** ascites, hepatic encephalopathy, varices, spider angiomas, edema, fatigue score, jaundice, spleen size
- **Severity scores:** fibrosis stage, MELD score, Child-Pugh class
- **Class balance:** Decompensated 42 · No_Cirrhosis 36 · Compensated 22

## 🔧 Methodology

**Preprocessing**
1. Dropped the `Patient_ID` identifier.
2. Kept `"None"` in `Alcohol_Consumption` as a real category (non-drinkers). A default `pd.read_csv` silently converts it to `NaN`.
3. Ordinal encoding with clinical order (alcohol, smoking, spleen size, fibrosis stage, Child-Pugh class) and binary encoding for gender.
4. Stratified 80/20 train/test split.
5. `StandardScaler` fitted inside each pipeline, so there is no leakage into validation folds.

**Leakage handling.** `Fibrosis_Stage`, `MELD_Score` and `Child_Pugh_Class` are severity assessments tied to the diagnosis. The main experiment (**Scenario A**) excludes them. **Scenario B** adds them back to measure their effect.

**Models (13):** Logistic Regression · KNN · SVM (RBF) · Decision Tree · Random Forest · Extra Trees · Gradient Boosting · HistGradient Boosting · AdaBoost · Bagging · Gaussian Naive Bayes · LDA · MLP Neural Network

**Evaluation**
- Repeated stratified 5-fold CV (5 repeats = 25 fits per model) on the training set
- Held-out test set (20 patients)
- Metrics: accuracy, balanced accuracy, macro/weighted precision, recall and F1, ROC-AUC (OvR), MCC, Cohen's κ
- Confusion matrices, ROC curves, per-class report, permutation importance

## 🏆 Results (Scenario A: no severity scores)

| Model | CV Accuracy | CV Macro-F1 | Test Accuracy |
|---|---|---|---|
| **Extra Trees** | 0.998 | 0.997 | 1.00 |
| Random Forest | 0.990 | 0.989 | 1.00 |
| AdaBoost | 0.990 | 0.988 | 0.95 |
| Bagging | 0.988 | 0.986 | 1.00 |
| Gradient Boosting | 0.983 | 0.979 | 1.00 |
| HistGradient Boosting | 0.978 | 0.974 | 0.90 |
| Decision Tree | 0.965 | 0.959 | 0.95 |
| Logistic Regression | 0.955 | 0.943 | 0.95 |
| MLP Neural Network | 0.948 | 0.935 | 0.95 |
| SVM (RBF) | 0.943 | 0.924 | 1.00 |
| KNN | 0.940 | 0.918 | 0.95 |
| Gaussian Naive Bayes | 0.928 | 0.901 | 0.90 |
| LDA | 0.715 | 0.563 | 1.00 |

Adding the three severity scores (Scenario B) improved CV F1 by only about 0 to 0.03, because the remaining features already separate the classes almost perfectly.

## ⚠️ Limitations

- **Small sample (n = 100, test n = 20).** One misclassified patient shifts test accuracy by 5 points, so rely on the cross-validated numbers.
- **Unusually clean data.** Many single features correlate above 0.9 with the target, which suggests synthetic or curated data. The near-perfect scores reflect this dataset and are **not** a clinical claim.
- **LDA instability.** Its low CV score comes from an ill-conditioned covariance estimate (many correlated features, few rows per fold). Shrinkage LDA would address it.
- **Not a medical device.** This is an educational and research project and must not be used for diagnosis.

## 🚀 Getting started

```bash
git clone https://github.com/<your-username>/liver-cirrhosis-ml-classification.git
cd liver-cirrhosis-ml-classification

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook liver_cirrhosis_ml.ipynb
```

Run all cells; the notebook takes a few minutes on a laptop. Keep the CSV in the same folder as the notebook.

## 🔮 Possible next steps

- Hyperparameter tuning with nested cross-validation
- Shrinkage LDA and gradient-boosting libraries (XGBoost, LightGBM, CatBoost)
- SHAP explanations and probability calibration
- External validation on an independent cohort

