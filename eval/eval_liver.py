#!/usr/bin/env python3
"""Reproducible Scenario A benchmark for liver-cirrhosis-ml-classification.

Mirrors the notebook's preprocessing (ordinal maps, keep_default_na=False),
seed 42, 80/20 stratified split and RepeatedStratifiedKFold, but uses a smaller
repeat count so the whole benchmark runs in seconds. Writes measured results.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "liver_cirrhosis_dataset.csv").exists() else HERE.parent

df = pd.read_csv(ROOT / "liver_cirrhosis_dataset.csv", keep_default_na=False, na_values=[""])
data = df.drop(columns=["Patient_ID"]).copy()
ordinal_maps = {
    "Alcohol_Consumption": {"None": 0, "Light": 1, "Moderate": 2, "Heavy": 3},
    "Smoking_Status": {"Never": 0, "Former": 1, "Current": 2},
    "Spleen_Size": {"Normal": 0, "Mild": 1, "Enlarged": 2},
    "Fibrosis_Stage": {"F0": 0, "F1": 1, "F2": 2, "F3": 3, "F4": 4},
    "Child_Pugh_Class": {"A": 0, "B": 1, "C": 2},
    "Gender": {"Female": 0, "Male": 1},
}
for col, mapping in ordinal_maps.items():
    data[col] = data[col].map(mapping)
assert data.isna().sum().sum() == 0
target_map = {"No_Cirrhosis": 0, "Compensated": 1, "Decompensated": 2}
data["Cirrhosis_Status"] = data["Cirrhosis_Status"].map(target_map)

severity = ["Fibrosis_Stage", "MELD_Score", "Child_Pugh_Class"]
features_a = [c for c in data.columns if c not in severity + ["Cirrhosis_Status"]]
y = data["Cirrhosis_Status"]
Xtr_full, Xte_full, ytr, yte = train_test_split(
    data, y, test_size=0.20, stratify=y, random_state=SEED)
Xtr, Xte = Xtr_full[features_a], Xte_full[features_a]

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
    "SVM (RBF)": SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    "Gaussian Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
}
scoring = {"accuracy": "accuracy", "f1": "f1_macro",
           "balanced_accuracy": "balanced_accuracy", "roc_auc": "roc_auc_ovr"}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
rows = []
for name, estimator in models.items():
    pipe = Pipeline([("scale", StandardScaler()), ("model", estimator)])
    cvr = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=-1)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    prob = pipe.predict_proba(Xte)
    rows.append({
        "model": name,
        "cv_accuracy_mean": round(float(cvr["test_accuracy"].mean()), 4),
        "cv_accuracy_std": round(float(cvr["test_accuracy"].std()), 4),
        "cv_f1_macro_mean": round(float(cvr["test_f1"].mean()), 4),
        "cv_roc_auc_ovr_mean": round(float(cvr["test_roc_auc"].mean()), 4),
        "holdout_accuracy": round(float(accuracy_score(yte, pred)), 4),
        "holdout_balanced_accuracy": round(float(balanced_accuracy_score(yte, pred)), 4),
        "holdout_f1_macro": round(float(f1_score(yte, pred, average="macro")), 4),
        "holdout_roc_auc_ovr": round(float(roc_auc_score(yte, prob, multi_class="ovr", average="macro")), 4),
    })

result = {
    "evaluated_at": pd.Timestamp.utcnow().isoformat(),
    "seed": SEED,
    "scenario": "A (severity features excluded: Fibrosis_Stage, MELD_Score, Child_Pugh_Class)",
    "dataset": {"rows": int(len(data)), "features": len(features_a),
                "train": int(len(Xtr)), "holdout": int(len(Xte)),
                "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()}},
    "cv": "RepeatedStratifiedKFold 5-fold x 3 repeats",
    "models": rows,
}
out = ROOT / "eval_results_scenario_a.json"
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
