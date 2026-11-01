"""Export compact, reproducible logistic models for frontend-only inference.

The Python inference service remains available for the full selected pipeline.
These model cards let the Vercel frontend make a genuine trained baseline
prediction when no separate API host is configured, without shipping records.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261101
IBM_FEATURES = ["JobRole", "Department", "BusinessTravel", "MaritalStatus", "OverTime",
                "Age", "JobLevel", "MonthlyIncome", "JobSatisfaction", "WorkLifeBalance",
                "EnvironmentSatisfaction", "JobInvolvement", "YearsAtCompany",
                "YearsSinceLastPromotion", "YearsWithCurrManager", "YearsInCurrentRole",
                "TotalWorkingYears", "StockOptionLevel", "NumCompaniesWorked", "DistanceFromHome"]
IBM_CATEGORICAL = ["JobRole", "Department", "BusinessTravel", "MaritalStatus", "OverTime"]
HR_FEATURES = ["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
               "time_spend_company", "Work_accident", "promotion_last_5years", "Department", "salary"]
HR_CATEGORICAL = ["Department", "salary"]


def fit_and_export(model_id: str, path: Path, features: list[str], categorical: list[str], target: str,
                   positive_value, target_label: str, version: str):
    frame = pd.read_csv(path)
    X = frame[features].copy()
    y = (frame[target] == positive_value).astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    numeric = [key for key in features if key not in categorical]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    pipeline = Pipeline([("prep", preprocessor),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))])
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_test)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    classifier = pipeline.named_steps["clf"]
    transformed = preprocessor.transform(X_train)
    names = preprocessor.get_feature_names_out().tolist()
    mean_vector = np.asarray(transformed).mean(axis=0)
    coefficients = classifier.coef_[0]
    numeric_model = {}
    categorical_model = {}
    baseline = float(classifier.intercept_[0])
    num_transformer = preprocessor.named_transformers_["num"]
    scaler = num_transformer.named_steps["scale"]
    num_imputer = num_transformer.named_steps["imputer"]
    # StandardScaler centers at the column mean, which differs from the median
    # used to fill missing entries.
    numeric_means = dict(zip(numeric, scaler.mean_))
    numeric_scale = dict(zip(numeric, scaler.scale_))
    for index, key in enumerate(numeric):
        coefficient = float(coefficients[names.index(f"num__{key}")])
        numeric_model[key] = {"coefficient": coefficient,
                              "mean": float(numeric_means[key]), "scale": float(numeric_scale[key])}
    encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_imputer = preprocessor.named_transformers_["cat"].named_steps["imputer"]
    category_fill = dict(zip(categorical, cat_imputer.statistics_))
    offset = len(numeric)
    for key, categories in zip(categorical, encoder.categories_):
        weights = {}
        frequencies = {}
        expected = 0.0
        for category in categories:
            name = f"cat__{key}_{category}"
            idx = names.index(name)
            weight = float(coefficients[idx])
            frequency = float(mean_vector[idx])
            weights[str(category)] = weight
            frequencies[str(category)] = frequency
            expected += weight * frequency
        categorical_model[key] = {"categories": [str(x) for x in categories], "weights": weights,
                                   "frequencies": frequencies, "impute": str(category_fill[key]),
                                   "expected_weight": expected}
        baseline += expected
    metric = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "pr_auc": float(average_precision_score(y_test, probability)),
        "brier": float(brier_score_loss(y_test, probability)),
        "base_rate": float(y_test.mean()),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
    }
    card = {
        "model_id": model_id, "model_version": version, "model_type": "logistic regression",
        "target": target_label, "features": features, "numeric_features": numeric,
        "categorical_features": categorical, "dataset_entries": int(len(frame)),
        "training_entries": int(len(X_train)), "test_entries": int(len(X_test)),
        "seed": SEED, "threshold": 0.5, "metrics": metric,
        "data_kind": "synthetic IBM benchmark" if model_id == "ibm-attrition" else "real anonymized single-company HRIS",
        "warning": ("Synthetic benchmark output, not real IBM employees." if model_id == "ibm-attrition"
                    else "One anonymized company's HRIS data; not a general population estimate."),
        "baseline_log_odds": baseline, "numeric": numeric_model, "categorical": categorical_model,
    }
    out = ROOT / "public" / "data" / f"{model_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, separators=(",", ":")), encoding="utf-8")
    reference = pipeline.predict_proba(X_test.iloc[:10])[:, 1]
    # Check the compact formula against sklearn before publishing it.
    for row, expected in zip(X_test.iloc[:10].to_dict(orient="records"), reference):
        actual = client_probability(card, row)
        if abs(actual - float(expected)) > 1e-8:
            raise AssertionError(f"browser model formula mismatch: {actual} vs {expected}")
    print(f"{model_id}: n={len(frame)} train={len(X_train)} test={len(X_test)} "
          f"ROC-AUC={metric['roc_auc']:.4f} PR-AUC={metric['pr_auc']:.4f} -> {out}")
    return card


def client_probability(card: dict, profile: dict) -> float:
    logit = card["baseline_log_odds"]
    for key, entry in card["numeric"].items():
        logit += ((float(profile[key]) - entry["mean"]) / entry["scale"]) * entry["coefficient"]
    for key, entry in card["categorical"].items():
        value = str(profile.get(key) or entry["impute"])
        logit += entry["weights"].get(value, 0.0) - entry["expected_weight"]
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -35, 35)))


if __name__ == "__main__":
    fit_and_export("ibm-attrition", ROOT / "public" / "data" / "ibm-hr.csv", IBM_FEATURES,
                   IBM_CATEGORICAL, "Attrition", "Yes", "P(Attrition=Yes)", "ibm-logreg-1.0.0")
    fit_and_export("workplace-turnover", ROOT / "data" / "hr-comma-sep.csv", HR_FEATURES,
                   HR_CATEGORICAL, "left", 1, "P(left=1)", "hris-logreg-1.0.0")
