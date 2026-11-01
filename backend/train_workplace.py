"""Train/evaluate the separate workplace-turnover model on the HRIS dataset.

Run from the repository root with the backend requirements installed:
    python backend/train_workplace.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hr-comma-sep.csv"
OUT = Path(__file__).resolve().parent / "artifacts" / "workplace-turnover"
MODEL_VERSION = "hris-turnover-1.0.0"
SEED = 20261101
FEATURES = ["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
            "time_spend_company", "Work_accident", "promotion_last_5years", "Department", "salary"]
NUMERIC = FEATURES[:-2]
CATEGORICAL = FEATURES[-2:]
SCORING = ["roc_auc", "average_precision", "f1", "recall", "precision"]


def build_pipeline(classifier):
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("clf", classifier)])


def train_model(data_path: Path = DATA, out_dir: Path = OUT, verbose: bool = True):
    df = pd.read_csv(data_path)
    missing = sorted(set(FEATURES + ["left"]) - set(df.columns))
    if missing:
        raise ValueError(f"HRIS dataset is missing required columns: {missing}")
    X = df[FEATURES].copy()
    y = df["left"].astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())
    candidates = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=350, min_samples_leaf=4, n_jobs=-1,
                                      class_weight="balanced_subsample", random_state=SEED),
        "xgb": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.85,
                             colsample_bytree=0.85, scale_pos_weight=pos_weight, eval_metric="logloss",
                             n_jobs=-1, random_state=SEED),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_table = {}
    fitted = {}
    for name, classifier in candidates.items():
        pipeline = build_pipeline(classifier)
        scores = cross_validate(pipeline, Xtr, ytr, cv=cv, scoring=SCORING, n_jobs=-1)
        cv_table[name] = {metric: {"mean": float(scores[f"test_{metric}"].mean()),
                                  "std": float(scores[f"test_{metric}"].std())} for metric in SCORING}
        fitted[name] = pipeline.fit(Xtr, ytr)
        if verbose:
            print(f"cv {name}: ROC-AUC {cv_table[name]['roc_auc']['mean']:.3f}")
    best = max(cv_table, key=lambda name: (cv_table[name]["roc_auc"]["mean"],
                                           cv_table[name]["average_precision"]["mean"]))
    pipeline = fitted[best]
    probability = pipeline.predict_proba(Xte)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    test = {
        "roc_auc": float(roc_auc_score(yte, probability)),
        "pr_auc": float(average_precision_score(yte, probability)),
        "accuracy": float(accuracy_score(yte, predicted)),
        "precision": float(precision_score(yte, predicted, zero_division=0)),
        "recall": float(recall_score(yte, predicted, zero_division=0)),
        "f1": float(f1_score(yte, predicted, zero_division=0)),
        "brier": float(brier_score_loss(yte, probability)),
        "base_rate": float(yte.mean()),
        "confusion_matrix": confusion_matrix(yte, predicted).tolist(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(timezone.utc).isoformat()
    background = pipeline.named_steps["prep"].transform(Xtr.iloc[:min(100, len(Xtr))])
    artifact = {
        "pipeline": pipeline, "model_type": best, "model_id": "workplace-turnover",
        "model_version": MODEL_VERSION, "features": FEATURES, "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL, "background": background,
        "trained_at": trained_at,
        "dataset": {"name": "HRIS workplace turnover", "path": str(data_path),
                    "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    "n": int(len(df)), "n_train": int(len(ytr)), "n_test": int(len(yte)),
                    "positive_rate": float(y.mean())},
    }
    dump(artifact, out_dir / "model.joblib")
    metrics = {"model_id": "workplace-turnover", "model_version": MODEL_VERSION,
               "model_type": best, "seed": SEED, "n_total": int(len(df)),
               "n_train": int(len(ytr)), "n_test": int(len(yte)), "cv": cv_table, "test": test}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    card = {"model_id": "workplace-turnover", "model_version": MODEL_VERSION,
            "target": "P(left=1)", "features": FEATURES, "training_data": artifact["dataset"],
            "metrics": {"cv": cv_table, "held_out_test": test},
            "warning": "One anonymized company HRIS dataset; not a general population estimate. Educational use only."}
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    if verbose:
        print(f"selected {best}; held-out ROC-AUC {test['roc_auc']:.3f}; saved {out_dir}")
    return metrics


if __name__ == "__main__":
    train_model()
