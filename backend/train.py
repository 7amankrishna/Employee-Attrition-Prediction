#!/usr/bin/env python3
"""Train the handoff attrition model and write backend/artifacts/.

Usage (from the repository root):

    .venv/bin/python backend/train.py
    .venv/bin/python backend/train.py --data public/data/ibm-hr.csv --out backend/artifacts

Artifacts written to --out (gitignored):
    model.joblib     fitted sklearn Pipeline + metadata (loaded by app.py)
    metrics.json     CV + held-out test metrics, confusion matrix, report
    model_card.json  human-readable model card (version, data, metrics, limits)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from model import (FEATURES, MODEL_VERSION, SEED, build_pipeline, candidate_models,
                   load_contract_frame)

BACKEND = Path(__file__).resolve().parent
DEFAULT_DATA = BACKEND.parent / "public" / "data" / "ibm-hr.csv"
DEFAULT_OUT = BACKEND / "artifacts"
SCORING = ["roc_auc", "average_precision", "f1", "recall", "precision"]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_model(data_path: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT, verbose: bool = True) -> dict:
    df = load_contract_frame(data_path)
    X = df[FEATURES]
    y = (df["Attrition"] == "Yes").astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_table, fitted = {}, {}
    for name, clf in candidate_models(pos_weight).items():
        pipe = build_pipeline(clf)
        scores = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=SCORING, n_jobs=-1)
        cv_table[name] = {m: {"mean": float(scores[f"test_{m}"].mean()),
                              "std": float(scores[f"test_{m}"].std())} for m in SCORING}
        if verbose:
            print(f"  cv {name:7s} roc_auc={cv_table[name]['roc_auc']['mean']:.3f} "
                  f"pr_auc={cv_table[name]['average_precision']['mean']:.3f}")
        fitted[name] = pipe.fit(Xtr, ytr)

    best = max(cv_table, key=lambda m: (cv_table[m]["roc_auc"]["mean"], cv_table[m]["average_precision"]["mean"]))
    pipe = fitted[best]
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {
        "roc_auc": float(roc_auc_score(yte, proba)),
        "pr_auc": float(average_precision_score(yte, proba)),
        "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred)),
        "f1": float(f1_score(yte, pred)),
        "brier": float(brier_score_loss(yte, proba)),
        "base_rate": float(yte.mean()),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        "classification_report": classification_report(yte, pred, target_names=["No", "Yes"], output_dict=True),
    }
    if verbose:
        print(f"  best={best}  test roc_auc={test_metrics['roc_auc']:.3f} "
              f"pr_auc={test_metrics['pr_auc']:.3f} recall={test_metrics['recall']:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    version = MODEL_VERSION
    background = pipe.named_steps["prep"].transform(Xtr)
    background = background[: min(100, background.shape[0])]
    artifact = {
        "pipeline": pipe,
        "model_type": best,
        "model_version": version,
        "features": FEATURES,
        "background": background,  # SHAP masker background for the linear explainer
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"path": str(data_path), "sha256": sha256_of(data_path),
                    "n": int(len(df)), "positive_rate": float(y.mean())},
    }
    joblib.dump(artifact, out_dir / "model.joblib")

    metrics = {"model_version": version, "model_type": best, "seed": SEED,
               "n_total": int(len(df)), "n_train": int(len(ytr)), "n_test": int(len(yte)),
               "cv": cv_table, "test": test_metrics}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    model_card = {
        "model_version": version,
        "model_type": best,
        "created_at": artifact["trained_at"],
        "task": "binary classification: P(Attrition=Yes)",
        "features": FEATURES,
        "training_data": artifact["dataset"],
        "data_warning": ("Trained on the synthetic IBM HR sample bundled with this repo "
                         "(fictional data created by IBM data scientists). Educational demo only - "
                         "retrain on real, consented HR data before any operational use."),
        "metrics": {"cv": cv_table, "held_out_test": {k: v for k, v in test_metrics.items() if k != "classification_report"}},
        "responsible_use": ("Use probabilities for triage/conversation only, never as an employment "
                            "decision. No calibration guarantees; 0.5 threshold is a demo convention. "
                            "Run fairness audits before any real deployment."),
    }
    (out_dir / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    if verbose:
        print(f"  wrote {out_dir}/model.joblib, metrics.json, model_card.json")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    print(f"training on {args.data}")
    train_model(args.data, args.out)


if __name__ == "__main__":
    main()
