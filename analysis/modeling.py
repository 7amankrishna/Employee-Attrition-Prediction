"""Predictive modeling study: can attrition be predicted, and from what?

Protocol (same for every employee-level dataset):
  * stratified 80/20 train/test split (seed 20261101)
  * preprocessing fit on train only (scale numerics, one-hot categoricals)
  * 5-fold stratified CV on train for Logistic Regression / Random Forest / XGBoost
  * best model by CV ROC-AUC refit on train, evaluated once on the held-out test
  * metrics: ROC-AUC, PR-AUC, accuracy, precision, recall, F1, Brier
  * permutation importance + SHAP on the test set
  * ROC/PR/calibration/confusion-matrix figures

Run:  python3 analysis/modeling.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from common import (FIGURES, RNG, RESULTS, load_academic, load_av, load_hr,
                    load_ibm, save_json)

IVORY, AUBERGINE, CORAL, SAGE, INK = "#faf6ef", "#5b2333", "#e0655a", "#7a9b76", "#2b2b2b"
plt.rcParams.update({
    "figure.facecolor": IVORY, "axes.facecolor": IVORY, "savefig.facecolor": IVORY,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

SCORING = ["roc_auc", "average_precision", "f1", "recall", "precision"]


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print("  figure:", name)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])


def make_models(pos_weight: float) -> dict:
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, n_jobs=-1,
                                     class_weight="balanced_subsample", random_state=RNG),
        "xgb": XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.08, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
                             n_jobs=-1, random_state=RNG),
    }


def study(name: str, df: pd.DataFrame, feature_cols: list[str], target: str = "left") -> dict:
    print(f"[model] {name}  n={len(df)}  features={len(feature_cols)}")
    X = df[feature_cols].copy()
    y = df[target].astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG)
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())

    pre = make_preprocessor(Xtr)
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
    cv_table = {}
    fitted = {}
    for mname, model in make_models(pos_weight).items():
        pipe = Pipeline([("prep", pre), ("clf", model)])
        scores = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=SCORING, n_jobs=-1)
        cv_table[mname] = {m: {"mean": float(scores[f"test_{m}"].mean()),
                               "std": float(scores[f"test_{m}"].std())} for m in SCORING}
        print(f"  cv {mname:7s} roc_auc={cv_table[mname]['roc_auc']['mean']:.3f} "
              f"pr_auc={cv_table[mname]['average_precision']['mean']:.3f}")
        fitted[mname] = pipe.fit(Xtr, ytr)

    best = max(cv_table, key=lambda m: (cv_table[m]["roc_auc"]["mean"], cv_table[m]["average_precision"]["mean"]))
    print(f"  best by CV ROC-AUC: {best}")
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
    }

    # ROC + PR curves for all three models on the same test split
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mname, p in fitted.items():
        pr = p.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, pr)
        axes[0].plot(fpr, tpr, label=f"{mname} (AUC={roc_auc_score(yte, pr):.3f})")
        prec, rec, _ = precision_recall_curve(yte, pr)
        axes[1].plot(rec, prec, label=f"{mname} (AP={average_precision_score(yte, pr):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color=INK, linewidth=0.7)
    axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
    axes[0].set_title(f"{name}: ROC curves (test)", loc="left", color=AUBERGINE)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(yte.mean(), color=INK, linestyle="--", linewidth=0.7, label=f"base rate {yte.mean():.2f}")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
    axes[1].set_title(f"{name}: precision-recall curves (test)", loc="left", color=AUBERGINE)
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, f"fig_{name}_roc_pr.png")

    # confusion matrix of the best model
    cm = confusion_matrix(yte, pred)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("t", [IVORY, AUBERGINE]))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color=IVORY if cm[i, j] > cm.max() / 2 else INK, fontsize=13)
    ax.set_xticks([0, 1], ["predicted stay", "predicted leave"])
    ax.set_yticks([0, 1], ["actually stayed", "actually left"])
    ax.set_title(f"{name}: confusion matrix ({best})", loc="left", color=AUBERGINE)
    _save(fig, f"fig_{name}_confusion.png")

    # calibration curve of the best model
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, 9)
    frac, mean = [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 0:
            frac.append(yte[m].mean()); mean.append(proba[m].mean())
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.plot([0, 1], [0, 1], "--", color=INK, linewidth=0.7)
    ax.plot(mean, frac, "o-", color=CORAL, label=f"{best}")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed attrition fraction")
    ax.set_title(f"{name}: calibration (test)", loc="left", color=AUBERGINE)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, f"fig_{name}_calibration.png")

    # permutation importance (top 15)
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(pipe, Xte, yte, n_repeats=8, scoring="roc_auc", n_jobs=-1, random_state=RNG)
    order = np.argsort(pi.importances_mean)[::-1][:15]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh([feature_cols[i] if False else Xte.columns[i] for i in order][::-1],
            pi.importances_mean[order][::-1], color=SAGE, edgecolor=INK, linewidth=0.3)
    ax.set_xlabel("permutation importance (drop in ROC-AUC)")
    ax.set_title(f"{name}: top features ({best})", loc="left", color=AUBERGINE)
    _save(fig, f"fig_{name}_importance.png")
    perm_top = [{"feature": Xte.columns[i], "importance": float(pi.importances_mean[i]),
                 "std": float(pi.importances_std[i])} for i in order]

    # SHAP on a sample of the test set
    Xte_t = pipe.named_steps["prep"].transform(Xte)
    feat_names = pipe.named_steps["prep"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    sample = min(600, Xte_t.shape[0])
    Xs = shap.sample(Xte_t, sample, random_state=RNG) if hasattr(shap, "sample") else Xte_t[:sample]
    try:
        if hasattr(clf, "get_booster") or hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
        else:
            explainer = shap.LinearExplainer(clf, Xte_t)
        sv = explainer.shap_values(Xs)
        if isinstance(sv, list):
            sv = sv[1]  # positive class
        elif getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1] if sv.shape[-1] == 2 else sv[..., 0]
        sv = np.asarray(sv)
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:15]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].barh([str(feat_names[i]) for i in order][::-1], mean_abs[order][::-1],
                     color=SAGE, edgecolor=INK, linewidth=0.3)
        axes[0].set_xlabel("mean |SHAP value|")
        axes[0].set_title(f"{name}: SHAP feature impact", loc="left", color=AUBERGINE)
        plt.sca(axes[1])
        shap.summary_plot(sv, Xs, feature_names=feat_names, show=False, max_display=15)
        axes[1].set_title(f"{name}: SHAP beeswarm", loc="left", color=AUBERGINE)
        _save(fig, f"fig_{name}_shap.png")
        shap_top = [{"feature": str(feat_names[i]), "mean_abs_shap": float(mean_abs[i])}
                    for i in order]
    except Exception as exc:  # pragma: no cover
        print("  shap failed:", exc)
        shap_top = []

    return {
        "dataset": name, "n": int(len(df)), "n_test": int(len(yte)), "base_rate_test": float(yte.mean()),
        "features": feature_cols, "cv": cv_table, "best_model": best, "test": test_metrics,
        "permutation_importance_top15": perm_top, "shap_top15": shap_top,
    }


IBM_DROP = ["Attrition", "left", "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


def ibm_features(df):
    return [c for c in df.columns if c not in IBM_DROP]


def hr_features(df):
    return [c for c in df.columns if c != "left"]


def av_features(df):
    return [c for c in df.columns if c not in ("enrollee_id", "target", "left", "experience", "city")]


def academic_study() -> dict:
    print("[model] academic (faculty census)")
    fac, _ = load_academic()
    fac["title_2011"] = fac["title_2011"].fillna("(new in 2011)")
    X = fac[["sex", "title_2011"]].astype(str)
    y = fac["left"].to_numpy()
    pre = make_preprocessor(X)
    pipe = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
    scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    cv_table = {m: {"mean": float(scores[f"test_{m}"].mean()), "std": float(scores[f"test_{m}"].std())} for m in SCORING}
    print("  cv logreg roc_auc=%.3f pr_auc=%.3f" % (cv_table["roc_auc"]["mean"], cv_table["average_precision"]["mean"]))
    return {"dataset": "academic", "n": int(len(df := fac)), "features": ["sex", "title_2011"],
            "cv": {"logreg": cv_table}, "best_model": "logreg", "note": "title/sex only; no rich covariates available"}


def main():
    ibm, hr, av = load_ibm(), load_hr(), load_av()
    results = {
        "ibm": study("ibm", ibm, ibm_features(ibm)),
        "hr": study("hr", hr, hr_features(hr)),
        "av": study("av", av, av_features(av)),
        "academic": academic_study(),
    }
    save_json("model_all.json", results)
    print("saved reports/results/model_all.json")


if __name__ == "__main__":
    main()
