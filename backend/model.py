"""Model definition shared by train.py and app.py.

The prediction contract (docs/backend-contract.md) uses exactly these eight
features, with the same value ranges the frontend validates in src/data.js:

  JobRole                 one of ROLES
  OverTime                "Yes" | "No"
  MonthlyIncome           int 100..100000
  JobSatisfaction         int 1..4
  WorkLifeBalance         int 1..4
  YearsAtCompany          int 0..40
  YearsSinceLastPromotion int 0..40 and <= YearsAtCompany
  DistanceFromHome        int 1..100
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROLES = [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources",
]

CATEGORICAL_FEATURES = ["JobRole", "OverTime"]
NUMERIC_FEATURES = [
    "MonthlyIncome", "JobSatisfaction", "WorkLifeBalance",
    "YearsAtCompany", "YearsSinceLastPromotion", "DistanceFromHome",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES  # contract order

RANGES = {
    "MonthlyIncome": (100, 100000),
    "JobSatisfaction": (1, 4),
    "WorkLifeBalance": (1, 4),
    "YearsAtCompany": (0, 40),
    "YearsSinceLastPromotion": (0, 40),
    "DistanceFromHome": (1, 100),
}

SEED = 20261101
MODEL_VERSION = "1.0.0+20261101"


def load_contract_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES + ["Attrition"] if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing contract columns: {missing}")
    return df


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES),
    ])


def candidate_models(pos_weight: float) -> dict[str, object]:
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                     class_weight="balanced_subsample", random_state=SEED),
        "xgb": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
                             colsample_bytree=0.9, scale_pos_weight=pos_weight, eval_metric="logloss",
                             n_jobs=-1, random_state=SEED),
    }


def build_pipeline(clf) -> Pipeline:
    return Pipeline([("prep", make_preprocessor()), ("clf", clf)])


def to_frame(profile: dict) -> pd.DataFrame:
    """One-row DataFrame in contract feature order (input already validated)."""
    return pd.DataFrame([[profile[f] for f in FEATURES]], columns=FEATURES)


def explainer_kind(clf) -> str:
    if hasattr(clf, "get_booster") or hasattr(clf, "feature_importances_"):
        return "tree"
    return "linear"


def shap_contributions(clf, prep: ColumnTransformer, row: pd.DataFrame, background=None, top: int = 8) -> dict:
    """SHAP contributions of the positive class for one row (log-odds scale)."""
    import shap

    Xt = prep.transform(row)
    names = prep.get_feature_names_out()
    if explainer_kind(clf) == "tree":
        exp = shap.TreeExplainer(clf)
    else:
        if background is None:
            background = Xt  # single-row background (degenerate but functional)
        exp = shap.LinearExplainer(clf, np.asarray(background))
    sv = exp.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1] if sv.shape[-1] == 2 else sv[..., 0]
    sv = np.asarray(sv).reshape(-1)
    order = np.argsort(-np.abs(sv))[:top]
    def _native(v):
        return v.item() if isinstance(v, np.generic) else v
    raw = {f: _native(v) for f, v in zip(FEATURES, row.iloc[0].tolist())}
    # map expanded one-hot names back to their source feature for display
    contribs = []
    for i in order:
        name = str(names[i])
        base = name.split("__", 1)[1] if "__" in name else name
        src = base.split("_")[0]
        for f in FEATURES:
            if name.startswith(f"num__{f}") or name.startswith(f"cat__{f}_"):
                src = f
                break
        contribs.append({"feature": src, "detail": name, "value": raw.get(src, None), "contribution": float(sv[i])})
    base_value = float(np.asarray(exp.expected_value if not isinstance(exp.expected_value, list)
                                  else exp.expected_value[1]).reshape(-1)[0]) if hasattr(exp, "expected_value") else None
    return {"method": "shap", "output": "log-odds", "base_value": base_value, "contributions": contribs}
