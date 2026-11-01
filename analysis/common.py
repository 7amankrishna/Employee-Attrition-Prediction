"""Shared helpers for the attrition research pipeline.

Datasets (see data/README.md for provenance):
  * ibm        - synthetic IBM HR sample bundled with the frontend (public/data/ibm-hr.csv)
  * hr         - real anonymized company HRIS export (data/hr-comma-sep.csv), target `left`
  * av         - real Analytics Vidhya job-change candidates (data/av-job-change-train.csv), target `target`
  * academic   - real published faculty census 2011->2017 (data/academic-faculty.csv + department rates)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PUBLIC_DATA = ROOT / "public" / "data"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RESULTS = REPORTS / "results"

for d in (FIGURES, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

RNG = 20261101


def save_json(name: str, obj: dict) -> Path:
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, default=_jsonable), encoding="utf-8")
    return path


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ---------------------------------------------------------------- loaders
def load_ibm() -> pd.DataFrame:
    df = pd.read_csv(PUBLIC_DATA / "ibm-hr.csv")
    df["left"] = (df["Attrition"] == "Yes").astype(int)
    return df


def load_hr() -> pd.DataFrame:
    df = pd.read_csv(DATA / "hr-comma-sep.csv")
    df["left"] = df["left"].astype(int)
    df = df.rename(columns={"Department": "department", "salary": "salary_band"})
    return df


def load_av() -> pd.DataFrame:
    df = pd.read_csv(DATA / "av-job-change-train.csv")
    df["left"] = df["target"].astype(int)
    # `experience` mixes numbers with '<1' / '>20'; keep a numeric version for tests
    exp = pd.to_numeric(df["experience"], errors="coerce")
    df["experience_num"] = exp.fillna(df["experience"].map({"<1": 0.5, ">20": 20.5}).fillna(exp))
    return df


def load_academic() -> tuple[pd.DataFrame, pd.DataFrame]:
    fac = pd.read_csv(DATA / "academic-faculty.csv", index_col=0)
    fac["left"] = fac["title_2017"].isna().astype(int)  # no title in 2017 => left the field
    dep = pd.read_csv(DATA / "academic-department-attrition.csv", index_col=0)
    return fac, dep


# ---------------------------------------------------------------- statistics
def chi2_test(df: pd.DataFrame, feature: str, target: str = "left") -> dict:
    """Chi-square test of independence + Cramer's V effect size."""
    sub = df[[feature, target]].dropna()
    if sub[feature].nunique() < 2 or sub[target].nunique() < 2:
        return {"feature": feature, "n": int(len(sub)), "skipped": "degenerate"}
    table = pd.crosstab(sub[feature], sub[target])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))) if min(table.shape) > 1 else 0.0
    rates = (table[1] / table.sum(axis=1)).sort_values(ascending=False) if 1 in table else pd.Series(dtype=float)
    return {
        "feature": feature,
        "n": int(n),
        "chi2": float(chi2),
        "dof": int(dof),
        "p": float(p),
        "cramers_v": cramers_v,
        "rates_by_level": {str(k): float(v) for k, v in rates.items()},
        "counts": {str(k): {str(c): int(v) for c, v in row.items()} for k, row in table.to_dict().items()},
    }


def odds_ratio_ci(df: pd.DataFrame, feature: str, level, target: str = "left") -> dict:
    """Odds ratio of target=1 for `level` vs all other levels, with 95% CI."""
    sub = df[[feature, target]].dropna()
    mask = sub[feature].astype(str) == str(level)
    a = int((mask & (sub[target] == 1)).sum())
    b = int((mask & (sub[target] == 0)).sum())
    c = int((~mask & (sub[target] == 1)).sum())
    d = int((~mask & (sub[target] == 0)).sum())
    if min(a, b, c, d) == 0:
        return {"level": str(level), "a": a, "b": b, "c": c, "d": d, "or": None, "lo": None, "hi": None}
    orr = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo, hi = orr * np.exp(-1.96 * se), orr * np.exp(1.96 * se)
    return {"level": str(level), "a": a, "b": b, "c": c, "d": d,
            "or": float(orr), "lo": float(lo), "hi": float(hi)}


def numeric_test(df: pd.DataFrame, feature: str, target: str = "left") -> dict:
    """Mann-Whitney U + rank-biserial correlation + Cohen's d."""
    sub = df[[feature, target]].dropna()
    sub = sub[pd.to_numeric(sub[feature], errors="coerce").notna()]
    x = pd.to_numeric(sub.loc[sub[target] == 1, feature]).to_numpy()
    y = pd.to_numeric(sub.loc[sub[target] == 0, feature]).to_numpy()
    if len(x) < 5 or len(y) < 5:
        return {"feature": feature, "skipped": "too few rows"}
    u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    rbc = 1 - 2 * u / (len(x) * len(y))  # rank-biserial correlation
    pooled = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    d = (x.mean() - y.mean()) / pooled if pooled > 0 else 0.0
    # Welch t-test p-value for the report
    t, pt = stats.ttest_ind(x, y, equal_var=False)
    return {
        "feature": feature, "n": int(len(sub)),
        "mean_left": float(x.mean()), "sd_left": float(x.std(ddof=1)),
        "mean_stay": float(y.mean()), "sd_stay": float(y.std(ddof=1)),
        "median_left": float(np.median(x)), "median_stay": float(np.median(y)),
        "mannwhitney_p": float(p), "welch_t_p": float(pt), "t": float(t),
        "rank_biserial": float(rbc), "cohens_d": float(d),
    }


def banded_rate(df: pd.DataFrame, feature: str, bands: list[tuple[float, float, str]], target: str = "left") -> dict:
    """Attrition rate within numeric bands. Bands are [lo, hi) except the last, which is inclusive."""
    x = pd.to_numeric(df[feature], errors="coerce")
    out = []
    for i, (lo, hi, label) in enumerate(bands):
        last = i == len(bands) - 1
        mask = (x >= lo) & (x <= hi if last else x < hi)
        n = int(mask.sum())
        left = int(df.loc[mask, target].sum()) if n else 0
        out.append({"band": label, "n": n, "left": left, "rate": (left / n) if n else 0.0})
    return {"feature": feature, "bands": out}


def rate_by(df: pd.DataFrame, feature: str, target: str = "left", min_n: int = 30) -> list[dict]:
    g = df.groupby(df[feature].astype(str))[target].agg(["count", "sum"])
    g["rate"] = g["sum"] / g["count"]
    g = g[g["count"] >= min_n].sort_values("rate", ascending=False)
    return [{"level": str(idx), "n": int(r["count"]), "left": int(r["sum"]), "rate": float(r["rate"])}
            for idx, r in g.iterrows()]


def overall(df: pd.DataFrame, target: str = "left") -> dict:
    return {"n": int(len(df)), "left": int(df[target].sum()),
            "stayed": int((1 - df[target]).sum()), "rate": float(df[target].mean())}


def missing(df: pd.DataFrame) -> dict:
    miss = df.isna().sum()
    return {c: int(v) for c, v in miss[miss > 0].items()}
