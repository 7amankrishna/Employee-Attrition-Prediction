"""Exploratory and statistical analysis of the four attrition datasets.

Produces:
  reports/results/eda_<dataset>.json   - all computed statistics / tests
  reports/figures/fig_*.png             - charts used in the report

Run:  python3 analysis/eda.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (FIGURES, banded_rate, chi2_test, load_academic, load_av,
                    load_hr, load_ibm, missing, numeric_test, odds_ratio_ci,
                    overall, rate_by, save_json)

# palette aligned with the THREAD frontend theme
IVORY, AUBERGINE, CORAL, SAGE, INK = "#faf6ef", "#5b2333", "#e0655a", "#7a9b76", "#2b2b2b"
plt.rcParams.update({
    "figure.facecolor": IVORY, "axes.facecolor": IVORY, "savefig.facecolor": IVORY,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _bar(ax, labels, rates, title, color=CORAL, overall_rate=None, rot=0):
    rates = [0.0 if r is None else r for r in rates]
    x = np.arange(len(labels))
    ax.bar(x, [r * 100 for r in rates], color=color, edgecolor=INK, linewidth=0.4)
    if overall_rate is not None:
        ax.axhline(overall_rate * 100, color=AUBERGINE, linestyle="--", linewidth=1,
                   label=f"overall {overall_rate*100:.1f}%")
        ax.legend(frameon=False, fontsize=8)
    for xi, r in zip(x, rates):
        ax.text(xi, r * 100 + 0.6, f"{r*100:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center", fontsize=8)
    ax.set_ylabel("attrition rate (%)")
    ax.set_title(title, fontsize=11, color=AUBERGINE, loc="left")
    ax.set_ylim(0, max(rates) * 100 * 1.25 + 1)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print("  figure:", name)


# ------------------------------------------------------------------ IBM
def eda_ibm() -> dict:
    print("[eda] ibm")
    df = load_ibm()
    out = {"dataset": "ibm", "description": "IBM HR Analytics sample (synthetic), 1,470 employees",
           "overall": overall(df), "missing": missing(df), "tests": {}}

    cat_features = ["OverTime", "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction",
                    "JobInvolvement", "MaritalStatus", "BusinessTravel", "Department", "Gender",
                    "EducationField", "JobRole", "StockOptionLevel", "PerformanceRating"]
    for f in cat_features:
        out["tests"][f] = chi2_test(df, f)
    num_features = ["Age", "MonthlyIncome", "DistanceFromHome", "YearsAtCompany", "TotalWorkingYears",
                    "YearsSinceLastPromotion", "YearsInCurrentRole", "YearsWithCurrManager",
                    "NumCompaniesWorked", "PercentSalaryHike", "TrainingTimesLastYear",
                    "DailyRate", "HourlyRate", "MonthlyRate", "PercentSalaryHike"]
    for f in num_features:
        out["tests"][f] = numeric_test(df, f)

    out["bands"] = {
        "tenure": banded_rate(df, "YearsAtCompany", [(0, 1, "0-1"), (1, 3, "2-3"), (3, 5, "4-5"),
                                                     (5, 10, "6-10"), (10, 20, "11-20"), (20, 41, "21+")]),
        "distance": banded_rate(df, "DistanceFromHome", [(1, 5, "1-5"), (5, 10, "6-10"), (10, 15, "11-15"),
                                                         (15, 20, "16-20"), (20, 30, "21-29")]),
        "promotion": banded_rate(df, "YearsSinceLastPromotion", [(0, 1, "0-1"), (1, 3, "2-3"),
                                                                  (3, 6, "4-6"), (6, 10, "7-10"), (10, 16, "11+")]),
        "age": banded_rate(df, "Age", [(18, 25, "18-25"), (25, 30, "26-30"), (30, 35, "31-35"),
                                       (35, 40, "36-40"), (40, 50, "41-50"), (50, 61, "51-60")]),
        "income": banded_rate(df, "MonthlyIncome", [(0, 3000, "<3k"), (3000, 5000, "3-5k"),
                                                    (5000, 8000, "5-8k"), (8000, 12000, "8-12k"), (12000, 20001, "12k+")]),
    }
    out["rates"] = {f: rate_by(df, f) for f in ["JobRole", "Department", "MaritalStatus", "OverTime", "BusinessTravel"]}
    out["odds_ratios"] = {
        "OverTime=Yes": odds_ratio_ci(df, "OverTime", "Yes"),
        "MaritalStatus=Single": odds_ratio_ci(df, "MaritalStatus", "Single"),
        "JobSatisfaction=1": odds_ratio_ci(df, "JobSatisfaction", 1),
        "WorkLifeBalance=1": odds_ratio_ci(df, "WorkLifeBalance", 1),
    }

    ov = out["overall"]["rate"]
    # figures
    r = rate_by(df, "OverTime", min_n=1)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    _bar(ax, [x["level"] for x in r], [x["rate"] for x in r], "IBM: attrition by overtime", overall_rate=ov)
    _save(fig, "fig_ibm_overtime.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, f in zip(axes, ["JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction"]):
        r = rate_by(df, f, min_n=1)
        r = sorted(r, key=lambda x: int(x["level"]))
        _bar(ax, [x["level"] for x in r], [x["rate"] for x in r], f.replace("Satisfaction", " sat."), overall_rate=ov)
    _save(fig, "fig_ibm_satisfaction.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    left = df[df.left == 1]["MonthlyIncome"]; stay = df[df.left == 0]["MonthlyIncome"]
    bins = np.arange(1000, 20000, 1000)
    axes[0].hist(stay, bins=bins, alpha=0.75, label="stayed", color=SAGE, edgecolor=INK, linewidth=0.3)
    axes[0].hist(left, bins=bins, alpha=0.75, label="left", color=CORAL, edgecolor=INK, linewidth=0.3)
    axes[0].set_title("IBM: monthly income distribution", loc="left", color=AUBERGINE)
    axes[0].set_xlabel("MonthlyIncome"); axes[0].legend(frameon=False, fontsize=8)
    r = out["bands"]["tenure"]["bands"]
    _bar(axes[1], [b["band"] for b in r], [b["rate"] for b in r], "IBM: attrition by tenure (years at company)", overall_rate=ov, rot=30)
    _save(fig, "fig_ibm_income_tenure.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    r = rate_by(df, "JobRole")
    _bar(axes[0], [x["level"].replace(" ", "\n") for x in r], [x["rate"] for x in r], "IBM: attrition by job role", overall_rate=ov, rot=0)
    axes[0].tick_params(axis="x", labelsize=6.5)
    r = out["bands"]["distance"]["bands"]
    _bar(axes[1], [b["band"] for b in r], [b["rate"] for b in r], "IBM: attrition by distance from home (km)", overall_rate=ov)
    _save(fig, "fig_ibm_role_distance.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    r = rate_by(df, "MaritalStatus", min_n=1)
    _bar(axes[0], [x["level"] for x in r], [x["rate"] for x in r], "IBM: attrition by marital status", overall_rate=ov)
    r = out["bands"]["promotion"]["bands"]
    _bar(axes[1], [b["band"] for b in r], [b["rate"] for b in r], "IBM: attrition by years since last promotion", overall_rate=ov, rot=30)
    _save(fig, "fig_ibm_marital_promotion.png")
    return out


# ------------------------------------------------------------------ HR (real company)
def eda_hr() -> dict:
    print("[eda] hr-comma-sep (real company HRIS)")
    df = load_hr()
    out = {"dataset": "hr", "description": "Real anonymized company HRIS export (HR_comma_sep), 14,995 employees",
           "overall": overall(df), "missing": missing(df), "tests": {}}
    for f in ["department", "salary_band", "promotion_last_5years", "Work_accident", "number_project"]:
        out["tests"][f] = chi2_test(df, f)
    for f in ["satisfaction_level", "last_evaluation", "average_montly_hours", "time_spend_company"]:
        out["tests"][f] = numeric_test(df, f)
    out["bands"] = {
        "satisfaction": banded_rate(df, "satisfaction_level", [(0, .2, "0-.2"), (.2, .4, ".2-.4"),
                                                               (.4, .6, ".4-.6"), (.6, .8, ".6-.8"), (.8, 1.01, ".8-1")]),
        "evaluation": banded_rate(df, "last_evaluation", [(0, .4, "0-.4"), (.4, .6, ".4-.6"), (.6, .75, ".6-.75"),
                                                           (.75, .9, ".75-.9"), (.9, 1.01, ".9-1")]),
        "hours": banded_rate(df, "average_montly_hours", [(0, 150, "<150"), (150, 200, "150-200"),
                                                          (200, 250, "200-250"), (250, 300, "250-300"), (300, 500, "300+")]),
        "tenure": banded_rate(df, "time_spend_company", [(0, 2, "0-2"), (2, 3, "3"), (3, 4, "4"),
                                                         (4, 5, "5"), (5, 7, "6-7"), (7, 11, "8-10")]),
    }
    out["rates"] = {f: rate_by(df, f) for f in ["department", "salary_band"]}
    out["odds_ratios"] = {
        "salary=low": odds_ratio_ci(df, "salary_band", "low"),
        "no promotion (promotion_last_5years=0)": odds_ratio_ci(df, "promotion_last_5years", 0),
        "work accident": odds_ratio_ci(df, "Work_accident", 1),
    }
    ov = out["overall"]["rate"]

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    r = rate_by(df, "department")
    _bar(ax, [x["level"] for x in r], [x["rate"] for x in r], "Real company: attrition by department", overall_rate=ov, rot=30)
    _save(fig, "fig_hr_department.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    r = rate_by(df, "salary_band", min_n=1)
    order = {"low": 0, "medium": 1, "high": 2}
    r = sorted(r, key=lambda x: order.get(x["level"], 9))
    _bar(axes[0], [x["level"] for x in r], [x["rate"] for x in r], "by salary band", overall_rate=ov)
    r = rate_by(df, "promotion_last_5years", min_n=1)
    _bar(axes[1], [("promoted" if x["level"] == "1" else "not promoted") for x in r], [x["rate"] for x in r],
         "by promotion in last 5y", overall_rate=ov)
    r = out["bands"]["tenure"]["bands"]
    _bar(axes[2], [b["band"] for b in r], [b["rate"] for b in r], "by tenure (years)", overall_rate=ov)
    _save(fig, "fig_hr_salary_promo_tenure.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    r = rate_by(df, "number_project", min_n=1)
    r = sorted(r, key=lambda x: int(x["level"]))
    _bar(axes[0], [x["level"] for x in r], [x["rate"] for x in r], "by number of projects", overall_rate=ov)
    r = out["bands"]["hours"]["bands"]
    _bar(axes[1], [b["band"] for b in r], [b["rate"] for b in r], "by avg monthly hours", overall_rate=ov, rot=20)
    r = out["bands"]["satisfaction"]["bands"]
    _bar(axes[2], [b["band"] for b in r], [b["rate"] for b in r], "by satisfaction level", overall_rate=ov)
    _save(fig, "fig_hr_workload.png")

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    sub = df.sample(min(4000, len(df)), random_state=1)
    stay = sub[sub.left == 0]; left = sub[sub.left == 1]
    ax.scatter(stay["satisfaction_level"], stay["last_evaluation"], s=6, alpha=0.35, color=SAGE, label="stayed", linewidths=0)
    ax.scatter(left["satisfaction_level"], left["last_evaluation"], s=6, alpha=0.5, color=CORAL, label="left", linewidths=0)
    ax.set_xlabel("satisfaction_level"); ax.set_ylabel("last_evaluation")
    ax.set_title("Real company: satisfaction vs last evaluation", loc="left", color=AUBERGINE)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig_hr_scatter.png")
    return out


# ------------------------------------------------------------------ AV job change (real)
def eda_av() -> dict:
    print("[eda] av job-change candidates (real)")
    df = load_av()
    out = {"dataset": "av", "description": "Real Analytics Vidhya data-science candidates (job-change intent), 18,359 rows",
           "overall": overall(df), "missing": missing(df), "tests": {}}
    for f in ["gender", "relevent_experience", "enrolled_university", "education_level",
              "major_discipline", "company_size", "company_type", "last_new_job"]:
        out["tests"][f] = chi2_test(df, f)
    for f in ["city_development_index", "training_hours", "experience_num"]:
        out["tests"][f] = numeric_test(df, f)
    exp_order = ["<1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", ">20"]
    df["experience_cat"] = pd.Categorical(df["experience"].astype(str), categories=exp_order, ordered=True)
    out["tests"]["experience_cat"] = chi2_test(df, "experience_cat")
    out["bands"] = {
        "cdi": banded_rate(df, "city_development_index", [(0, .6, "0-.6"), (.6, .7, ".6-.7"), (.7, .8, ".7-.8"),
                                                          (.8, .9, ".8-.9"), (.9, 1.01, ".9-1")]),
        "training": banded_rate(df, "training_hours", [(0, 20, "0-20"), (20, 40, "21-40"), (40, 60, "41-60"),
                                                        (60, 80, "61-80"), (80, 101, "81+")]),
    }
    out["rates"] = {f: rate_by(df, f) for f in ["education_level", "company_type", "relevent_experience", "experience_cat"]}
    out["odds_ratios"] = {
        "relevent_experience=No": odds_ratio_ci(df, "relevent_experience", "No relevent experience"),
        "education=Graduate": odds_ratio_ci(df, "education_level", "Graduate"),
        "last_new_job=never": odds_ratio_ci(df, "last_new_job", "never"),
        "gender=female": odds_ratio_ci(df, "gender", "f"),
    }
    ov = out["overall"]["rate"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    r = rate_by(df, "experience_cat", min_n=1)
    r = [x for x in r if x["level"] in exp_order]
    r = sorted(r, key=lambda x: exp_order.index(x["level"]))
    _bar(axes[0], [x["level"] for x in r], [x["rate"] for x in r], "Job-change intent by experience (years)", overall_rate=ov, rot=45)
    edu_order = ["No Graduate", "Graduate", "Masters", "High School", "Phd"]
    r = rate_by(df, "education_level", min_n=1)
    r = sorted(r, key=lambda x: edu_order.index(x["level"]) if x["level"] in edu_order else 99)
    _bar(axes[1], [x["level"] for x in r], [x["rate"] for x in r], "by education level", overall_rate=ov, rot=20)
    _save(fig, "fig_av_experience_education.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    r = rate_by(df, "company_type", min_n=1)
    _bar(axes[0], [x["level"] for x in r], [x["rate"] for x in r], "by company type", overall_rate=ov, rot=20)
    r = rate_by(df, "relevent_experience", min_n=1)
    _bar(axes[1], [x["level"] for x in r], [x["rate"] for x in r], "by relevant experience", overall_rate=ov, rot=15)
    r = out["bands"]["cdi"]["bands"]
    _bar(axes[2], [b["band"] for b in r], [b["rate"] for b in r], "by city development index", overall_rate=ov)
    _save(fig, "fig_av_company_cdi.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    g = rate_by(df, "gender", min_n=1)
    _bar(axes[0], [x["level"] for x in g], [x["rate"] for x in g], "by gender (m/f/other/blank)", overall_rate=ov)
    r = out["bands"]["training"]["bands"]
    _bar(axes[1], [b["band"] for b in r], [b["rate"] for b in r], "by training hours", overall_rate=ov)
    _save(fig, "fig_av_gender_training.png")
    return out


# ------------------------------------------------------------------ academic census (real, published)
def eda_academic() -> dict:
    print("[eda] academic census (Morgan et al. 2018, real)")
    fac, dep = load_academic()
    out = {"dataset": "academic",
           "description": "Published census of US physics faculty 2011->2017 (Morgan, Way & Clauset 2018)",
           "overall": overall(fac), "missing": missing(fac), "tests": {}}
    fac["title_2011"] = fac["title_2011"].fillna("(new in 2011)")
    for f in ["sex", "title_2011"]:
        out["tests"][f] = chi2_test(fac, f)
    out["rates"] = {f: rate_by(fac, f, min_n=10) for f in ["sex", "title_2011"]}
    out["odds_ratios"] = {
        "sex=F vs M": {**odds_ratio_ci(fac, "sex", "F"), "note": "F vs M reference"},
        "Assistant vs Associate+": None,
    }
    # proper OR: F vs M
    sub = fac[fac["sex"].isin(["M", "F"])]
    a = int(((sub.sex == "F") & (sub.left == 1)).sum()); b = int(((sub.sex == "F") & (sub.left == 0)).sum())
    c = int(((sub.sex == "M") & (sub.left == 1)).sum()); d = int(((sub.sex == "M") & (sub.left == 0)).sum())
    orr = (a * d) / (b * c); se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    out["odds_ratios"]["sex=F vs M"] = {"a": a, "b": b, "c": c, "d": d, "or": float(orr),
                                        "lo": float(orr * np.exp(-1.96 * se)), "hi": float(orr * np.exp(1.96 * se))}
    out["department_stats"] = {
        "n_departments": int(len(dep)),
        "mean_attrition_frac": float(dep["attrition_frac"].mean()),
        "median_attrition_frac": float(dep["attrition_frac"].median()),
        "min": float(dep["attrition_frac"].min()), "max": float(dep["attrition_frac"].max()),
        "weighted_overall": float(dep["count_not"].sum() / (dep["count_ret"].sum() + dep["count_not"].sum())),
        "top10": dep.nlargest(10, "attrition_frac")["place"].tolist(),
        "bottom10": dep.nsmallest(10, "attrition_frac")["place"].tolist(),
    }
    ov = out["overall"]["rate"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    r = rate_by(fac, "sex", min_n=10)
    _bar(axes[0], [x["level"] for x in r], [x["rate"] for x in r], "Faculty attrition 2011-2017 by sex", overall_rate=ov)
    r = rate_by(fac, "title_2011", min_n=10)
    _bar(axes[1], [x["level"] for x in r], [x["rate"] for x in r], "by 2011 title", overall_rate=ov, rot=20)
    _save(fig, "fig_academic_sex_title.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(dep["attrition_frac"], bins=30, color=SAGE, edgecolor=INK, linewidth=0.4)
    axes[0].axvline(dep["attrition_frac"].mean(), color=CORAL, linestyle="--",
                    label=f"mean {dep['attrition_frac'].mean():.2f}")
    axes[0].set_xlabel("department attrition fraction 2011-2017"); axes[0].set_ylabel("departments")
    axes[0].set_title("Distribution across 205 departments", loc="left", color=AUBERGINE)
    axes[0].legend(frameon=False, fontsize=8)
    top = dep.nlargest(10, "attrition_frac")[::-1]
    axes[1].barh([p[:38] for p in top["place"]], top["attrition_frac"] * 100, color=CORAL, edgecolor=INK, linewidth=0.3)
    axes[1].set_xlabel("attrition (%)"); axes[1].set_title("Highest-attrition departments", loc="left", color=AUBERGINE)
    axes[1].tick_params(axis="y", labelsize=7)
    _save(fig, "fig_academic_departments.png")
    return out


def main():
    results = {"ibm": eda_ibm(), "hr": eda_hr(), "av": eda_av(), "academic": eda_academic()}
    save_json("eda_all.json", results)
    print("saved reports/results/eda_all.json")


if __name__ == "__main__":
    main()
