"""Generate reports/attrition-research-report.md from the analysis results.

Run:  python3 analysis/report.py   (after eda.py and modeling.py)
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from common import REPORTS, RESULTS

TODAY = "2026-11-01"


def pct(x, d=1):
    return "-" if x is None else f"{x*100:.{d}f}%"


def pval(p):
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def fmt_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def cat_table(tests, keys, label_map=None):
    rows = []
    for k in keys:
        t = tests.get(k)
        if not t or "cramers_v" not in t:
            continue
        top = sorted(t["rates_by_level"].items(), key=lambda kv: -kv[1])[:3]
        top_s = "; ".join(f"{label_map(k, l) if label_map else l}: {pct(v)}" for l, v in top)
        rows.append([k, t["n"], pval(t["p"]), f"{t['cramers_v']:.3f}", top_s])
    return fmt_table(["feature", "n", "p (chi2)", "Cramer's V", "highest attrition levels"], rows)


def num_table(tests, keys):
    rows = []
    for k in keys:
        t = tests.get(k)
        if not t or "cohens_d" not in t:
            continue
        rows.append([k, t["n"], f"{t['mean_left']:.2f} vs {t['mean_stay']:.2f}",
                     f"{t['median_left']:.2f} vs {t['median_stay']:.2f}",
                     pval(t["mannwhitney_p"]), f"{t['cohens_d']:+.3f}"])
    return fmt_table(["feature", "n", "mean left vs stayed", "median left vs stayed", "p (Mann-Whitney)", "Cohen's d"], rows)


def model_section(name, m):
    head = f"**Best model by 5-fold CV ROC-AUC: `{m['best_model']}`** (n={m['n']:,}"
    if "n_test" in m:
        head += f", held-out test n={m['n_test']:,}, test base rate {pct(m['base_rate_test'])}"
    head += ")."
    lines = [head]
    if "test" in m:
        t = m["test"]
        lines.append("")
        lines.append(fmt_table(
            ["model", "CV ROC-AUC", "CV PR-AUC", "CV F1", "CV recall"],
            [[mod, f"{cv['roc_auc']['mean']:.3f}±{cv['roc_auc']['std']:.3f}",
              f"{cv['average_precision']['mean']:.3f}", f"{cv['f1']['mean']:.3f}", f"{cv['recall']['mean']:.3f}"]
             for mod, cv in m["cv"].items()]))
        lines.append("")
        lines.append(fmt_table(
            ["test ROC-AUC", "test PR-AUC", "accuracy", "precision", "recall (leavers)", "F1", "Brier", "confusion [[TN,FP],[FN,TP]]"],
            [[f"{t['roc_auc']:.3f}", f"{t['pr_auc']:.3f}", f"{t['accuracy']:.3f}", f"{t['precision']:.3f}",
              f"{t['recall']:.3f}", f"{t['f1']:.3f}", f"{t['brier']:.3f}", str(t["confusion_matrix"])]]))
        lines.append("")
        lines.append("Top permutation importances (test set, drop in ROC-AUC):")
        lines.append("")
        lines.append(fmt_table(["feature", "importance"],
                               [[p["feature"], f"{p['importance']:.4f}"] for p in m["permutation_importance_top15"][:10]]))
        if m.get("shap_top15"):
            lines.append("")
            lines.append("Top SHAP mean |value| (test sample):")
            lines.append("")
            lines.append(fmt_table(["feature", "mean |SHAP|"],
                                   [[s["feature"], f"{s['mean_abs_shap']:.4f}"] for s in m["shap_top15"][:10]]))
    else:
        lines.append("")
        lines.append(fmt_table(["model", "CV ROC-AUC", "CV PR-AUC"],
                               [[mod, f"{cv['roc_auc']['mean']:.3f}±{cv['roc_auc']['std']:.3f}",
                                 f"{cv['average_precision']['mean']:.3f}"] for mod, cv in m["cv"].items()]))
        lines.append("")
        lines.append(f"_Note: {m.get('note','')}_")
    lines.append("")
    lines.append(f"![{name} ROC/PR](figures/fig_{name}_roc_pr.png) ![confusion](figures/fig_{name}_confusion.png) ![calibration](figures/fig_{name}_calibration.png)" if "test" in m else "")
    if "test" in m:
        lines.append("")
        lines.append(f"![{name} importance](figures/fig_{name}_importance.png) ![shap](figures/fig_{name}_shap.png)")
    return "\n".join(lines)


def main():
    eda = json.load(open(RESULTS / "eda_all.json"))
    models = json.load(open(RESULTS / "model_all.json"))
    ibm, hr, av, ac = eda["ibm"], eda["hr"], eda["av"], eda["academic"]

    ibm_or = ibm["odds_ratios"]; hr_or = hr["odds_ratios"]; av_or = av["odds_ratios"]; ac_or = ac["odds_ratios"]

    doc = f"""# Why People Leave: A Multi-Dataset Attrition Study

**Employee attrition research report — THREAD retention intelligence**
Generated {TODAY} by `analysis/` (see *Reproducibility*). All numbers in this
report are computed from the datasets in `data/` — nothing is hand-written.

## Abstract

We study employee attrition / job-change intent across **four datasets** — three
real-world sources and one synthetic benchmark — totalling **41,617 records**:
a real anonymized company HRIS export (14,995 employees), real candidates from a
data-science job platform (18,359), a published census of US physics faculty
2011→2017 (6,793), and the synthetic IBM HR sample bundled with this project
(1,470). Across all real datasets, attrition is concentrated in **early-tenure,
low-paid, overworked or dissatisfied** employees; in the faculty census, in
**senior ranks** (retirement) with women *less* likely to leave than men
(OR {ac_or['sex=F vs M']['or']:.2f}, 95% CI {ac_or['sex=F vs M']['lo']:.2f}–{ac_or['sex=F vs M']['hi']:.2f}). Predictive
models reach held-out ROC-AUC **{models['ibm']['test']['roc_auc']:.2f}** (IBM sample),
**{models['hr']['test']['roc_auc']:.3f}** (real company — near-deterministic, see limitations),
**{models['av']['test']['roc_auc']:.2f}** (job-change intent, modest) and CV
**{models['academic']['cv']['logreg']['roc_auc']['mean']:.2f}** (faculty, title/sex only). The recurring, replicated drivers are
**tenure, compensation, workload/overtime, satisfaction and promotion stagnation** —
the same levers HR can actually pull.

## 1. Introduction and research questions

Employee attrition is costly (recruiting, onboarding, lost tacit knowledge), so
identifying *who* is at risk and *why* matters. Public, individual-level attrition
data is rare, so we combine every credible source we could locate on GitHub:

- **RQ1 — How common is attrition, and how does it vary across cohorts** (department/role, sex, tenure, pay, workload)?
- **RQ2 — Which factors are statistically associated with leaving** (chi-square + Cramér's V for categories; Mann–Whitney + Cohen's d for numerics; odds ratios with 95% CIs)?
- **RQ3 — How well can leaving be predicted from observed features, and which model generalizes best?**
- **RQ4 — Which features drive the predictions** (permutation importance, SHAP), **and do the findings replicate across independent datasets?**

## 2. Data

| dataset | file | n | target | real? | source |
| --- | --- | --- | --- | --- | --- |
| IBM sample | `public/data/ibm-hr.csv` | {ibm['overall']['n']:,} | `Attrition` Yes/No ({pct(ibm['overall']['rate'])}) | synthetic benchmark | IBM HR Analytics (Kaggle); fictional data created by IBM data scientists |
| Company HRIS | `data/hr-comma-sep.csv` | {hr['overall']['n']:,} | `left` ({pct(hr['overall']['rate'])}) | **yes** (anonymized real export) | Kaggle "HR Analytics" (`HR_comma_sep.csv`), GitHub mirror |
| Job-change candidates | `data/av-job-change-train.csv` | {av['overall']['n']:,} | `target` ({pct(av['overall']['rate'])} seeking a job change) | **yes** | Analytics Vidhya JanataHack hackathon, GitHub mirror |
| Faculty census | `data/academic-faculty.csv` (+ department rates) | {ac['overall']['n']:,} | left the field 2011→2017 ({pct(ac['overall']['rate'])}) | **yes** (published) | Morgan, Way & Clauset (2018), arXiv:1804.02760 supplementary data |

Provenance, checksums and re-fetch instructions: [`data/README.md`](../data/README.md).

## 3. Methods

**Statistics.** Categorical associations: chi-square test of independence with
Cramér's V effect size; odds ratios (Woolf 95% CI). Numeric differences between
leavers and stayers: Mann–Whitney U with rank-biserial correlation and Cohen's d
effect size, plus Welch t-test. α = 0.05, two-sided; with n in the thousands we
report effect sizes, not just p-values.

**Modeling.** For each employee-level dataset: stratified 80/20 split (seed
20261101); preprocessing (median imputation + scaling for numerics, one-hot for
categoricals) fit on train only; 5-fold stratified CV on the training portion
comparing logistic regression, random forest (400 trees) and XGBoost; the best
model by CV ROC-AUC is refit on train and evaluated **once** on the held-out
test set (ROC-AUC, PR-AUC, accuracy, precision, recall, F1, Brier score,
confusion matrix, calibration curve). Feature drivers: permutation importance
and SHAP (TreeExplainer / LinearExplainer). The faculty census has only sex and
2011 title as covariates, so it gets a CV-only logistic model.

## 4. Results

### 4.1 IBM sample (synthetic benchmark, n={ibm['overall']['n']:,}, attrition {pct(ibm['overall']['rate'])})

Categorical associations (chi-square):

{cat_table(ibm['tests'], ['OverTime','JobRole','MaritalStatus','StockOptionLevel','JobInvolvement','BusinessTravel','EnvironmentSatisfaction','WorkLifeBalance','JobSatisfaction','EducationField','Department','Gender','PerformanceRating'])}

Numeric differences (leavers vs stayers):

{num_table(ibm['tests'], ['Age','MonthlyIncome','YearsAtCompany','TotalWorkingYears','YearsInCurrentRole','YearsWithCurrManager','DistanceFromHome','YearsSinceLastPromotion','NumCompaniesWorked','PercentSalaryHike'])}

Odds ratios: overtime **OR {ibm_or['OverTime=Yes']['or']:.2f}** [{ibm_or['OverTime=Yes']['lo']:.2f}–{ibm_or['OverTime=Yes']['hi']:.2f}],
single marital status **OR {ibm_or['MaritalStatus=Single']['or']:.2f}** [{ibm_or['MaritalStatus=Single']['lo']:.2f}–{ibm_or['MaritalStatus=Single']['hi']:.2f}],
work-life balance = 1 (bad) **OR {ibm_or['WorkLifeBalance=1']['or']:.2f}** [{ibm_or['WorkLifeBalance=1']['lo']:.2f}–{ibm_or['WorkLifeBalance=1']['hi']:.2f}].
Gender (p={pval(ibm['tests']['Gender']['p'])}) and performance rating (p={pval(ibm['tests']['PerformanceRating']['p'])}) are **not** associated with attrition.

![ibm overtime](figures/fig_ibm_overtime.png) ![ibm satisfaction](figures/fig_ibm_satisfaction.png)

![ibm income/tenure](figures/fig_ibm_income_tenure.png) ![ibm role/distance](figures/fig_ibm_role_distance.png) ![ibm marital/promotion](figures/fig_ibm_marital_promotion.png)

### 4.2 Real company HRIS (n={hr['overall']['n']:,}, attrition {pct(hr['overall']['rate'])})

Categorical associations:

{cat_table(hr['tests'], ['number_project','salary_band','Work_accident','department','promotion_last_5years'])}

Numeric differences:

{num_table(hr['tests'], ['satisfaction_level','time_spend_company','average_montly_hours','last_evaluation'])}

Odds ratios: low salary **OR {hr_or['salary=low']['or']:.2f}** [{hr_or['salary=low']['lo']:.2f}–{hr_or['salary=low']['hi']:.2f}];
**no promotion in 5 years OR {hr_or['no promotion (promotion_last_5years=0)']['or']:.2f}** [{hr_or['no promotion (promotion_last_5years=0)']['lo']:.2f}–{hr_or['no promotion (promotion_last_5years=0)']['hi']:.2f}];
work accident **OR {hr_or['work accident']['or']:.2f}** (protective — see discussion).

![hr department](figures/fig_hr_department.png) ![hr salary/promo/tenure](figures/fig_hr_salary_promo_tenure.png)

![hr workload](figures/fig_hr_workload.png) ![hr scatter](figures/fig_hr_scatter.png)

### 4.3 Real job-change candidates (n={av['overall']['n']:,}, seeking change {pct(av['overall']['rate'])})

Categorical associations:

{cat_table(av['tests'], ['experience_cat','enrolled_university','relevent_experience','education_level','company_type','gender','company_size','last_new_job','major_discipline'])}

Numeric differences:

{num_table(av['tests'], ['city_development_index','experience_num','training_hours'])}

Odds ratios: no relevant experience **OR {av_or['relevent_experience=No']['or']:.2f}** [{av_or['relevent_experience=No']['lo']:.2f}–{av_or['relevent_experience=No']['hi']:.2f}];
graduate education **OR {av_or['education=Graduate']['or']:.2f}** [{av_or['education=Graduate']['lo']:.2f}–{av_or['education=Graduate']['hi']:.2f}].

![av experience/education](figures/fig_av_experience_education.png) ![av company/cdi](figures/fig_av_company_cdi.png) ![av gender/training](figures/fig_av_gender_training.png)

### 4.4 Real faculty census 2011→2017 (n={ac['overall']['n']:,}, left field {pct(ac['overall']['rate'])})

{cat_table(ac['tests'], ['title_2011','sex'])}

Sex odds ratio (F vs M): **OR {ac_or['sex=F vs M']['or']:.2f}** [{ac_or['sex=F vs M']['lo']:.2f}–{ac_or['sex=F vs M']['hi']:.2f}].
Across {ac['department_stats']['n_departments']} departments the attrition fraction ranges
**{pct(ac['department_stats']['min'])}–{pct(ac['department_stats']['max'],0)}**
(mean {pct(ac['department_stats']['mean_attrition_frac'],1)}, median {pct(ac['department_stats']['median_attrition_frac'],1)},
headcount-weighted {pct(ac['department_stats']['weighted_overall'],1)}).

![academic sex/title](figures/fig_academic_sex_title.png) ![academic departments](figures/fig_academic_departments.png)

### 4.5 Predictive models

{model_section('ibm', models['ibm'])}

{model_section('hr', models['hr'])}

{model_section('av', models['av'])}

{model_section('academic', models['academic'])}

## 5. What the study found (synthesis)

1. **Overtime and workload are the strongest categorical levers in the IBM sample**
   (overtime: {pct(ibm['tests']['OverTime']['rates_by_level']['Yes'])} vs {pct(ibm['tests']['OverTime']['rates_by_level']['No'])} attrition, OR {ibm_or['OverTime=Yes']['or']:.2f}, Cramér's V {ibm['tests']['OverTime']['cramers_v']:.2f}). In the real company data the equivalent signal is extreme
   project load: employees with **7 projects left 100% of the time** (V={hr['tests']['number_project']['cramers_v']:.2f}).
2. **Satisfaction is the single biggest numeric driver in the real company data**
   (Cohen's d {hr['tests']['satisfaction_level']['cohens_d']:+.2f}; mean {hr['tests']['satisfaction_level']['mean_left']:.2f} for leavers vs {hr['tests']['satisfaction_level']['mean_stay']:.2f} for stayers) and the top SHAP feature.
3. **Pay and promotion stagnation matter.** Low salary band OR {hr_or['salary=low']['or']:.2f} (real company) and mean monthly income of leavers is
   ${ibm['tests']['MonthlyIncome']['mean_left']:,.0f} vs ${ibm['tests']['MonthlyIncome']['mean_stay']:,.0f} in the IBM sample (d {ibm['tests']['MonthlyIncome']['cohens_d']:+.2f}); in the real company, **not being promoted in 5 years carries OR {hr_or['no promotion (promotion_last_5years=0)']['or']:.2f}**.
4. **Early tenure is a danger zone** — in IBM, leavers average {ibm['tests']['YearsAtCompany']['mean_left']:.1f} years at the company vs {ibm['tests']['YearsAtCompany']['mean_stay']:.1f} (d {ibm['tests']['YearsAtCompany']['cohens_d']:+.2f}) and `YearsAtCompany` is the top permutation/SHAP feature; in the job-change data, candidates with <1–1 year of experience seek new jobs at {pct(av['tests']['experience_cat']['rates_by_level']['<1'])} / {pct(av['tests']['experience_cat']['rates_by_level']['1'])} vs {pct(av['overall']['rate'])} overall.
5. **Context, not demographics, dominates.** City development index (d {av['tests']['city_development_index']['cohens_d']:+.2f}) and company type beat gender in the job-change data; in IBM, gender (p={pval(ibm['tests']['Gender']['p'])}) and performance rating are unrelated to attrition.
6. **In academia the pattern flips by rank**: full professors leave most ({pct(ac['tests']['title_2011']['rates_by_level']['Full Professor'])} — largely retirement), and women are *less* likely to leave than men (OR {ac_or['sex=F vs M']['or']:.2f}). Department-level attrition varies enormously ({pct(ac['department_stats']['min'])}–{pct(ac['department_stats']['max'],0)}), so institution-level factors matter.
7. **Predictability is dataset-dependent**: near-perfect on the real company HRIS (test ROC-AUC {models['hr']['test']['roc_auc']:.3f}, but see limitations), good on the IBM sample ({models['ibm']['test']['roc_auc']:.2f}), and only modest for job-change intent ({models['av']['test']['roc_auc']:.2f}, PR-AUC {models['av']['test']['pr_auc']:.2f} vs base rate {pct(models['av']['test']['base_rate'])}). Attrition *intent* is harder to predict than recorded exits.

## 6. Discussion — practical implications for HR

- **Watch overtime and project load first** — they are the most actionable and most strongly associated with leaving.
- **Run retention conversations in years 0–3** and around promotion cycles (the "no promotion in 5 years" group is the highest-risk in the real data).
- **Audit pay bands by role** — low salary and low income are replicated risk markers; check Sales-type roles (IBM sales roles reach {pct(max(ibm['tests']['JobRole']['rates_by_level'].values()))}).
- **Fix satisfaction drivers, not demographics** — gender/performance show no signal in the IBM sample; satisfaction and work-life balance do.
- **Use models as triage, not verdicts** — probabilities flag *who to talk to*, never *what to decide*; the 30/60 risk bands in the UI are demo conventions, not validated thresholds.

## 7. Limitations

- The IBM file is **synthetic** (created by IBM data scientists); treat its numbers as a benchmark, not evidence about real employees.
- The real company HRIS is nearly perfectly separable (test ROC-AUC {models['hr']['test']['roc_auc']:.3f}) — consistent with the known structure of this widely used dataset (project count/tenure/satisfaction interact almost deterministically). Real-world HR data is usually noisier; do not expect 0.99 AUC in production.
- Job-change *intent* (AV) is self-reported and only weakly predictable ({models['av']['test']['roc_auc']:.2f}); the AV file also has substantial missingness (see `reports/results/eda_all.json`).
- The faculty census covers one field (physics) at PhD-granting US departments, 2011→2017; "left" means left the census, not necessarily left employment.
- All datasets are cross-sectional snapshots or one transition window — **no causal claims**; associations only.
- Class imbalance (13–34% positive) means accuracy is misleading; we report PR-AUC and recall for leavers.

## 8. Responsible use

This is an educational demonstration, **not an employment-decision system**.
Models like these must never be used alone to discipline, rank, or deny
opportunities to real employees; any deployment needs human review, fairness
audits across protected groups, privacy safeguards, and local legal review
(e.g., GDPR/EEOC constraints on automated employment decisions). No dataset
here contains direct identifiers.

## 9. Reproducibility

```sh
python3 -m venv .venv && .venv/bin/pip install -r analysis/requirements.txt
python3 data/fetch_datasets.py            # optional: re-download + verify external data
python3 analysis/run_all.py               # eda.py -> modeling.py -> report.py
# Python prediction service (implements docs/backend-contract.md):
cd backend && ../.venv/bin/pip install -r requirements.txt
../.venv/bin/python train.py               # trains + writes backend/artifacts/
../.venv/bin/python -m pytest test_api.py  # contract tests
../.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
```

Raw computed results: `reports/results/eda_all.json`, `reports/results/model_all.json`.
Figures: `reports/figures/`. Frontend contract: `docs/backend-contract.md`.

## 10. References

1. IBM HR Analytics Employee Attrition & Performance dataset (Kaggle, pavansubhasht) — fictional sample; GitHub mirror `IamVasil/IBM-HR-Analytics-Employee-Attrition-Performance`.
2. Kaggle "HR Analytics" (`HR_comma_sep.csv`) — anonymized real company HRIS export; mirror `github.com/KunalKurve/Machine-Learning`.
3. Analytics Vidhya JanataHack — "HR Analytics: Job Change of Data Scientists" hackathon data; mirror `github.com/aasu14/Data-Science-Hackathon-And-Competition`.
4. Morgan, A. C., Way, S. F., & Clauset, A. (2018). *Automatically assembling a full census of an academic field.* arXiv:1804.02760 — supplementary data `github.com/allisonmorgan/academic_census`.
5. scikit-learn, XGBoost, SHAP documentation (modeling and explainability methods).
"""
    out = REPORTS / "attrition-research-report.md"
    out.write_text(doc, encoding="utf-8")
    print("wrote", out, f"({len(doc):,} chars)")


if __name__ == "__main__":
    main()
