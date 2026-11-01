# THREAD — Retention Intelligence

**Understand why people leave.**

A presentation-ready, responsive React frontend for the Employee Attrition Prediction university ML mini-project (Group 13). Built around warm ivory, aubergine, coral, sage, expressive serif typography and custom SVG thread illustrations.

## Run locally

Requires Node.js 22+ and npm.

```sh
npm ci
npm run dev
```

```sh
npm test
npm run build
npm run preview
```

## Deploy to Vercel

Import this GitHub repository into Vercel. Use the repository root, the **Vite** framework preset, build command `npm run build`, and output directory `dist`. `vercel.json` already declares these settings. No environment variables are required for the frontend demo. Navigation uses URL hashes, so direct links and refreshes work without rewrite rules.

## Five connected experiences

| Route | Experience |
| --- | --- |
| `/#overview` | Dataset summary and charts recalculated from the globally selected source: fictional IBM sample or separate anonymized HRIS |
| `/#predict` | Dataset-specific validated profile form and trained probability output; each result reports training and total data entry counts |
| `/#drivers` | Model contributions in log-odds for the browser logistic model, or SHAP from the optional API |
| `/#patterns` | Dataset-specific department/role, salary, tenure and overtime filters with recalculated rates and cohort CSV export |
| `/#model` | Actual training metadata, held-out metrics and confusion matrix for the selected dataset; backend API may supply the same contract |

A guided tour connects the five pages. Forms and the latest result persist across page navigation during the current session; employee inputs are not saved to browser storage.

## Data and scientific honesty

The included CSV contains 1,470 fictional IBM HR sample records, including 237 `Attrition=Yes` records and 1,233 `No` records. Descriptive charts are calculated from these records in the browser, including all cohort filters. The flowing thread paths are explicitly conceptual rather than employee trajectories.

- [Dataset listing](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- [CSV mirror used](https://github.com/IamVasil/IBM-HR-Analytics-Employee-Attrition-Performance/blob/main/WA_Fn-UseC_-HR-Employee-Attrition.csv)
- Local file: `public/data/ibm-hr.csv`

The header switch changes the source across every section. IBM uses the 1,470-row fictional benchmark; Real HRIS uses a separate 14,995-row anonymized single-company dataset with its own target, input schema, model card, and charts. The datasets are never mixed. The Vercel frontend bundles balanced logistic-regression artifacts generated from stratified 80/20 splits, so predictions and held-out confusion matrices work without a separate API host. Each prediction reports training rows and dataset rows. Configure the optional Python API for the richer cross-validated model pipeline and SHAP where supported. This remains an educational demonstration, not an employee decision system.

## Python backend handoff

See [docs/backend-contract.md](docs/backend-contract.md). `src/api.js` already isolates the prediction transport. Set `VITE_API_BASE_URL` only when your Python service implements the contract, then rebuild. This environment value is public configuration, never a place for secrets.

The optional Python backend lives in [`backend/`](backend/README.md): `train.py` serves the IBM profile, while `train_workplace.py` builds a separate HRIS model. `app.py` exposes prediction, model, and evaluation endpoints for both sources. `export_browser_models.py` regenerates the compact frontend model cards and their held-out metrics:

```sh
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
.venv/bin/python backend/train.py
.venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

In frontend mode, the model cards provide fitted logistic-regression inference and real held-out metrics. When the Python service is configured, the UI renders returned metadata and SHAP contributions when supported.

## Research datasets and analysis

Beyond the bundled IBM sample, three **real-world datasets** were located on GitHub and added under [`data/`](data/README.md) (provenance, SHA-256 checksums and a re-fetch script included):

- `data/hr-comma-sep.csv` — 14,995 employees from a real anonymized company HRIS export (`left` target)
- `data/av-job-change-train.csv` — 18,359 real candidates from the Analytics Vidhya job-change hackathon (`target`)
- `data/academic-faculty.csv` + `data/academic-department-attrition.csv` — published census of 6,793 US physics faculty, 2011→2017 (Morgan, Way & Clauset 2018, arXiv:1804.02760)

`analysis/` runs the full study (`python3 analysis/run_all.py`): statistical tests with effect sizes (chi-square/Cramér's V, Mann–Whitney/Cohen's d, odds ratios), per-dataset EDA figures, and a modeling study (logistic regression vs random forest vs XGBoost, 5-fold CV, held-out ROC-AUC/PR-AUC/F1/Brier, calibration, permutation importance, SHAP). Findings are written up research-style in **[reports/attrition-research-report.md](reports/attrition-research-report.md)** (figures in `reports/figures/`, raw results in `reports/results/`). Headline: across the real datasets, attrition concentrates in early-tenure, low-paid, overworked or dissatisfied employees (e.g. overtime OR 3.8 and 7-project employees leaving 100% of the time), while gender and performance rating show no signal; held-out ROC-AUC is 0.84 (IBM sample), 0.995 (real company HRIS), 0.63 (job-change intent).

## Structure

```text
src/
  main.jsx       Five pages, shared UI, charts and app state
  styles.css     Design tokens, responsive layouts, reduced-motion support
  data.js        CSV parsing, source-specific validation, and dataset summaries
  api.js         Bundled trained-model inference / optional Python API adapter
public/data/     IBM and HRIS CSVs plus compact fitted model cards
tests/         Dataset and profile validation checks
docs/          Python API handoff and validation notes
```

## Accessibility and responsive behavior

Semantic navigation and labels, keyboard-accessible charts, visible focus states, live prediction feedback, mobile navigation, reduced-motion handling, and print styles are included. Color is supplemented with text and values. Desktop interactions and the 390px mobile layout/navigation were checked in a browser; physical-device testing is still recommended. Fonts use Google Fonts with local Georgia/Arial fallbacks.

## Scope

The repository includes an optional Python training and inference service, but the live Vercel URL currently serves the frontend only; no backend deployment is claimed. There is no user authentication or employee database. This is an educational demonstration, **not an employment decision system**.
