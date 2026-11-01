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
| `/#overview` | Dataset summary, conceptual signal flow, calculated overtime and satisfaction charts |
| `/#predict` | IBM profile form with dependent department/role options, 20 model inputs, validation, loading/error states and probability gauge |
| `/#drivers` | Model-returned SHAP contributions and same-model what-if comparison when the API is connected; explicitly labeled local illustration otherwise |
| `/#patterns` | Role, tenure and overtime cohort filters, role comparisons, heatmap, accessible tooltips, CSV export |
| `/#model` | Training workflow, live metadata and held-out metrics when returned by the API, confusion matrix, responsible-use notes |

A guided tour connects the five pages. Forms and the latest result persist across page navigation during the current session; employee inputs are not saved to browser storage.

## Data and scientific honesty

The included CSV contains 1,470 fictional IBM HR sample records, including 237 `Attrition=Yes` records and 1,233 `No` records. Descriptive charts are calculated from these records in the browser, including all cohort filters. The flowing thread paths are explicitly conceptual rather than employee trajectories.

- [Dataset listing](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- [CSV mirror used](https://github.com/IamVasil/IBM-HR-Analytics-Employee-Attrition-Performance/blob/main/WA_Fn-UseC_-HR-Employee-Attrition.csv)
- Local file: `public/data/ibm-hr.csv`

The prediction studio offers **IBM sample profiles** and **manual non-identifying scenario entry**. In frontend-only mode, it compares the entered 20-feature profile with its nearest records in the bundled, fictional IBM sample and reports a sample-based similarity estimate; it is not a trained, validated or calibrated model and must not be interpreted as an employee assessment. Monthly income remains in the source dataset's units because its currency is undocumented; commute distance is converted to the API's miles unit when kilometres are selected. For a trained model and real SHAP explanations, configure the optional Python backend. The what-if explorer uses a second API prediction when the trained service is configured; it is explicitly illustrative in browser-only mode. No validation metrics are fabricated.

## Python backend handoff

See [docs/backend-contract.md](docs/backend-contract.md). `src/api.js` already isolates the prediction transport. Set `VITE_API_BASE_URL` only when your Python service implements the contract, then rebuild. This environment value is public configuration, never a place for secrets.

The optional Python backend lives in [`backend/`](backend/README.md): `train.py` trains and evaluates a model on the same 20 IBM profile features used by the form (5-fold CV model selection, held-out metrics, model card), and `app.py` exposes `POST /predict` (with SHAP explanations), `GET /model` and `GET /evaluation`, plus contract tests (`pytest backend/test_api.py`):

```sh
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
.venv/bin/python backend/train.py
.venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

In browser demo mode, explainability and evaluation do not invent metrics. When the Python service is configured, the UI renders returned SHAP contributions, model metadata, held-out evaluation, and a second prediction for what-if comparisons.

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
  data.js        CSV parsing, profile validation, and IBM-sample similarity estimate
  api.js         Local sample estimator / optional Python prediction API adapter
public/data/     Bundled sample CSV
tests/         Dataset and profile validation checks
docs/          Python API handoff and validation notes
```

## Accessibility and responsive behavior

Semantic navigation and labels, keyboard-accessible charts, visible focus states, live prediction feedback, mobile navigation, reduced-motion handling, and print styles are included. Color is supplemented with text and values. Desktop interactions and the 390px mobile layout/navigation were checked in a browser; physical-device testing is still recommended. Fonts use Google Fonts with local Georgia/Arial fallbacks.

## Scope

The repository includes an optional Python training and inference service, but the live Vercel URL currently serves the frontend only; no backend deployment is claimed. There is no user authentication or employee database. This is an educational demonstration, **not an employment decision system**.
