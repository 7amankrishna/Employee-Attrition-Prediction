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
| `/#predict` | Eight-feature employee form, validation, loading/error states and probability gauge |
| `/#drivers` | Honest explainer placeholder and interactive illustrative what-if comparison |
| `/#patterns` | Role, tenure and overtime cohort filters, role comparisons, heatmap, accessible tooltips, CSV export |
| `/#model` | Interactive planned ML workflow, empty evaluation metrics, confusion matrix, responsible-use notes |

A guided tour connects the five pages. Forms and the latest result persist across page navigation during the current session; employee inputs are not saved to browser storage.

## Data and scientific honesty

The included CSV contains 1,470 fictional IBM HR sample records, including 237 `Attrition=Yes` records and 1,233 `No` records. Descriptive charts are calculated from these records in the browser, including all cohort filters. The flowing thread paths are explicitly conceptual rather than employee trajectories.

- [Dataset listing](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- [CSV mirror used](https://github.com/IamVasil/IBM-HR-Analytics-Employee-Attrition-Performance/blob/main/WA_Fn-UseC_-HR-Employee-Attrition.csv)
- Local file: `public/data/ibm-hr.csv`

**There is no trained model in this frontend.** Prediction and what-if values use a deliberately simple, untrained demonstration formula and are labeled illustrative. This formula does not learn from the dataset, is not calibrated, and must not be interpreted as an employee assessment. Job role is collected for the future API but does not affect this formula. No SHAP values, accuracy, precision, recall, F1, ROC-AUC, or confusion-matrix counts are invented.

## Python backend handoff

See [docs/backend-contract.md](docs/backend-contract.md). `src/api.js` already isolates the prediction transport. Set `VITE_API_BASE_URL` only when your Python service implements the contract, then rebuild. This environment value is public configuration, never a place for secrets.

A complete **reference implementation** of that contract now lives in [`backend/`](backend/README.md): `train.py` trains and evaluates a model on the eight contract features (5-fold CV model selection, held-out metrics, model card), and `app.py` is a FastAPI service exposing `POST /predict` (with real SHAP explanations), `GET /model` and `GET /evaluation`, plus contract tests (`pytest backend/test_api.py`):

```sh
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
.venv/bin/python backend/train.py
.venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

The explainability and evaluation panels intentionally remain empty in demo mode until the prediction API is configured. The what-if explorer remains labeled as an untrained demo even when the prediction API is configured.

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
  data.js        CSV parsing, aggregations, validation, labeled demo formula
  api.js         Demo / future Python prediction API adapter
public/data/     Bundled sample CSV
tests/         Dataset and profile validation checks
docs/          Python API handoff and validation notes
```

## Accessibility and responsive behavior

Semantic navigation and labels, keyboard-accessible charts, visible focus states, live prediction feedback, mobile navigation, reduced-motion handling, and print styles are included. Color is supplemented with text and values. Desktop interactions and the 390px mobile layout/navigation were checked in a browser; physical-device testing is still recommended. Fonts use Google Fonts with local Georgia/Arial fallbacks.

## Scope

Frontend only. No training, Python server, user authentication, employee database, or employment decision automation. This is an educational demonstration, **not an employment decision system**.
