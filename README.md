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

The explainability and evaluation panels intentionally remain empty until their future data contracts and model validation are implemented. The what-if explorer remains labeled as an untrained demo even when the prediction API is configured.

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
