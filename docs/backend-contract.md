# Python prediction service handoff

> **Reference implementation available:** [`backend/`](../backend/) contains a working
> FastAPI service (`app.py`) plus trainer (`train.py`) and contract tests (`test_api.py`)
> that implement everything described below, including the `/model` and `/evaluation`
> endpoints from "Further integration work" and a real SHAP `explanation`. See
> [backend/README.md](../backend/README.md).

## Current integration

`src/api.js` uses the illustrative formula when `VITE_API_BASE_URL` is unset. When set, it sends `POST {base}/predict` with JSON, rejects non-2xx responses, validates that probability is a number between 0 and 1, and preserves form inputs on errors. Configure CORS for your actual frontend origin. This frontend contains no authentication or secret credentials.

Request:

```json
{
  "JobRole": "Sales Executive",
  "OverTime": "Yes",
  "MonthlyIncome": 4000,
  "JobSatisfaction": 2,
  "WorkLifeBalance": 2,
  "YearsAtCompany": 3,
  "YearsSinceLastPromotion": 1,
  "DistanceFromHome": 24
}
```

Required response:

```json
{
  "probability": 0.42,
  "model_version": "your-validated-model-version",
  "explanation": null
}
```

The value above is an API format example, not a model result. Probability means P(Attrition=Yes), not predicted time-to-departure. The sample dataset has no defined future prediction window.

Before connecting: train the model on the same eight features or explicitly update this contract and the UI. Fit preprocessing on training data only. Return consistent category mappings and validate all inputs server-side. Never silently fill missing trained features with arbitrary defaults.

## Further integration work

- Add `/model` metadata and `/evaluation` contracts, then load real held-out metrics and matrix counts.
- Add a supported explainer endpoint only after selecting and validating the model/explainer combination. Return baseline, output scale (probability or log-odds), contribution values, feature names, model version and explainer name. Do not label arbitrary feature importances as SHAP.
- Replace the what-if demo formula with two calls to the same model for baseline and scenario. Keep comparisons clearly non-causal.
- Define and validate probability calibration and risk-band thresholds. Current 30/60 cutoffs are UI demo conventions, not validated operational thresholds.
- Add API contract tests and timeout handling, then verify all loading/error states with the real service.

## Demonstration formula

The UI-only formula begins at 15 percentage points, adds 23 for overtime, 8 per satisfaction level below 4, 5 per balance level below 4, 7 for income under 4,500, 5 for tenure under 4 years, 0.15 per distance unit, and up to 8 for years since promotion. The total is clamped to 5–92 and rounded. These arbitrary values are for demonstrating interaction, not statistical claims.
