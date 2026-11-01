# Python prediction service

`backend/` trains separate models for fictional IBM sample attrition and anonymized single-company HRIS turnover. With `VITE_API_BASE_URL`, the browser sends profiles to the selected API endpoint. Without it, the frontend uses bundled logistic-regression cards built from stratified 80/20 splits and reports their training counts and held-out evaluation.

## Endpoints

- `GET /health` — service and artifact status
- `GET /model` — model version, algorithm, ordered features, dataset fingerprint, model card, and valid department-to-role options
- `GET /evaluation` — cross-validation and held-out metrics
- `POST /predict` — validated IBM profile, probability, model version/type, target, training counts and SHAP contributions
- `GET /workplace/model`, `GET /workplace/evaluation`, `POST /workplace/predict` — separate HRIS turnover model contract

Training is offline. The inference service loads the saved pipeline once; it does not train during requests.

## Prediction request

All 20 keys are required. Numeric values are integers. `MonthlyIncome` uses the source dataset's units per month because the IBM sample does not document currency. `DistanceFromHome` is in miles at the API boundary; the UI converts kilometre input to miles.

```json
{
  "JobRole": "Sales Executive",
  "Department": "Sales",
  "BusinessTravel": "Travel_Rarely",
  "MaritalStatus": "Single",
  "OverTime": "Yes",
  "Age": 32,
  "JobLevel": 1,
  "MonthlyIncome": 4000,
  "JobSatisfaction": 2,
  "WorkLifeBalance": 2,
  "EnvironmentSatisfaction": 3,
  "JobInvolvement": 3,
  "YearsAtCompany": 3,
  "YearsSinceLastPromotion": 1,
  "YearsWithCurrManager": 2,
  "YearsInCurrentRole": 2,
  "TotalWorkingYears": 8,
  "StockOptionLevel": 0,
  "NumCompaniesWorked": 2,
  "DistanceFromHome": 24
}
```

The backend rejects unsupported category pairs, out-of-range values, tenure contradictions, and incomplete inputs with HTTP 422. The probability is `P(Attrition=Yes)` for this IBM sample; the dataset defines no prediction time window.

## Prediction response

```json
{
  "probability": 0.42,
  "model_version": "2.0.0+20261101",
  "model_type": "logreg",
  "target": "P(Attrition=Yes)",
  "explanation": {
    "method": "shap",
    "output": "log-odds",
    "base_value": -1.2,
    "contributions": []
  }
}
```

The numeric values above illustrate the response shape; they are not measured results. SHAP values are specific to the selected fitted model and expressed in log-odds. Positive values raise that model output and negative values lower it; they are associations, not causes.

## Local run

```sh
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
.venv/bin/python backend/train.py
THREAD_ALLOWED_ORIGINS=http://localhost:5173 .venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

Model artifacts and held-out metrics are generated under `backend/artifacts/` and are intentionally not committed. Configure the API origin in the frontend build environment and restrict `THREAD_ALLOWED_ORIGINS` to the exact frontend origin in deployment. `VITE_API_BASE_URL` is public configuration, not a place for secrets.

## Scope and limits

The Vercel app can run the fitted browser logistic models from bundled static cards; this does not deploy the optional Python service. The IBM and HRIS sources have different targets, feature schemas, forms and fitted models. They remain separate and must not be pooled.

The IBM records are synthetic. Performance on them does not establish validity for real employees or another organization. This project is an educational demonstration, not a hiring, firing, promotion, or employee scoring system. Review calibration, subgroup performance, privacy, access controls, and representative consented data before considering any operational use.
