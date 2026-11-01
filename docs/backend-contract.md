# Python prediction service

`backend/` trains a model on the fictional IBM HR Analytics sample and exposes a FastAPI service. The browser sends a profile to `POST /predict` when `VITE_API_BASE_URL` is set. Without that setting, the UI explicitly labels its local nearest-sample result as an exploratory estimate; it does not quietly claim a trained model is running.

## Endpoints

- `GET /health` — service and artifact status
- `GET /model` — model version, algorithm, ordered features, dataset fingerprint, model card, and valid department-to-role options
- `GET /evaluation` — cross-validation and held-out metrics
- `POST /predict` — validated IBM profile, probability, model version/type, target, and SHAP contributions

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

The deployed Vercel app currently hosts the Vite frontend. This change prepares the Python API contract and UI integration; it does not deploy a Python service or verify a live trained prediction. The connected model is IBM-only. The HRIS and job-change datasets remain research datasets and have different targets and feature schemas; they are not mixed into this IBM training table. Before serving those as prediction modes, each needs its own documented, trained, evaluated pipeline and matching form.

The IBM records are synthetic. Performance on them does not establish validity for real employees or another organization. This project is an educational demonstration, not a hiring, firing, promotion, or employee scoring system. Review calibration, subgroup performance, privacy, access controls, and representative consented data before considering any operational use.
