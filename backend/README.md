# Python prediction service

This is the optional trained-model backend for the prediction studio. The browser-only experience still works without it by comparing the profile with similar rows from the bundled fictional IBM dataset; that fallback is not a validated model.

## Model and API

`train.py` trains a scikit-learn pipeline on 20 available features in `public/data/ibm-hr.csv`: job role, department, business travel, marital status, overtime, age, job level, source-unit monthly income, job satisfaction, work–life balance, environment satisfaction, job involvement, company tenure, years since promotion, years with current manager, years in current role, total working years, stock option level, number of previous companies, and commute distance. It compares logistic regression, random forest, and XGBoost with 5-fold stratified CV, selects using CV ROC-AUC, evaluates on a held-out 20% split, and writes model, metrics, and model card artifacts (gitignored).

`app.py` exposes:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service and model liveness |
| `GET /model` | Model metadata and model card |
| `GET /evaluation` | CV and held-out metrics |
| `POST /predict` | Validated 20-feature profile → probability and SHAP explanation |

The request contract is documented in [`../docs/backend-contract.md`](../docs/backend-contract.md). Monthly income remains in source dataset units because currency is undocumented; commute distance is miles at the API boundary. The UI converts kilometre input to miles. Department/role combinations and numeric ranges are validated, as are tenure relationships.

## Run

```sh
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
.venv/bin/python backend/train.py
.venv/bin/python -m pytest backend/test_api.py -q
.venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Configure the frontend with `VITE_API_BASE_URL=http://localhost:8000` and rebuild. The API defaults to the Vercel origin and local development origins; set `THREAD_ALLOWED_ORIGINS` to the exact allowed origins at deployment. Add authentication and access controls before any private use, and do not expose employee-level predictions publicly.

## Limits

- The IBM HR dataset is a fictional synthetic sample, not IBM employee records. Model performance on it does not establish validity on real workers or another organization.
- A probability is not a time-to-departure, destiny, or employment decision. Do not use for hiring, firing, promotion, or other employment decisions.
- Before any operational use, retrain with properly consented and representative data, assess calibration, audit fairness, review privacy and employment law, and provide human oversight.
- When the service is configured, the what-if comparison sends the changed profile through the same prediction API. It remains a hypothetical model comparison and cannot establish causal effects. In local sample-estimate mode it remains clearly labeled as illustrative.
