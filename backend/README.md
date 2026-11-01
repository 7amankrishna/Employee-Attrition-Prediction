# Python prediction service

This is the optional trained-model backend for the prediction studio. The browser includes compact logistic-regression model cards for both the fictional IBM sample and the separate anonymized HRIS dataset. The Python service supplies full pipeline inference and SHAP when supported.

## Model and API

`train.py` trains a scikit-learn pipeline on 20 available features in `public/data/ibm-hr.csv`: job role, department, business travel, marital status, overtime, age, job level, source-unit monthly income, job satisfaction, work–life balance, environment satisfaction, job involvement, company tenure, years since promotion, years with current manager, years in current role, total working years, stock option level, number of previous companies, and commute distance. It compares logistic regression, random forest, and XGBoost with 5-fold stratified CV, selects using CV ROC-AUC, evaluates on a held-out 20% split, and writes model, metrics, and model card artifacts (gitignored).

`train_workplace.py` trains and evaluates a separate model on `data/hr-comma-sep.csv`. `export_browser_models.py` regenerates the compact frontend model cards for both sources and checks their inference formula against scikit-learn. `app.py` exposes:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service and model liveness |
| `GET /model` | Model metadata and model card |
| `GET /evaluation` | CV and held-out metrics |
| `POST /predict` | Validated IBM profile → probability, training counts and SHAP explanation |
| `GET /workplace/model`, `/workplace/evaluation` | Metadata and held-out results for the separate HRIS model |
| `POST /workplace/predict` | Validated HRIS profile → probability, training counts and SHAP explanation when available |

The request contract is documented in [`../docs/backend-contract.md`](../docs/backend-contract.md). Monthly income remains in source dataset units because currency is undocumented; commute distance is miles at the API boundary. The UI converts kilometre input to miles. Department/role combinations and numeric ranges are validated, as are tenure relationships.

## Run

```sh
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
.venv/bin/python backend/train.py
.venv/bin/python backend/train_workplace.py
.venv/bin/python backend/export_browser_models.py
.venv/bin/python -m pytest backend/test_api.py -q
.venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Configure the frontend with `VITE_API_BASE_URL=http://localhost:8000` and rebuild. The API defaults to the Vercel origin and local development origins; set `THREAD_ALLOWED_ORIGINS` to the exact allowed origins at deployment. Add authentication and access controls before any private use, and do not expose employee-level predictions publicly.

## Limits

- The IBM HR dataset is a fictional synthetic sample, not IBM employee records. Model performance on it does not establish validity on real workers or another organization.
- A probability is not a time-to-departure, destiny, or employment decision. Do not use for hiring, firing, promotion, or other employment decisions.
- Before any operational use, retrain with properly consented and representative data, assess calibration, audit fairness, review privacy and employment law, and provide human oversight.
- When the service is configured, the what-if comparison sends the changed profile through the same prediction API. It remains a hypothetical model comparison and cannot establish causal effects. In the browser, what-if comparisons use the same bundled model card; no hand-written estimate is presented as a model result.
