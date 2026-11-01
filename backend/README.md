# Python prediction service

This is the optional trained-model backend for the prediction studio. The browser-only experience still works without it by comparing the profile with similar rows from the bundled fictional IBM dataset; that fallback is not a validated model.

## Model and API

`train.py` trains a scikit-learn pipeline on 15 available features in `public/data/ibm-hr.csv`: job role, department, business travel, marital status, overtime, age, job level, monthly income, job satisfaction, work–life balance, company tenure, years since promotion, years with current manager, number of previous companies, and commute distance. It compares logistic regression, random forest, and XGBoost with 5-fold stratified CV, selects using CV ROC-AUC, evaluates on a held-out 20% split, and writes model, metrics, and model card artifacts (gitignored).

`app.py` exposes:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service and model liveness |
| `GET /model` | Model metadata and model card |
| `GET /evaluation` | CV and held-out metrics |
| `POST /predict` | Validated 15-feature profile → probability and SHAP explanation |

The request contract is documented in [`../docs/backend-contract.md`](../docs/backend-contract.md). Income is monthly USD and commute distance is miles; the UI converts INR and km inputs to these model units. All fields are range-checked and the tenure fields cannot exceed company tenure.

## Run

```sh
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
.venv/bin/python backend/train.py
.venv/bin/python -m pytest backend/test_api.py -q
.venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Configure the frontend with `VITE_API_BASE_URL=http://localhost:8000` and rebuild. In production, restrict CORS to the frontend origin, add authentication and access controls, and do not expose employee-level predictions publicly.

## Limits

- The IBM HR dataset is a fictional synthetic sample, not IBM employee records. Model performance on it does not establish validity on real workers or another organization.
- A probability is not a time-to-departure, destiny, or employment decision. Do not use for hiring, firing, promotion, or other employment decisions.
- Before any operational use, retrain with properly consented and representative data, assess calibration, audit fairness, review privacy and employment law, and provide human oversight.
- The UI's what-if illustration is separate from this model; it does not call this API and cannot establish causal effects.
