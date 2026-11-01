# Python prediction service handoff

> The working reference implementation is in [`backend/`](../backend/): FastAPI,
> trainer, held-out evaluation, model card, SHAP explanations, and API tests.

## Current integration

`src/api.js` uses a local nearest-neighbour similarity estimate against the bundled IBM sample when `VITE_API_BASE_URL` is unset. When configured, it sends `POST {base}/predict`, validates the probability response, and preserves form inputs on errors. Configure CORS for the deployed frontend; this frontend has no authentication or secrets.

The contract contains the same 15 fields used by the studio:

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
  "YearsAtCompany": 3,
  "YearsSinceLastPromotion": 1,
  "YearsWithCurrManager": 2,
  "NumCompaniesWorked": 2,
  "DistanceFromHome": 24
}
```

`MonthlyIncome` is monthly USD and `DistanceFromHome` is miles, regardless of the frontend display units. The INR input uses a fixed, clearly disclosed demo conversion; the distance input converts kilometres to miles. Both model values are normalized before sending.

Required response:

```json
{
  "probability": 0.42,
  "model_version": "your-validated-model-version",
  "explanation": null
}
```

The value is an API format example, not a model result. Probability means P(Attrition=Yes), not predicted time-to-departure; the dataset has no defined prediction window. The optional backend validates categories, ranges, and tenure consistency server-side.

The browser-only estimate uses 50 weighted nearest IBM records and shrinks toward the dataset-wide rate. It is clearly labeled, but it is not a trained, held-out-validated or calibrated model. The Python backend provides the trained alternative. Both use fictional IBM data: neither has demonstrated accuracy on real employees.

## Further work

- Validate performance and calibration on consented, representative real data before use beyond education.
- Add fairness audits, access controls, retention/deletion policies, and security review before handling real employee data.
- Keep any what-if comparison clearly non-causal; the separate UI explorer is still a formula illustration, not a backend inference.
- Risk bands are demo conventions, not validated operational thresholds.
