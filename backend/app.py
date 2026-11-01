"""FastAPI reference implementation of docs/backend-contract.md.

Endpoints
  GET  /health       liveness
  GET  /model        model card / metadata (contract: "/model metadata")
  GET  /evaluation   held-out metrics + confusion matrix (contract: "/evaluation")
  POST /predict      {15 profile features} -> {probability, model_version, explanation}

Run (from the repository root, after `python backend/train.py`):

    .venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000

Then point the frontend at it:  VITE_API_BASE_URL=http://localhost:8000
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal, Optional

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from model import (BUSINESS_TRAVEL, DEPARTMENTS, FEATURES, MARITAL_STATUSES, RANGES, ROLES, ROLES_BY_DEPARTMENT,
                   explainer_kind, shap_contributions, to_frame)
from train_workplace import FEATURES as WORKPLACE_FEATURES

ARTIFACTS = BACKEND / "artifacts"


class Profile(BaseModel):
    """The 20 IBM model features, validated to match the browser form."""
    JobRole: Literal[tuple(ROLES)]
    Department: Literal[tuple(DEPARTMENTS)]
    BusinessTravel: Literal[tuple(BUSINESS_TRAVEL)]
    MaritalStatus: Literal[tuple(MARITAL_STATUSES)]
    OverTime: Literal["Yes", "No"]
    Age: int = Field(..., ge=18, le=60)
    JobLevel: int = Field(..., ge=1, le=5)
    MonthlyIncome: int = Field(..., ge=RANGES["MonthlyIncome"][0], le=RANGES["MonthlyIncome"][1])
    JobSatisfaction: int = Field(..., ge=1, le=4)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    JobInvolvement: int = Field(..., ge=1, le=4)
    YearsAtCompany: int = Field(..., ge=0, le=40)
    YearsSinceLastPromotion: int = Field(..., ge=0, le=40)
    YearsWithCurrManager: int = Field(..., ge=0, le=40)
    YearsInCurrentRole: int = Field(..., ge=0, le=40)
    TotalWorkingYears: int = Field(..., ge=0, le=40)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    NumCompaniesWorked: int = Field(..., ge=0, le=9)
    DistanceFromHome: int = Field(..., ge=1, le=100)

    @model_validator(mode="after")
    def tenure_consistency(self):
        if self.JobRole not in ROLES_BY_DEPARTMENT[self.Department]:
            raise ValueError("JobRole is not available in the selected Department")
        if self.YearsSinceLastPromotion > self.YearsAtCompany:
            raise ValueError("YearsSinceLastPromotion cannot exceed YearsAtCompany")
        if self.YearsWithCurrManager > self.YearsAtCompany:
            raise ValueError("YearsWithCurrManager cannot exceed YearsAtCompany")
        if self.YearsInCurrentRole > self.YearsAtCompany:
            raise ValueError("YearsInCurrentRole cannot exceed YearsAtCompany")
        if self.TotalWorkingYears < self.YearsAtCompany:
            raise ValueError("TotalWorkingYears cannot be less than YearsAtCompany")
        return self


class WorkplaceProfile(BaseModel):
    """Feature contract for the separate anonymized HRIS turnover dataset."""
    satisfaction_level: float = Field(..., ge=0.09, le=1.0)
    last_evaluation: float = Field(..., ge=0.36, le=1.0)
    number_project: int = Field(..., ge=2, le=7)
    average_montly_hours: int = Field(..., ge=96, le=310)
    time_spend_company: int = Field(..., ge=2, le=10)
    Work_accident: int = Field(..., ge=0, le=1)
    promotion_last_5years: int = Field(..., ge=0, le=1)
    Department: Literal["IT", "RandD", "accounting", "hr", "management", "marketing", "product_mng", "sales", "support", "technical"]
    salary: Literal["low", "medium", "high"]


def workplace_shap(artifact: dict, profile: dict, top: int = 8) -> dict:
    """Return SHAP contributions for the real-HRIS model, if its explainer works."""
    import numpy as np
    import pandas as pd
    import shap

    pipe = artifact["pipeline"]
    row = pd.DataFrame([{key: profile[key] for key in WORKPLACE_FEATURES}], columns=WORKPLACE_FEATURES)
    transformed = pipe.named_steps["prep"].transform(row)
    names = pipe.named_steps["prep"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    if explainer_kind(clf) == "tree":
        explainer = shap.TreeExplainer(clf)
    else:
        explainer = shap.LinearExplainer(clf, np.asarray(artifact.get("background", transformed)))
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[1]
    elif getattr(values, "ndim", 2) == 3:
        values = values[:, :, 1] if values.shape[-1] == 2 else values[..., 0]
    values = np.asarray(values).reshape(-1)
    order = np.argsort(-np.abs(values))[:top]
    contributions = []
    for i in order:
        detail = str(names[i])
        feature = detail.split("__", 1)[1] if "__" in detail else detail
        for key in WORKPLACE_FEATURES:
            if detail.startswith(f"num__{key}") or detail.startswith(f"cat__{key}_"):
                feature = key
                break
        contributions.append({"feature": feature, "detail": detail,
                              "value": profile.get(feature), "contribution": float(values[i])})
    expected = explainer.expected_value
    if isinstance(expected, list):
        expected = expected[1]
    base = float(np.asarray(expected).reshape(-1)[0])
    return {"method": "shap", "output": "log-odds", "base_value": base,
            "contributions": contributions}


def create_app(artifacts_dir: Path = ARTIFACTS) -> FastAPI:
    from contextlib import asynccontextmanager

    state = {"artifact": None, "metrics": None, "workplace_artifact": None, "workplace_metrics": None}

    def load(force: bool = False):
        if state["artifact"] is None or force:
            path = artifacts_dir / "model.joblib"
            if not path.exists():
                raise HTTPException(503, "model not trained - run `python backend/train.py` first")
            state["artifact"] = joblib.load(path)
            mpath = artifacts_dir / "metrics.json"
            state["metrics"] = json.loads(mpath.read_text()) if mpath.exists() else None
        return state["artifact"]

    def load_workplace():
        if state["workplace_artifact"] is None:
            path = artifacts_dir / "workplace-turnover" / "model.joblib"
            if not path.exists():
                raise HTTPException(503, "workplace model is not trained - run `python backend/train_workplace.py` first")
            state["workplace_artifact"] = joblib.load(path)
            mpath = artifacts_dir / "workplace-turnover" / "metrics.json"
            state["workplace_metrics"] = json.loads(mpath.read_text()) if mpath.exists() else None
        return state["workplace_artifact"]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            load()
        except HTTPException:
            pass  # /predict and /model will return 503 until train.py has run
        yield

    app = FastAPI(title="THREAD attrition prediction service",
                  version="1.0.0",
                  description="Reference Python handoff for the THREAD retention-intelligence "
                              "frontend. Educational demo - not an employment decision system.",
                  lifespan=lifespan)
    # Restrict browser access to the demo frontend and local development.
    allowed_origins = [origin.strip() for origin in os.getenv(
        "THREAD_ALLOWED_ORIGINS",
        "https://employeeattrition-topaz.vercel.app,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",") if origin.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    @app.get("/")
    def root():
        return {"service": "THREAD attrition prediction service", "docs": "/docs",
                "endpoints": ["/health", "/model", "/evaluation", "/predict",
                              "/workplace/model", "/workplace/evaluation", "/workplace/predict"]}

    @app.get("/health")
    def health():
        loaded = state["artifact"] is not None
        if not loaded:
            try:
                load()
                loaded = True
            except HTTPException:
                loaded = False
        return {"status": "ok", "model_loaded": loaded}

    @app.get("/model")
    def model_info():
        art = load()
        card_path = artifacts_dir / "model_card.json"
        card = json.loads(card_path.read_text()) if card_path.exists() else {}
        return {
            "model_version": art["model_version"],
            "model_type": art["model_type"],
            "target": "P(Attrition=Yes)",
            "features": art["features"],
            "roles_by_department": ROLES_BY_DEPARTMENT,
            "trained_at": art["trained_at"],
            "training_data": art["dataset"],
            "explainer": explainer_kind(art["pipeline"].named_steps["clf"]),
            "card": card,
        }

    @app.get("/workplace/model")
    def workplace_model_info():
        art = load_workplace()
        return {"model_id": "workplace-turnover", "model_version": art["model_version"],
                "model_type": art["model_type"], "target": "P(left=1)", "features": art["features"],
                "training_data": art["dataset"], "data_kind": "real anonymized company HRIS",
                "warning": "This single-company dataset is not a general population estimate."}

    @app.get("/workplace/evaluation")
    def workplace_evaluation():
        load_workplace()
        metrics = state["workplace_metrics"]
        if metrics is None:
            raise HTTPException(503, "workplace metrics are missing; retrain the model")
        return {"model_id": metrics["model_id"], "model_version": metrics["model_version"],
                "model_type": metrics["model_type"], "seed": metrics["seed"],
                "n_total": metrics["n_total"], "n_train": metrics["n_train"],
                "n_test": metrics["n_test"], "cv": metrics["cv"], "held_out_test": metrics["test"]}

    @app.post("/workplace/predict")
    def predict_workplace(profile: WorkplaceProfile):
        art = load_workplace()
        row = pd.DataFrame([profile.model_dump()], columns=WORKPLACE_FEATURES)
        probability = float(art["pipeline"].predict_proba(row)[0, 1])
        try:
            explanation = workplace_shap(art, profile.model_dump())
        except Exception as exc:
            explanation = {"method": "shap", "error": str(exc), "contributions": []}
        return {"probability": round(probability, 4), "model_id": "workplace-turnover",
                "model_version": art["model_version"], "model_type": art["model_type"],
                "target": "P(left=1)", "training_entries": art["dataset"]["n_train"],
                "dataset_entries": art["dataset"]["n"], "explanation": explanation,
                "warning": "This model reflects one anonymized company dataset and is not a real employee assessment."}

    @app.get("/evaluation")
    def evaluation():
        load()
        if state["metrics"] is None:
            raise HTTPException(503, "metrics.json missing - run `python backend/train.py`")
        m = state["metrics"]
        return {"model_version": m["model_version"], "model_type": m["model_type"], "seed": m["seed"],
                "n_total": m["n_total"], "n_train": m["n_train"], "n_test": m["n_test"],
                "cv": m["cv"], "held_out_test": m["test"]}

    @app.post("/predict")
    def predict(profile: Profile):
        art = load()
        pipe = art["pipeline"]
        row = to_frame(profile.model_dump())
        probability = float(pipe.predict_proba(row)[0, 1])
        explanation: Optional[dict] = None
        try:
            explanation = shap_contributions(pipe.named_steps["clf"], pipe.named_steps["prep"], row,
                                             background=art.get("background"))
        except Exception as exc:  # never fail a prediction because of the explainer
            explanation = {"method": "shap", "error": str(exc), "contributions": []}
        return {
            "probability": round(probability, 4),
            "model_version": art["model_version"],
            "model_type": art["model_type"],
            "target": "P(Attrition=Yes)",
            "training_entries": art["dataset"]["n_train"],
            "dataset_entries": art["dataset"]["n"],
            "explanation": explanation,
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
