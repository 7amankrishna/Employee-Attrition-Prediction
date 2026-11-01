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
import sys
from pathlib import Path
from typing import Literal, Optional

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from model import (BUSINESS_TRAVEL, DEPARTMENTS, FEATURES, MARITAL_STATUSES, RANGES, ROLES,
                   explainer_kind, shap_contributions, to_frame)

ARTIFACTS = BACKEND / "artifacts"


class Profile(BaseModel):
    """The 15 modeled IBM sample features, validated to match the browser form."""
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
    YearsAtCompany: int = Field(..., ge=0, le=40)
    YearsSinceLastPromotion: int = Field(..., ge=0, le=40)
    YearsWithCurrManager: int = Field(..., ge=0, le=40)
    NumCompaniesWorked: int = Field(..., ge=0, le=9)
    DistanceFromHome: int = Field(..., ge=1, le=100)

    @model_validator(mode="after")
    def tenure_consistency(self):
        if self.YearsSinceLastPromotion > self.YearsAtCompany:
            raise ValueError("YearsSinceLastPromotion cannot exceed YearsAtCompany")
        if self.YearsWithCurrManager > self.YearsAtCompany:
            raise ValueError("YearsWithCurrManager cannot exceed YearsAtCompany")
        return self


def create_app(artifacts_dir: Path = ARTIFACTS) -> FastAPI:
    from contextlib import asynccontextmanager

    state = {"artifact": None, "metrics": None}

    def load(force: bool = False):
        if state["artifact"] is None or force:
            path = artifacts_dir / "model.joblib"
            if not path.exists():
                raise HTTPException(503, "model not trained - run `python backend/train.py` first")
            state["artifact"] = joblib.load(path)
            mpath = artifacts_dir / "metrics.json"
            state["metrics"] = json.loads(mpath.read_text()) if mpath.exists() else None
        return state["artifact"]

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
    # Demo service: allow the Vite dev server and the deployed frontend. Restrict in production.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    @app.get("/")
    def root():
        return {"service": "THREAD attrition prediction service", "docs": "/docs",
                "endpoints": ["/health", "/model", "/evaluation", "/predict"]}

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
            "features": art["features"],
            "trained_at": art["trained_at"],
            "training_data": art["dataset"],
            "explainer": explainer_kind(art["pipeline"].named_steps["clf"]),
            "card": card,
        }

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
            "explanation": explanation,
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
