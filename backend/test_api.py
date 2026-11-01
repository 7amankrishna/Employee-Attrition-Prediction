"""Contract tests for the FastAPI handoff (docs/backend-contract.md).

Run from the repository root:

    .venv/bin/python -m pytest backend/test_api.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import train  # noqa: E402
from app import create_app  # noqa: E402

VALID = {
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
    "DistanceFromHome": 24,
}


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    train.train_model(verbose=False, out_dir=out)
    return TestClient(create_app(artifacts_dir=out))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model_loaded"] is True


def test_model_metadata(client):
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"]
    assert body["features"] == ["JobRole", "Department", "BusinessTravel", "MaritalStatus", "OverTime",
                                "Age", "JobLevel", "MonthlyIncome", "JobSatisfaction", "WorkLifeBalance",
                                "YearsAtCompany", "YearsSinceLastPromotion", "YearsWithCurrManager",
                                "NumCompaniesWorked", "DistanceFromHome"]
    assert body["model_type"] in {"logreg", "rf", "xgb"}


def test_evaluation_contract(client):
    r = client.get("/evaluation")
    assert r.status_code == 200
    body = r.json()
    for key in ("cv", "held_out_test"):
        assert key in body
    t = body["held_out_test"]
    for key in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier", "confusion_matrix"):
        assert key in t
    assert 0 <= t["roc_auc"] <= 1


def test_predict_contract(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"probability", "model_version", "explanation"}
    assert isinstance(body["probability"], (int, float)) and 0 <= body["probability"] <= 1
    assert isinstance(body["model_version"], str) and body["model_version"]
    assert body["explanation"]["method"] == "shap"
    assert isinstance(body["explanation"]["contributions"], list)
    assert all({"feature", "contribution"} <= set(c) for c in body["explanation"]["contributions"])


def test_predict_probability_increases_with_risk(client):
    low = {**VALID, "OverTime": "No", "JobSatisfaction": 4, "WorkLifeBalance": 4,
           "MonthlyIncome": 15000, "YearsAtCompany": 15, "YearsSinceLastPromotion": 0}
    high = {**VALID, "OverTime": "Yes", "JobSatisfaction": 1, "WorkLifeBalance": 1,
            "MonthlyIncome": 2000, "YearsAtCompany": 1, "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 1, "DistanceFromHome": 29}
    p_low = client.post("/predict", json=low).json()["probability"]
    p_high = client.post("/predict", json=high).json()["probability"]
    assert p_high > p_low


@pytest.mark.parametrize("patch", [
    {"JobRole": "Not A Role"},
    {"OverTime": "Maybe"},
    {"MonthlyIncome": 10},
    {"MonthlyIncome": 200000},
    {"MonthlyIncome": 4000.5},
    {"JobSatisfaction": 0},
    {"JobSatisfaction": 5},
    {"WorkLifeBalance": 9},
    {"YearsAtCompany": -1},
    {"YearsAtCompany": 41},
    {"YearsSinceLastPromotion": 5, "YearsAtCompany": 3},  # exceeds tenure
    {"DistanceFromHome": 0},
    {"DistanceFromHome": 101},
])
def test_predict_rejects_invalid(client, patch):
    r = client.post("/predict", json={**VALID, **patch})
    assert r.status_code == 422, (patch, r.status_code, r.text)


def test_predict_rejects_missing_field(client):
    body = {k: v for k, v in VALID.items() if k != "OverTime"}
    assert client.post("/predict", json=body).status_code == 422
