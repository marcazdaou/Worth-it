import json
import math

import pytest
from fastapi.testclient import TestClient

from app.main import META_PATH, app

META = json.loads(META_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the startup hook, so the model is loaded once
        yield c


def predict(client, **body):
    return client.post("/predict", json={"title": "Data Analyst", **body})


# ── /health and /options ─────────────────────────────────────────────────────
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": META["model"]}


def test_options_lists_are_filled_and_sorted(client):
    options = client.get("/options").json()
    for key in ["cities", "countries", "skills", "employment_types"]:
        assert options[key], key
    for key in ["cities", "countries", "skills"]:
        assert options[key] == sorted(options[key]), key
    assert "New York" in options["cities"]
    assert "python" in options["skills"]


# ── /predict ─────────────────────────────────────────────────────────────────
def test_predict_returns_a_sensible_range(client):
    r = predict(client, title="Senior Data Engineer", city="New York", skills=["python", "sql", "aws"])
    assert r.status_code == 200
    body = r.json()
    assert body["role_family"] == "Data & AI"
    assert body["seniority"] == "Senior / Lead"
    assert body["low"] < body["predicted_salary"] < body["high"]
    assert 50_000 < body["predicted_salary"] < 500_000
    assert body["range_hit_rate"] == META["test"]["interval_coverage"]


def test_range_width_matches_the_model_interval(client):
    body = predict(client).json()
    expected = math.exp(META["interval"]["log_high"] - META["interval"]["log_low"])
    assert body["high"] / body["low"] == pytest.approx(expected, rel=0.02)


def test_seniority_raises_pay(client):
    junior = predict(client, title="Junior Software Engineer").json()["predicted_salary"]
    senior = predict(client, title="Senior Software Engineer").json()["predicted_salary"]
    assert senior > junior


def test_city_overrides_wrong_country(client):
    # New York is in the US whatever country is sent, so both requests must match.
    a = predict(client, city="New York", country="United States").json()
    b = predict(client, city="New York", country="Germany").json()
    assert a["predicted_salary"] == b["predicted_salary"]


def test_skills_are_case_and_duplicate_insensitive(client):
    a = predict(client, skills=["python", "sql"]).json()
    b = predict(client, skills=["SQL", " Python ", "sql"]).json()
    assert a["predicted_salary"] == b["predicted_salary"]


def test_unknown_city_and_skill_still_predict(client):
    r = predict(client, city="Atlantis", skills=["not-a-real-skill"])
    assert r.status_code == 200


def test_matches_the_original_model_pipeline(client):
    """The API must give the same answer as calling the model the way Job-Market-Analysis does."""
    import numpy as np
    import pandas as pd
    from app.features import model_frame
    from app.main import load_model

    row = pd.DataFrame([{"title": "Machine Learning Engineer", "company": None, "industry": None,
                         "department": None, "country": "United States", "city": "Seattle",
                         "employment_type": "Full Time", "remote": True, "skills": ["python", "pytorch"]}])
    direct = float(np.exp(load_model().predict(model_frame(row))[0]))
    api = predict(client, title="Machine Learning Engineer", city="Seattle", country="United States",
                  remote=True, skills=["python", "pytorch"]).json()["predicted_salary"]
    assert api == pytest.approx(direct, abs=250)  # API rounds to the nearest $500


@pytest.mark.parametrize("body", [
    {"title": ""},
    {"title": "   "},
    {"title": "x" * 121},
    {"title": "Analyst", "employment_type": "Freelance"},
    {"title": "Analyst", "skills": ["s"] * 31},
    {},
])
def test_predict_rejects_bad_input(client, body):
    assert client.post("/predict", json=body).status_code == 422


# ── /skills ──────────────────────────────────────────────────────────────────
def test_skills_for_a_data_title(client):
    body = client.get("/skills", params={"title": "Senior Data Scientist"}).json()
    assert body["role_family"] == "Data & AI"
    assert body["level_used"] == "Senior / Lead"
    names = [s["skill"] for s in body["top_skills"]]
    assert "python" in names
    shares = [s["share"] for s in body["top_skills"]]
    assert shares == sorted(shares, reverse=True)
    assert all(0 < s <= 1 for s in shares)


def test_skills_rejects_short_title(client):
    assert client.get("/skills", params={"title": " a "}).status_code == 422
    assert client.get("/skills").status_code == 422
