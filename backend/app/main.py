"""Worth It API: salary estimates and in-demand skills for a job title.

The salary model and the skills data both come from Job-Market-Analysis
(394,300 job postings). This service only reads them; it never retrains.

Endpoints
    GET  /health    is the server up and the model loaded?
    GET  /options   dropdown values for the app (cities, countries, skills, ...)
    POST /predict   predicted salary and likely range for a job
    GET  /skills    top skills and typical pay for the role family of a title
"""
import json
import pickle
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.features import core_title, model_frame, role_family, seniority

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "salary_model.pkl"
META_PATH = APP_DIR / "model" / "model_meta.json"
SKILLS_PATH = APP_DIR / "data" / "skills.json"

EmploymentType = Literal["Full Time", "Part Time", "Contract", "Internship", "Temporary"]


# ── Data loading (once per process) ──────────────────────────────────────────
@lru_cache(maxsize=1)
def load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)["model"]


@lru_cache(maxsize=1)
def load_meta() -> dict:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_skills() -> dict:
    return json.loads(SKILLS_PATH.read_text(encoding="utf-8"))


# ── App ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load everything at startup so the first request is fast and a broken file fails loudly."""
    load_model(), load_meta(), load_skills(), title_patterns()
    yield


app = FastAPI(title="Worth It API", version="0.1.0", lifespan=lifespan,
              description="Salary estimates and in-demand skills from 394,300 job postings.")

# Public, read-only API with no logins or cookies, so any origin may call it.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])


# ── Request / response shapes ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    title: str = Field(..., min_length=2, max_length=120, examples=["Senior Data Engineer"])
    city: str | None = Field(None, max_length=80, examples=["New York"])
    country: str | None = Field(None, max_length=80, examples=["United States"])
    remote: bool = False
    employment_type: EmploymentType = "Full Time"
    skills: list[str] = Field(default_factory=list, max_length=30, examples=[["python", "sql", "aws"]])

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title must not be blank")
        return value.strip()

    @field_validator("skills")
    @classmethod
    def clean_skills(cls, value: list[str]) -> list[str]:
        # The model's skill vocabulary is lower-case, one token per skill.
        return sorted({s.strip().lower() for s in value if s and s.strip()})


class PredictResponse(BaseModel):
    title: str
    role_family: str
    seniority: str
    predicted_salary: int
    low: int
    high: int
    currency: str = "USD"
    range_hit_rate: float = Field(..., description="Share of test postings whose real salary fell inside the range")
    note: str


class SkillItem(BaseModel):
    skill: str
    share: float
    median_salary: int | None


class SkillsResponse(BaseModel):
    title: str
    role_family: str
    seniority: str
    matched_on: Literal["title", "role_family"] = Field(
        ..., description="'title' when a common job name inside the title was found, else the broad role family")
    matched_name: str = Field(..., description="The job name or role family the numbers describe")
    level_used: str
    postings: int
    median_salary: int | None
    skill_coverage: float = Field(..., description="Share of these postings that list any skill")
    top_skills: list[SkillItem]
    note: str | None = None


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": load_meta()["model"]}


@app.get("/options")
def options() -> dict:
    return load_skills()["options"]


def round_to(value: float, step: int = 500) -> int:
    return int(round(value / step) * step)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    meta = load_meta()
    country = req.country
    if req.city:  # a known city decides the country, like the dashboard does
        country = meta["city_country"].get(req.city, country)

    row = pd.DataFrame([{
        "title": req.title, "company": None, "industry": None, "department": None,
        "country": country, "city": req.city, "employment_type": req.employment_type,
        "remote": req.remote, "skills": req.skills,
    }])
    pred_log = float(load_model().predict(model_frame(row))[0])
    if not np.isfinite(pred_log):
        raise HTTPException(status_code=500, detail="The model could not score this job.")

    interval = meta["interval"]
    return PredictResponse(
        title=req.title,
        role_family=role_family(req.title),
        seniority=seniority(req.title),
        predicted_salary=round_to(np.exp(pred_log)),
        low=round_to(np.exp(pred_log + interval["log_low"])),
        high=round_to(np.exp(pred_log + interval["log_high"])),
        range_hit_rate=meta["test"]["interval_coverage"],
        note="Estimate of the advertised annual salary, based on job postings that listed one "
             "(mostly US). It is not a guarantee of what any employer pays.",
    )


MIN_SKILL_COVERAGE = 0.5  # below this, too few postings list skills for the list to mean anything


@lru_cache(maxsize=1)
def title_patterns() -> list[tuple[str, re.Pattern]]:
    """Known job names, most specific (most words, then longest) first."""
    names = sorted(load_skills()["titles"], key=lambda n: (-len(n.split()), -len(n)))
    return [(name, re.compile(rf"\b{re.escape(name)}\b")) for name in names]


def match_title(title: str) -> str | None:
    """The most specific known job name contained in the title ("data analyst" in "Sr Data Analyst, Marketing")."""
    core = core_title(title)
    return next((name for name, pattern in title_patterns() if pattern.search(core)), None)


@app.get("/skills", response_model=SkillsResponse)
def skills(title: str = Query(..., min_length=2, max_length=120)) -> SkillsResponse:
    title = title.strip()
    if len(title) < 2:
        raise HTTPException(status_code=422, detail="title must have at least 2 characters")
    family, level = role_family(title), seniority(title)
    data_file = load_skills()

    name = match_title(title)
    if name:
        data, matched_on, matched_name, level_used = data_file["titles"][name], "title", name, "All levels"
    elif family in data_file["families"]:
        entry = data_file["families"][family]
        by_level = entry["by_seniority"].get(level)
        data, level_used = (by_level, level) if by_level else (entry, "All levels")
        matched_on, matched_name = "role_family", family
    else:
        raise HTTPException(status_code=404, detail=f"Not enough postings for the '{family}' role family.")

    reliable = data["skill_coverage"] >= MIN_SKILL_COVERAGE
    return SkillsResponse(
        title=title, role_family=family, seniority=level, matched_on=matched_on, matched_name=matched_name,
        level_used=level_used, postings=data["postings"], median_salary=data["median_salary"],
        skill_coverage=data["skill_coverage"], top_skills=data["top_skills"] if reliable else [],
        note=None if reliable else "Few postings for this job list skills, so no skill ranking is shown.")
