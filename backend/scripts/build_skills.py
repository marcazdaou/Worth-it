"""Pre-compute the skills and dropdown data the API serves.

The cleaned postings file is 11 MB and needs pandas/pyarrow to read, so instead of
shipping it, this script turns it into one small JSON file (app/data/skills.json).
Run it again whenever Job-Market-Analysis produces a new jobs_clean.parquet:

    python scripts/build_skills.py path/to/Job-Market-Analysis/data/jobs_clean.parquet
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
from app.features import core_title  # noqa: E402  (same rule the API uses at request time)

OUT = BACKEND / "app" / "data" / "skills.json"
TOP_SKILLS = 15
MIN_GROUP = 50          # smallest group (postings) we report skills for
MIN_SALARIED = 20       # smallest number of salaries we report a median for
MIN_OPTION = 20         # a city/country needs this many salaried postings to be offered
MIN_TITLE_LENGTH = 4    # skip tiny job names such as "ai" or "it" that match too much


def median_or_none(values: pd.Series) -> int | None:
    values = values.dropna()
    return int(round(values.median(), -2)) if len(values) >= MIN_SALARIED else None


def summarize(group: pd.DataFrame) -> dict:
    """Top skills for one group of postings, with how often each appears and what it pays."""
    long = group[["skills", "salary_mid"]].explode("skills").dropna(subset=["skills"])
    counts = long["skills"].value_counts().head(TOP_SKILLS)
    top = []
    for skill, count in counts.items():
        top.append({
            "skill": skill,
            "share": round(count / len(group), 3),
            "median_salary": median_or_none(long.loc[long["skills"] == skill, "salary_mid"]),
        })
    return {
        "postings": int(len(group)),
        # Share of postings that list any skill. The skill tags are mostly technical,
        # so for many non-tech jobs this is low and the skill list is not meaningful.
        "skill_coverage": round(float((group["skills"].map(len) > 0).mean()), 3),
        "salaried": int(group["salary_mid"].notna().sum()),
        "median_salary": median_or_none(group["salary_mid"]),
        "top_skills": top,
    }


def build_titles(jobs: pd.DataFrame) -> dict:
    """Skills per common job name, counting every posting whose title contains it.

    "data analyst" covers "Senior Data Analyst" and "Data Analyst, Marketing" too, so
    each common name gets enough postings to say something reliable.
    """
    counts = jobs["core"].value_counts()
    names = [n for n, c in counts.items() if c >= MIN_GROUP and len(n) >= MIN_TITLE_LENGTH]
    titles = {}
    for name in names:
        group = jobs[jobs["core"].str.contains(rf"\b{re.escape(name)}\b", regex=True)]
        if len(group) >= MIN_GROUP:
            titles[name] = summarize(group)
    return titles


def build(parquet: Path) -> dict:
    cols = ["title", "role_family", "seniority", "skills", "salary_mid", "city", "country", "employment_type"]
    jobs = pd.read_parquet(parquet, columns=cols)
    for col in ["role_family", "seniority", "employment_type", "country"]:
        jobs[col] = jobs[col].astype("object")
    jobs["core"] = jobs["title"].map(core_title)

    families = {}
    for family, group in jobs.groupby("role_family"):
        if len(group) < MIN_GROUP:
            continue
        entry = summarize(group)
        entry["by_seniority"] = {level: summarize(sub) for level, sub in group.groupby("seniority")
                                 if len(sub) >= MIN_GROUP}
        families[family] = entry

    salaried = jobs[jobs["salary_mid"].notna()]
    city_counts = salaried["city"].value_counts()
    country_counts = salaried["country"].value_counts()
    skill_counts = jobs["skills"].explode().value_counts()
    return {
        "source": "Job-Market-Analysis/data/jobs_clean.parquet",
        "total_postings": int(len(jobs)),
        "families": families,
        "titles": build_titles(jobs),
        "options": {
            "cities": sorted(city_counts[city_counts >= MIN_OPTION].index.tolist()),
            "countries": sorted(country_counts[country_counts >= MIN_OPTION].index.tolist()),
            "skills": sorted(skill_counts[skill_counts >= MIN_GROUP].index.tolist()),
            "employment_types": ["Full Time", "Part Time", "Contract", "Internship", "Temporary"],
        },
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/build_skills.py path/to/jobs_clean.parquet")
    data = build(Path(sys.argv[1]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(data['families'])} role families, "
          f"{len(data['titles'])} job titles)")
