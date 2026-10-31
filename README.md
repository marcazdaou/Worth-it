# Worth It

A mobile app that tells you what a job pays and which skills it asks for, powered by the
salary model and data from [Job Market Analytics](https://github.com/marcazdaou/Job-Market-Analysis)
(394,300 job postings from 65 countries).

> **Status:** week 1 of 4. The backend API is done; the mobile app (React Native + Expo) comes next.

## How it works

```
Mobile app (Expo)  ──►  Worth It API (FastAPI, this repo)  ──►  salary model + skills data
                                                                 (copied from Job-Market-Analysis)
```

The API only **reads** a copy of the trained model; it never retrains or touches the original
project, so Job Market Analytics and its live dashboard keep running exactly as before.

## API

| Endpoint | What it returns |
|---|---|
| `GET /health` | Whether the server is up and which model is loaded |
| `GET /options` | Dropdown values for the app: cities, countries, skills, employment types |
| `POST /predict` | Predicted salary and likely range for a job |
| `GET /skills?title=...` | Top skills and typical pay for the title's role family and level |

Interactive docs are served at `/docs` once the server is running.

**Example**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Data Analyst", "city": "Boston", "skills": ["sql", "python", "tableau"]}'
```

```json
{
  "title": "Data Analyst",
  "role_family": "Data & AI",
  "seniority": "Mid / Unspecified",
  "predicted_salary": 111000,
  "low": 87500,
  "high": 140000,
  "currency": "USD",
  "range_hit_rate": 0.809,
  "note": "Estimate of the advertised annual salary, based on job postings that listed one (mostly US). It is not a guarantee of what any employer pays."
}
```

The range is the model's 80% interval: in testing, the real salary fell inside it 81% of the time.

## Run it locally

Requires Python 3.11+.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                          # 17 tests
uvicorn app.main:app --reload      # http://localhost:8000/docs
```

## Updating the model or data

The model files in `backend/app/model/` and the title rules in `backend/app/features.py` are copies
from Job-Market-Analysis and must stay in sync. After retraining there:

1. Copy `models/salary_model.pkl` and `models/model_meta.json` into `backend/app/model/`.
2. Copy any changed rules from `pipeline/features.py` into `backend/app/features.py`.
3. Rebuild the skills data: `python scripts/build_skills.py path/to/jobs_clean.parquet`
4. Keep `numpy`, `pandas` and `scikit-learn` pinned to the versions the model was trained with.

## Deploy

`render.yaml` deploys the API to Render's free plan (the server sleeps when idle, so the first
request after a while takes about a minute).

---

**Marc Abi Zeid Daou**, Data Analytics Engineering, Northeastern University ·
[Portfolio](https://marcazdaou.github.io/my-portfolio/) · [LinkedIn](https://www.linkedin.com/in/marc-abi-zeid-daou/)
