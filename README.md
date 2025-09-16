# Procurement Risk Detection AI

Cloud-native, investigator-friendly analytics to surface integrity risks in public procurement.
This first commit provides a **runnable skeleton** with API + UI, basic risk scorer,
Dev tooling, and an Agile-friendly project layout.

## Quickstart

```bash
# 1) Create & activate a virtual env (example using Python venv)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the API (FastAPI on http://127.0.0.1:8000)
python -m uvicorn procurement_risk_detection_ai.app.api.main:app --reload --port 8000

# 4) Run the UI (Streamlit on http://localhost:8501)
python -m streamlit run app/ui/streamlit_app.py

# 5) Run Data Ingestion from World Bank
python -m src.procurement_risk_detection_ai.pipelines.ingestion.wb_projects --rows 500 --max-pages 40 --out-dir data
```

### Environment
Copy `.env.example` to `.env` and edit as needed:
```
API_URL=http://127.0.0.1:8000
GEMINI_API_KEY=your_gemini_api_key_here
```

## What’s in this commit

- **FastAPI microservice** with `/health` and `/v1/score` (toy risk scoring).
- **Streamlit UI** that calls the API and displays the risk score.
- **Project layout** for ingestion, features, graph, models, and LLM schemas.
- **Tooling**: `Makefile`, `pre-commit` (Black, Ruff), GitHub Actions CI, `.gitignore`.
- **Tests**: a simple unit test for the scorer.

## Architecture (minimal skeleton)

```
FastAPI (uvicorn)  <-- JSON requests -->  Risk Scorer (python)
         ^                                        |
         |                                        v
    Streamlit UI  -------------------------->  /v1/score
```

This will grow toward: ingestion (WB/OCDS/GDELT), feature pipelines, graph DB,
Databricks/MLflow, and responsible AI evaluators.

## API

- `GET /health` → `{ "status": "ok" }`
- `POST /v1/score`
  ```json
  {
    "amount": 100000,
    "past_awards_count": 3,
    "is_sanctioned": false,
    "adverse_media_count": 1
  }
  ```
  → `{ "risk_score": 0.42, "notes": "demo-only heuristic" }`

## License
MIT © 2025 Sohan Puthran
