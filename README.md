# Procurement Risk Detection AI

Cloud-native, investigator-friendly analytics to surface integrity risks in public procurement using **public data** and **Google Gemini** for structured extraction.

- **API:** FastAPI (`/health`, `/v1/score`, `/v1/extract/adverse-media`, `/v1/score/batch`)
- **UI:** Streamlit demo (now includes a **Datasets & Batch Scoring** page)
- **LLM:** Google GenAI SDK (Gemini) with strict, schema-validated JSON output
- **Layout:** `src/` package (editable install)

---

## Quickstart

```bash
# 1) Create & activate a virtual env
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2) Install runtime deps
pip install -r requirements.txt

# 3) Install the package (src-layout)
pip install -e .

# 4) Environment
cp .env.example .env
# Set your Gemini key (get it from Google AI Studio)
# export GEMINI_API_KEY=YOUR_KEY

# 5) Run API & UI
python -m uvicorn procurement_risk_detection_ai.app.api.main:app --reload --port 8000
python -m streamlit run app/ui/streamlit_app.py
```

### Windows notes
PowerShell aliases `curl` to `Invoke-WebRequest`. For JSON POSTs, either:
- Use **PowerShell-native** (`Invoke-RestMethod`) or
- Call **`curl.exe`** explicitly (see examples below).

---

## Endpoints

### Health (datasets probe)
```
GET /health
```
Returns API version and row counts/availability for configured datasets:
```json
{
  "status": "ok",
  "api_version": "…",
  "datasets": {
    "features":        {"path": ".../contracts_features.parquet", "rows": 12345, "available": true},
    "graph_metrics":   {"path": ".../metrics.parquet",            "rows": 6789,  "available": true},
    "wb_ineligible":   {"path": ".../ineligible.parquet",         "rows": 321,   "available": true},
    "ocds_tenders":    {"path": ".../tenders.parquet",            "rows": 1000,  "available": true},
    "ocds_awards":     {"path": ".../awards.parquet",             "rows": 1000,  "available": true}
  }
}
```

Environment overrides (optional):
```
FEATURES_PATH=data/feature_store/contracts_features.parquet
GRAPH_METRICS_PATH=data/graph/metrics.parquet
WB_INELIGIBLE_PATH=data/curated/worldbank/ineligible.parquet
OCDS_TENDERS_PATH=data/curated/ocds/tenders.parquet
OCDS_AWARDS_PATH=data/curated/ocds/awards.parquet
```

### Risk score (demo heuristic)
```
POST /v1/score
Content-Type: application/json
{
  "amount": 100000,
  "past_awards_count": 3,
  "is_sanctioned": false,
  "adverse_media_count": 1
}
→ { "risk_score": 0.42, "notes": "demo-only heuristic" }
```

### Adverse media extraction (Gemini → structured JSON)
```
POST /v1/extract/adverse-media
Content-Type: application/json
{ "text": "…paste a news paragraph or snippet…" }
```

### Batch scoring (features merge + ML explanations)
Compute scores for many awards at once by merging prebuilt features and (optionally) **graph metrics**.

**Prerequisites**
1) **Build features** (includes `supplier_id` for graph join):
```bash
python -m procurement_risk_detection_ai.pipelines.features.contracts_features   --tenders data/curated/ocds/tenders.parquet   --awards  data/curated/ocds/awards.parquet   --out     data/feature_store/contracts_features.parquet
```
2) *(Optional)* **Build graph metrics**:
```bash
python -m procurement_risk_detection_ai.graph.graph_utils   --tenders  data/curated/ocds/tenders.parquet   --awards   data/curated/ocds/awards.parquet   --sanctions data/curated/worldbank/ineligible.parquet   --out-dir  data
```

**Swagger UI:** open `http://127.0.0.1:8000/docs` → `POST /v1/score/batch`.

---

## Streamlit — Datasets & Batch Scoring page

Path: `app/ui/pages/1_Datasets.py`

- Shows **dataset availability** and **row counts** by calling `/health`.
- Lets you **upload a CSV** with an `award_id` column to call `/v1/score/batch` and view results (plus a quick bar chart of top risks).

Set the API base URL for the UI:
```
API_URL=http://127.0.0.1:8000
```

Run the UI:
```bash
python -m streamlit run app/ui/streamlit_app.py
```

---


---

## Baseline ML model (Logistic Regression) + SHAP-style explanations

You can now replace the demo heuristic with a simple ML baseline trained on your curated
`contracts_features.parquet`. Until labeled outcomes are available, the training script derives
a **proxy label** from feature heuristics (`near_threshold_flag`, high `amount_zscore_by_category`,
high `repeat_winner_ratio`, and `award_concentration_by_buyer`).

**Train the baseline**
```bash
# Train baseline (saves to models/)
python -m procurement_risk_detection_ai.models.train_baseline --features data/feature_store/contracts_features.parquet
```

**How batch scoring uses the model**
- If `models/baseline_logreg.joblib` exists, `POST /v1/score/batch` will use it to compute:
  - `risk_score` in [0,1]
  - `top_factors` — linear logit contributions per feature (sorted by absolute magnitude; SHAP-style)
- If no model is found, the endpoint **falls back** to the previous heuristic rules.

**Response shape (excerpt)**
```json
{
  "items": [
    {
      "award_id": "A1",
      "supplier_id": "S1",
      "risk_score": 0.78,
      "top_factors": [
        {"name": "repeat_winner_ratio", "value": 0.92, "contribution": 1.15},
        {"name": "amount_zscore_by_category", "value": 2.7, "contribution": 0.84}
      ]
    }
  ],
  "provenance_id": "abc123...",
  "used_model": true
}
```

> Tip: set `FEATURES_PATH` (and optionally `GRAPH_METRICS_PATH`) so the batch endpoint can merge features (and graph metrics via `?join_graph=true`).


## LLMs: Google Gemini (public-only)

- SDK: `google-genai`
- Key: `GEMINI_API_KEY` (environment variable)

Minimal example (already wired in the API):
```python
from google import genai
from procurement_risk_detection_ai.llm.gemini_client import AdverseMediaPayload

client = genai.Client()  # reads GEMINI_API_KEY
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Extract adverse media for ACME bribery at http://example.com",
    config={
        "response_mime_type": "application/json",
        "response_schema": AdverseMediaPayload,
    },
)
print(resp.parsed or resp.text)
```

> Tip: if `GEMINI_API_KEY` is unset, LLM tests are auto-skipped.

---

## Public data ingestion

> Ensure Parquet/Excel engines as needed:
>
> ```bash
> pip install pyarrow openpyxl
> ```

### World Bank — Projects → JSONL + Parquet
```bash
python -m procurement_risk_detection_ai.pipelines.ingestion.wb_projects   --rows 500 --max-pages 40 --out-dir data
```

### World Bank — Ineligible (Sanctions) → JSONL + Parquet
Use the official **Excel** (download locally) for reliable parsing.
```bash
python -m procurement_risk_detection_ai.pipelines.ingestion.wb_ineligible   --xlsx-file "C:\Users\you\Downloads\Listing of Ineligible Firms and Individuals.xlsx"   --out-dir data
```

### OCDS — Release Packages / JSONL feeds → Parquet
From URL (JSONL.GZ — use `--take` to sample):
```bash
python -m procurement_risk_detection_ai.pipelines.ingestion.ocds_loader   --url "https://data.open-contracting.org/en/publication/155/download?name=full.jsonl.gz"   --print-only --take 200
python -m procurement_risk_detection_ai.pipelines.ingestion.ocds_loader   --url "https://data.open-contracting.org/en/publication/155/download?name=full.jsonl.gz"   --out-dir data
```
From local file:
```bash
python -m procurement_risk_detection_ai.pipelines.ingestion.ocds_loader   --path data/raw/ocds/releases.jsonl.gz   --out-dir data
```

**Curated outputs**
- `data/curated/ocds/tenders.parquet`
- `data/curated/ocds/awards.parquet`
- `data/curated/ocds/suppliers.parquet`

---

## Features — Contract-level feature seeds

```bash
python -m procurement_risk_detection_ai.pipelines.features.contracts_features   --tenders data/curated/ocds/tenders.parquet   --awards  data/curated/ocds/awards.parquet   --out     data/feature_store/contracts_features.parquet
```
Outputs:
- `data/feature_store/contracts_features.parquet`
  - `award_id`, **`supplier_id`**, `award_concentration_by_buyer`, `repeat_winner_ratio`,
    `amount_zscore_by_category`, `near_threshold_flag`, `time_to_award_days`

---

## Graph metrics — buyers ↔ suppliers + distance-to-sanctioned

```bash
python -m procurement_risk_detection_ai.graph.graph_utils   --tenders  data/curated/ocds/tenders.parquet   --awards   data/curated/ocds/awards.parquet   --sanctions data/curated/worldbank/ineligible.parquet   --out-dir  data   --ego-supplier-id S1           # optional (saves an ego PNG)
# For faster runs on big graphs, skip betweenness with --no-betweenness
```

Outputs:
- `data/graph/metrics.parquet` (`supplier_id`, `supplier_name`, `degree`, `betweenness`, `distance_to_sanctioned`)
- `data/graph/ego_<supplier_id>.png` (optional)

If you see a NetworkX “multi_source_shortest_path_length” error, upgrade:
```
pip install --upgrade "networkx>=3.0"
```

---

## Project structure

```
.github/workflows/                # CI
app/ui/                           # Streamlit demo
  pages/                          # 1_Datasets.py
data/                             # raw/ & curated/ outputs (git-ignored except .gitkeep)
docs/                             # architecture, notes
llm/schemas/                      # JSON/Pydantic schemas
src/procurement_risk_detection_ai/
  app/api/                        # FastAPI app (health, batch, etc.)
  llm/                            # Gemini wrapper (structured output)
  pipelines/ingestion/            # public-data ingesters (WB/OCDS/GDELT)
  pipelines/features/             # feature builders
  graph/                          # buyer↔supplier graph + metrics
tests/                            # pytest tests
```

---

## Development

```bash
# Lint & format (install if not in requirements.txt)
pip install ruff black
ruff check .
black .

# Tests
pytest -q
```

## Troubleshooting

- **`ModuleNotFoundError: procurement_risk_detection_ai`**
  Run `pip install -e .`; ensure tests import via the package path (not `src.`).
  Add a `pytest.ini` if needed:
  ```ini
  [pytest]
  pythonpath =
      src
  ```

- **JSON/JSONL decode errors on OCDS URL**
  Feeds like `*.jsonl.gz` are line‑delimited. Use `--take` for sampling.

- **422/503 or JSON errors on batch**
  - Ensure the POST body is a JSON **array** (`[{ "award_id": "A1" }, …]`).
  - Rebuild features so `supplier_id` exists for graph joins.
  - Install `pyarrow` (Parquet) and `openpyxl` (Excel) as needed.

- **UI cannot reach API**
  Set `API_URL` (env or `.env`) to your running FastAPI base URL.

---

## Environment variables

Create `.env` (see `.env.example`):

```
API_URL=http://127.0.0.1:8000
GEMINI_API_KEY=your_key_here
FEATURES_PATH=data/feature_store/contracts_features.parquet
GRAPH_METRICS_PATH=data/graph/metrics.parquet
WB_INELIGIBLE_PATH=data/curated/worldbank/ineligible.parquet
OCDS_TENDERS_PATH=data/curated/ocds/tenders.parquet
OCDS_AWARDS_PATH=data/curated/ocds/awards.parquet
```

---

## License

MIT © 2025 Sohan Puthran
