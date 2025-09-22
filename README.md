# Procurement Risk Detection AI

Cloud-native, investigator-friendly analytics to surface integrity risks in public procurement using **public data** and **Google Gemini** for structured extraction.

- **API:** FastAPI (`/health`, `/v1/score`, `/v1/extract/adverse-media`, `/v1/score/batch`)
- **UI:** Streamlit demo (includes **Datasets & Batch Scoring** page with CSV upload/download)
- **ML Baseline:** Logistic Regression with SHAP-style linear contributions
- **LLM:** Google GenAI SDK (Gemini) with strict, schema-validated JSON output
- **Layout:** `src/` package (editable install)
- **OS:** Windows-friendly commands throughout

---

## What's new (Sep 2025)

- **Batch endpoint shape fixed:** `POST /v1/score/batch` now mirrors the input shape.
  - **List in → List out** (no wrapper keys) — satisfies `tests/test_api_batch.py::test_batch_scores_with_graph` (`len(response) == len(input)`).
  - **Envelope in → Envelope out** (returns `{ "items": [...], "provenance_id": "...",
    "used_model": true }`).
  - Enforces **1:1 outputs per input row** even if joins fan out (uses `_input_row_id` + `groupby(...).first()`).
  - Optional `?join_graph=true` to attach graph metrics when available.
- **Baseline model fix:** `fit_baseline(...)` now **returns** the fitted `LogisticRegression`, fixing
  `tests/test_model_explanations.py` (`isinstance(model, LogisticRegression)`).
- **Training CLI hardened:** `models/train_baseline.py`
  - `--features-cols auto` to select present columns from `FEATURE_COLS_DEFAULT`.
  - Robust Parquet IO (pyarrow/fastparquet) with CSV fallback + clearer errors.
  - Prints proxy-label balance, sample features, and **top weights** after training.
  - Windows-friendly and exits with helpful codes on failure.
- **Streamlit UX:** Batch page supports CSV upload with `award_id` and **downloads** results as CSV.
- **Provenance:** Batch scoring logs provenance (endpoint, counts, durations) and returns `provenance_id` in the envelope form.

---

## Quickstart

```bash
# 1) Create & activate a virtual env
python -m venv .venv && .venv\Scripts\activate     # Windows

# 2) Install runtime deps
pip install -r requirements.txt

# 3) Install the package (src-layout)
pip install -e .

# 4) Environment
copy .env.example .env   # Windows

# Set your Gemini key:
#   GEMINI_API_KEY=YOUR_KEY

# 5) Run API & UI
python -m uvicorn procurement_risk_detection_ai.app.api.main:app --reload --port 8000
python -m streamlit run app/ui/streamlit_app.py
```

### Windows notes
PowerShell aliases `curl` to `Invoke-WebRequest`. For JSON POSTs, either:
- Use **PowerShell-native** (`Invoke-RestMethod`) or
- Call **`curl.exe`** explicitly.

---

## Endpoints

### Health
```
GET /health
```
Returns API version and row counts/availability for configured datasets.

Environment overrides (optional):
```
FEATURES_PATH=data/feature_store/contracts_features.parquet
GRAPH_METRICS_PATH=data/graph/metrics.parquet
WB_INELIGIBLE_PATH=data/curated/worldbank/ineligible.parquet
OCDS_TENDERS_PATH=data/curated/ocds/tenders.parquet
OCDS_AWARDS_PATH=data/curated/ocds/awards.parquet
```

### Risk score (simple demo)
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

### Batch scoring (features merge + optional graph + ML explanations)

**Prerequisites**
1) **Build features** (must include `supplier_id` for graph join):
```bash
python -m procurement_risk_detection_ai.pipelines.features.contracts_features ^
  --tenders data/curated/ocds/tenders.parquet ^
  --awards  data/curated/ocds/awards.parquet ^
  --out     data/feature_store/contracts_features.parquet
```
2) *(Optional)* **Build graph metrics**:
```bash
python -m procurement_risk_detection_ai.graph.graph_utils ^
  --tenders  data/curated/ocds/tenders.parquet ^
  --awards   data/curated/ocds/awards.parquet ^
  --sanctions data/curated/worldbank/ineligible.parquet ^
  --out-dir  data
```

**List-in → List-out example**
```bash
curl.exe -s -X POST "http://127.0.0.1:8000/v1/score/batch" ^
  -H "Content-Type: application/json" ^
  -d "[{\"award_id\":\"A1\"},{\"award_id\":\"A2\"}]"
# → returns a JSON array with 2 items
```

**Envelope-in → Envelope-out example**
```bash
curl.exe -s -X POST "http://127.0.0.1:8000/v1/score/batch?join_graph=true" ^
  -H "Content-Type: application/json" ^
  -d "{\"items\":[{\"award_id\":\"A1\"},{\"award_id\":\"A2\"}]}"
# → returns: { "items": [...], "provenance_id": "...", "used_model": true }
```

**Swagger UI:** open `http://127.0.0.1:8000/docs` → `POST /v1/score/batch`.

---

## Streamlit — Datasets & Batch Scoring page

Path: `app/ui/pages/1_Datasets.py`

- Shows **dataset availability** and **row counts** by calling `/health`.
- Lets you **upload a CSV** with an `award_id` column to call `/v1/score/batch` and view results
  (plus a bar chart of highest risk). Supports **download** of scored results as CSV.

Set the API base URL for the UI:
```
API_URL=http://127.0.0.1:8000
```

Run the UI:
```bash
python -m streamlit run app/ui/streamlit_app.py
```

---

## Baseline ML model (Logistic Regression) + SHAP-style explanations

Train on curated `contracts_features.parquet`. Until labeled outcomes are available,
the training script derives a **proxy label** from feature heuristics
(`near_threshold_flag`, high `amount_zscore_by_category`, high `repeat_winner_ratio`,
`award_concentration_by_buyer`).

**Train the baseline**
```bash

# Train baseline (saves artifacts to models/)
python -m procurement_risk_detection_ai.models.train_baseline --features data/feature_store/contracts_features.parquet --features-cols auto
```

**How batch scoring uses the model**
- If `models/baseline_logreg.joblib` exists, `POST /v1/score/batch` will use it to compute:
  - `risk_score` in [0,1]
  - `top_factors` — linear logit contributions per feature (sorted by |contribution|; SHAP-style)
- If no model is found, the endpoint **falls back** to intrinsic heuristic rules.

**Response (list-in excerpt)**
```jsonc
[
  {
    "award_id": "A1",
    "supplier_id": "S1",
    "risk_score": 0.78,
    "top_factors": [
      {"name": "repeat_winner_ratio", "value": 0.92, "contribution": 1.15},
      {"name": "amount_zscore_by_category", "value": 2.7, "contribution": 0.84}
    ]
  }
]
```

> Tip: set `FEATURES_PATH` (and optionally `GRAPH_METRICS_PATH`) so batch can merge features (and graph metrics via `?join_graph=true`).

---

## Public data ingestion

> Ensure Parquet/Excel engines as needed:
>
> ```bash
> pip install pyarrow openpyxl
> ```

### World Bank — Projects → JSONL + Parquet
```bash
python -m procurement_risk_detection_ai.pipelines.ingestion.wb_projects ^
  --rows 500 --max-pages 40 --out-dir data
```

### World Bank — Ineligible (Sanctions) → JSONL + Parquet
Use the official **Excel** (download locally) for reliable parsing.
```bash
python -m procurement_risk_detection_ai.pipelines.ingestion.wb_ineligible ^
  --xlsx-file "C:\Users\you\Downloads\Listing of Ineligible Firms and Individuals.xlsx" ^
  --out-dir data
```

### OCDS — Release Packages / JSONL feeds → Parquet
From URL (JSONL.GZ — use `--take` to sample):
```bash
python -m procurement_risk_detection_ai.pipelines.ingestion.ocds_loader ^
  --url "https://data.open-contracting.org/en/publication/155/download?name=full.jsonl.gz" ^
  --print-only --take 200
python -m procurement_risk_detection_ai.pipelines.ingestion.ocds_loader ^
  --url "https://data.open-contracting.org/en/publication/155/download?name=full.jsonl.gz" ^
  --out-dir data
```
From local file:
```bash
python -m procurement_risk_detection_ai.pipelines.ingestion.ocds_loader ^
  --path data/raw/ocds/releases.jsonl.gz ^
  --out-dir data
```

**Curated outputs**
- `data/curated/ocds/tenders.parquet`
- `data/curated/ocds/awards.parquet`
- `data/curated/ocds/suppliers.parquet`

---

## Features — Contract-level feature seeds

```bash
python -m procurement_risk_detection_ai.pipelines.features.contracts_features ^
  --tenders data/curated/ocds/tenders.parquet ^
  --awards  data/curated/ocds/awards.parquet ^
  --out     data/feature_store/contracts_features.parquet
```
Outputs:
- `data/feature_store/contracts_features.parquet`
  - `award_id`, **`supplier_id`**, `award_concentration_by_buyer`, `repeat_winner_ratio`,
    `amount_zscore_by_category`, `near_threshold_flag`, `time_to_award_days`

---

## Graph metrics — buyers ↔ suppliers + distance-to-sanctioned

```bash
python -m procurement_risk_detection_ai.graph.graph_utils ^
  --tenders  data/curated/ocds/tenders.parquet ^
  --awards   data/curated/ocds/awards.parquet ^
  --sanctions data/curated/worldbank/ineligible.parquet ^
  --out-dir  data ^
  --ego-supplier-id S1           # optional (saves an ego PNG)

# For faster runs on big graphs, skip betweenness with:
#   --no-betweenness
```
Outputs:
- `data/graph/metrics.parquet` (`supplier_id`, `supplier_name`, `degree`, `betweenness`, `distance_to_sanctioned`)
- `data/graph/ego_<supplier_id>.png` (optional)

If you see a NetworkX “multi_source_shortest_path_length” error, upgrade:
```
pip install --upgrade "networkx>=3.0"
```

---

## Tests

```bash
pytest -q
# Key checks:
# - tests/test_api_batch.py::test_batch_scores_with_graph  ✅ list-in → list-out, len matches
# - tests/test_model_explanations.py::test_predict_proba_and_contrib_linear_explanations ✅ model + contributions
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
  app/services/                   # scoring, provenance
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

# Editable install for tests & CLI
pip install -e .
```

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
