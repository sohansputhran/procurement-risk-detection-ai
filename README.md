# Procurement Risk Detection AI

Cloud-native, investigator-friendly analytics to surface integrity risks in public procurement using **public data** and **Google Gemini** for structured extraction.

- **API:** FastAPI (`/health`, `/v1/score`, `/v1/extract/adverse-media`, `/v1/score/batch`)
- **UI:** Streamlit demo
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

### Health
```
GET /health
→ { "status": "ok" }
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

→ {
  "items": [
    {
      "entity": "ACME Contractors",
      "allegation_type": "bid-rigging",
      "date": "2023-08-14",
      "location": "Lagos",
      "source_url": "https://example.com/news/acme-riverbridge-collusion",
      "confidence": 0.82,
      "snippet": "…"
    }
  ]
}
```

### Batch scoring (features merge + top-factor attributions)
Compute scores for many awards at once by merging prebuilt features.

**Build features first** (see *Features* section), then:

**PowerShell (recommended):**
```powershell
$body = @(
  @{ award_id = "A1" }
  @{ award_id = "A2" }
) | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/v1/score/batch" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

**curl.exe (Windows) with a file:**
```powershell
@"
[
  {"award_id":"A1"},
  {"award_id":"A2"}
]
"@ | Set-Content -NoNewline -Path payload.json

curl.exe -s -X POST "http://127.0.0.1:8000/v1/score/batch" `
  -H "Content-Type: application/json" `
  --data-binary "@payload.json"
```

**Swagger UI:** open `http://127.0.0.1:8000/docs` → `POST /v1/score/batch`.

Environment overrides:
```
FEATURES_PATH=data/feature_store/contracts_features.parquet
GRAPH_METRICS_PATH=data/graph/metrics.parquet  # optional
```

Response shape (example):
```json
[
  {
    "award_id": "A1",
    "risk_score": 0.73,
    "top_factors": {
      "award_concentration_by_buyer": 0.30,
      "amount_zscore_by_category_norm": 0.20,
      "adjacency_to_sanctioned": 0.15
    },
    "provenance": { "features": "data/feature_store/contracts_features.parquet", "ts": "..." },
    "warnings": null
  }
]
```

---

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
        "response_schema": AdverseMediaPayload,  # strict schema
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

**Outputs**
- `data/raw/worldbank/projects_*.jsonl`
- `data/curated/worldbank/projects.parquet`

---

### World Bank — Ineligible (Sanctions) → JSONL + Parquet
Use the official **Excel** (download locally) for reliable parsing.

```bash
# Windows example path
python -m procurement_risk_detection_ai.pipelines.ingestion.wb_ineligible   --xlsx-file "C:\Users\you\Downloads\Listing of Ineligible Firms and Individuals.xlsx"   --out-dir data
```

**Outputs**
- `data/raw/worldbank/ineligible_*.jsonl`
- `data/curated/worldbank/ineligible.parquet`

---

### OCDS — Release Packages / JSONL feeds → Parquet
The loader supports **`.json` / `.json.gz`** (single package) **and** **`.jsonl` / `.jsonl.gz`** (line‑delimited releases).

#### From URL (JSONL.GZ — large feeds; use --take to sample)
```bash
# Print a sample (does not write)
python -m procurement_risk_detection_ai.pipelines.ingestion.ocds_loader   --url "https://data.open-contracting.org/en/publication/155/download?name=full.jsonl.gz"   --print-only --take 200

# Write curated parquet tables
python -m procurement_risk_detection_ai.pipelines.ingestion.ocds_loader   --url "https://data.open-contracting.org/en/publication/155/download?name=full.jsonl.gz"   --out-dir data
```

#### From local file
```bash
python -m procurement_risk_detection_ai.pipelines.ingestion.ocds_loader   --path data/raw/ocds/releases.jsonl.gz   --out-dir data
```

**Outputs**
- `data/curated/ocds/tenders.parquet`
- `data/curated/ocds/awards.parquet`
- `data/curated/ocds/suppliers.parquet`
- `data/raw/ocds/ocds_counts_*.json` (counts snapshot)

---

## Features — Contract-level feature seeds

Build features from curated OCDS outputs for use by `/v1/score/batch`.

```bash
python -m procurement_risk_detection_ai.pipelines.features.contracts_features   --tenders data/curated/ocds/tenders.parquet   --awards  data/curated/ocds/awards.parquet   --out     data/feature_store/contracts_features.parquet
```

**Outputs**
- `data/feature_store/contracts_features.parquet`

---

## Project structure

```
.github/workflows/                # CI
app/ui/                           # Streamlit demo
data/                             # raw/ & curated/ outputs (git-ignored except .gitkeep)
docs/                             # architecture, notes
llm/schemas/                      # JSON/Pydantic schemas
src/procurement_risk_detection_ai/
  app/api/                        # FastAPI app
  llm/                            # Gemini wrapper (structured output)
  pipelines/ingestion/            # public-data ingesters (WB/OCDS/GDELT)
  pipelines/features/             # feature builders
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
pip install pytest
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
  Feeds like `*.jsonl.gz` are line‑delimited. The loader detects JSONL/JSONL.GZ and wraps lines into a releases package. Use `--take` while testing.

- **422 Unprocessable Entity on batch**
  Ensure the POST body is a JSON **array** (see examples above). In PowerShell, prefer `Invoke-RestMethod` or `curl.exe --data-binary @file.json`.

- **500 / NaN or Infinity in JSON**
  The API sanitizes numbers, but if you derive your own scores ensure you don’t return NaN/Inf (JSON‑invalid).

- **Parquet engine error**
  Install `pyarrow`.

- **Excel read error**
  Install `openpyxl` and ensure you used `--xlsx-file` with a real `.xlsx` path.

---

## Roadmap (Agile)

- **Sprint 1 – Ingestion & Feature Seeds**
  - World Bank Projects ingest → Parquet ✅
  - World Bank Ineligible (sanctions) ingest ✅
  - OCDS ingest (Release Package & JSONL/GZ) ✅
  - GDELT DOC fetcher
  - Seed contract features (award concentration, repeat‑winner, near‑threshold, z‑scores) ✅

- **Sprint 2 – Graph & Model**
  - NetworkX prototype (buyer–award–supplier)
  - Centralities + distance‑to‑sanctioned
  - Replace heuristic with an ML model + basic explanations

- **Sprint 3 – App & Responsible AI**
  - Batch scoring endpoint + dataset dashboard ✅
  - Model card, data sheet, provenance logging
  - Responsible AI controls & prompts for LLM extraction

---

## Environment variables

Create `.env` (see `.env.example`):

```
API_URL=http://127.0.0.1:8000
GEMINI_API_KEY=your_key_here
FEATURES_PATH=data/feature_store/contracts_features.parquet
GRAPH_METRICS_PATH=data/graph/metrics.parquet
```

---

## License

MIT © 2025 Sohan Puthran
