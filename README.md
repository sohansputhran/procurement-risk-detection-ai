# Procurement Risk Detection AI

Cloud-native, investigator-friendly analytics to surface integrity risks in public procurement using **public data** and **Google Gemini** for structured extraction.
Stack: Python, FastAPI, Pydantic v1/v2, pandas/pyarrow, NetworkX, Streamlit, pytest. Repo layout uses `src/` (editable install).

---

## 🟢 Current Progress (what’s built)

### API (FastAPI)
- `GET /health` – dataset availability & row counts (paths configurable via env).
- `POST /v1/score` – simple heuristic demo.
- `POST /v1/score/batch` – contract-level risk scoring:
  - Accepts **list-in → list-out** or **envelope-in → envelope-out**.
  - Left-joins features from `FEATURES_PATH` (+ optional graph metrics with `?join_graph=true`).
  - Preserves **1:1** input→output using `_input_row_id`; input order preserved.
  - Explanations: `limit_top_factors` (1–20). **`explain=false`** returns scores/bands only (fast vectorized path).
  - Adds `risk_band` via thresholds saved at train time.
- `POST /v1/score/validate` – validation-only mirror (schema + existence in features).
- `GET /v1/model/info` – model availability, meta (feature list, weights summary, thresholds, latest evaluation).
- `POST /v1/extract/adverse-media` – Gemini → schema-validated JSON extraction.
- `GET /metrics` – Prometheus exposition (if `prometheus_client` installed).

### ML baseline
- Logistic Regression with **linear SHAP-style** contributions (coef × value on the logit).
- Meta file stores feature list and **risk band thresholds**.
- Optional **Platt calibration** during training when `CALIBRATE_PROBS=true` (saved calibrator is used automatically in vectorized path).

### Batch scoring behavior
- Vectorized probability prediction when `explain=false` (big performance win).
- Heuristic fallback when model is unavailable (keeps API usable).
- Per-item `error` field; still 1:1 in outputs.

### Streamlit UI
- Pages: **Model info** and **Datasets & Batch Scoring**.
- CSV upload with `award_id` (+ optional `supplier_id`).
- Toggles: **Join graph**, **Explain (top factors)**, **Top factors slider**.
- Shows errors, scores, factor previews, risk-band filter and downloadable CSV.

### Observability
- Prometheus counters/histograms: `api_requests_total`, `request_latency_seconds`, `batch_scored_items_total`.

### Tests (pytest)
- API: batch scoring, graph join, top-factor limits, explain flag, validate-only.
- Models: linear explanations, baseline evaluation CLI.

---

## 🚀 Quickstart

```powershell
# 1) Create & activate venv (Windows)
python -m venv .venv
.venv\Scripts\activate

# 2) Install
pip install -r requirements.txt
pip install -e .

# 3) Environment
copy .env.example .env   # set GEMINI_API_KEY and data paths

# 4) Train baseline
python -m procurement_risk_detection_ai.models.train_baseline --features data/feature_store/contracts_features.parquet --features-cols auto

# 5) Evaluate (writes reports/metrics/*.json)
python -m procurement_risk_detection_ai.models.evaluate_baseline --features data/feature_store/contracts_features.parquet --features-cols auto

# 6) Run API & UI
python -m uvicorn procurement_risk_detection_ai.app.api.main:app --reload --port 8000
python -m streamlit run app/ui/streamlit_app.py
```

**Notes**
- Use `curl.exe` instead of PowerShell’s `curl` alias for raw JSON calls.
- Parquet engine: `pyarrow` (or `fastparquet`).

---

## 🔧 Key Environment Variables

```
API_URL=http://127.0.0.1:8000
FEATURES_PATH=data/feature_store/contracts_features.parquet
GRAPH_METRICS_PATH=data/graph/metrics.parquet
WB_INELIGIBLE_PATH=data/curated/worldbank/ineligible.parquet
OCDS_TENDERS_PATH=data/curated/ocds/tenders.parquet
OCDS_AWARDS_PATH=data/curated/ocds/awards.parquet

# Models & metrics
MODELS_DIR=models_data
MODEL_PATH=models_data/baseline_logreg.joblib
MODEL_META_PATH=models_data/baseline_logreg_meta.json
MODEL_CALIBRATOR_PATH=models_data/baseline_calibrator.joblib
METRICS_DIR=reports/metrics

# API behavior
DEFAULT_TOP_K=5
MAX_BATCH_ITEMS=1000
MAX_REQUEST_BYTES=1048576
FEATURES_CACHE_DISABLE=false
```

---

## 📡 Endpoint Cheatsheet

- **Batch scoring**
  - List-in → List-out:
    ```powershell
    curl.exe -s -X POST "http://127.0.0.1:8000/v1/score/batch" `
      -H "Content-Type: application/json" `
      -d "[{\"award_id\":\"A1\"},{\"award_id\":\"A2\"}]"
    ```
  - Envelope-in (with graph + top 3 factors):
    ```powershell
    curl.exe -s -X POST "http://127.0.0.1:8000/v1/score/batch?join_graph=true&limit_top_factors=3" `
      -H "Content-Type: application/json" `
      -d "{\"items\":[{\"award_id\":\"A1\"},{\"award_id\":\"A2\"}]}"
    ```
  - **Faster** (skip explanations):
    ```powershell
    curl.exe -s -X POST "http://127.0.0.1:8000/v1/score/batch?explain=false" `
      -H "Content-Type: application/json" `
      -d "{\"items\":[{\"award_id\":\"A1\"},{\"award_id\":\"A2\"}]}"
    ```

- **Validate-only (no scoring)**
  ```powershell
  curl.exe -s -X POST "http://127.0.0.1:8000/v1/score/validate" `
    -H "Content-Type: application/json" `
    -d "{\"items\":[{\"award_id\":\"A1\"},{\"award_id\":\"A2\"}]}"
  ```

- **Metrics**
  ```powershell
  curl.exe -s "http://127.0.0.1:8000/metrics"
  ```

---

## 🧪 Running Tests

```powershell
pytest -q
# Key checks:
# - tests/test_api_batch.py::test_batch_scores_with_graph
# - tests/test_api_batch_limit_top_factors.py
# - tests/test_api_batch_explain_flag.py
# - tests/test_model_explanations.py::test_predict_proba_and_contrib_linear_explanations
# - tests/test_evaluate_baseline.py
# - tests/test_api_validate.py
```

---

## 🗺️ High-level Project Structure

```
.github/workflows/                # CI
app/ui/                           # Streamlit demo
  pages/
    0_Model.py                    # /v1/model/info view
    1_Datasets.py                 # batch scoring page
data/                             # raw & curated outputs (git-ignored)
docs/                             # architecture, notes
llm/schemas/                      # JSON/Pydantic schemas
src/procurement_risk_detection_ai/
  app/api/                        # FastAPI app (health, batch, model_info, metrics)
  app/services/                   # scoring, provenance
  llm/                            # Gemini wrapper
  models_data/                         # baseline & evaluation CLIs
  pipelines/ingestion/            # public-data ingesters (WB/OCDS/GDELT)
  pipelines/features/             # feature builders
  graph/                          # buyer↔supplier graph + metrics
tests/                            # pytest tests
```

---

## 🧭 Future Work (prioritized roadmap)

1. **Rate limiting for `/v1/score/batch`**
   Lightweight in-process token bucket (default 5 RPS, burst 10), env-tunable; return 429 with mirrored shape.

2. **Structured JSON logging (optional)**
   `STRUCTURED_LOGS=true` → log request start/finish, latency, items, used_model, explain, join_graph.

3. **Model info enrichment**
   `/v1/model/info` to include `calibrator_path`, `calibration_method`, vectorized path availability, and last metrics snapshot link.

4. **Feature pipeline hardening**
   Add schema contracts for parquet columns, validation CLI, and simple drift checks (summary stats, nulls, ranges).

5. **Model tracking**
   Optional MLflow or light-weight experiment log for metrics and artifacts; version pinning in meta.

6. **Auth & multi-tenant hardening**
   API keys + request quotas; provenance logs include tenant id; secure CORS config for UI.

7. **Packaging & ops**
   Dockerfile + compose profile; optional Helm chart; `/metrics` scrape example; `.env.example` parity checks in CI.

8. **Labeling & supervised upgrades**
   Integrate weak/true labels when available; try calibrated tree models & monotonic GBMs; export SHAP values for tree-based models.

9. **Graph analytics enrichment**
   Add additional features (clustering, assortativity, triadic closure) and supplier similarity; optional community detection features.

10. **Fairness & transparency**
    Add transparency statements for risk factors/features; simple fairness diagnostics dashboard in Streamlit.

---

## 📄 License

MIT © 2025 Sohan Puthran
