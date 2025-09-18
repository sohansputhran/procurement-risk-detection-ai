## Provenance logging (per request)

Every API request appends a JSON line to `data/logs/provenance/YYYY-MM-DD.jsonl` with:
- `request_id` (also returned in API responses as `provenance_id`)
- timestamp, duration
- payload preview (first few items for batch)
- data source paths (from env): `FEATURES_PATH`, `GRAPH_METRICS_PATH`, `WB_INELIGIBLE_PATH`, `OCDS_TENDERS_PATH`, `OCDS_AWARDS_PATH`
- status/error (if any)

**Env (optional):**
PROVENANCE_LOG_DIR=data/logs/provenance

**Windows PowerShell examples**
```powershell
# Single
Invoke-RestMethod -Method Post -Uri "$env:API_URL/v1/score" -ContentType "application/json" -Body (@{
  amount=100000; past_awards_count=2; is_sanctioned=$false; adverse_media_count=1
} | ConvertTo-Json)

# Batch (array body)
$items = @(@{award_id="A1"}, @{award_id="A2"})
Invoke-RestMethod -Method Post -Uri "$env:API_URL/v1/score/batch" -ContentType "application/json" -Body ($items | ConvertTo-Json)

---

✅ Copy these into your repo at the specified paths:
- `src/procurement_risk_detection_ai/app/api/provenance.py`
- `tests/test_provenance_logging.py`
- `docs/README_Provenance.md`

Then run:

```bash
pip install -e .
pytest -q tests/test_provenance_logging.py
