import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from procurement_risk_detection_ai.app.api.main import app


@pytest.fixture
def tmp_log_dir(tmp_path, monkeypatch):
    d = tmp_path / "logs"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PROVENANCE_LOG_DIR", str(d))
    # also set dataset paths so they appear in logs
    monkeypatch.setenv("FEATURES_PATH", "data/feature_store/contracts_features.parquet")
    monkeypatch.setenv("GRAPH_METRICS_PATH", "data/graph/metrics.parquet")
    monkeypatch.setenv(
        "WB_INELIGIBLE_PATH", "data/curated/worldbank/ineligible.parquet"
    )
    monkeypatch.setenv("OCDS_TENDERS_PATH", "data/curated/ocds/tenders.parquet")
    monkeypatch.setenv("OCDS_AWARDS_PATH", "data/curated/ocds/awards.parquet")
    return d


def _read_all_jsonl(p: Path):
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def test_score_logs_provenance(tmp_log_dir):
    client = TestClient(app)

    payload = {
        "amount": 100000,
        "past_awards_count": 2,
        "is_sanctioned": False,
        "adverse_media_count": 0,
    }
    r = client.post("/v1/score", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert (
        "provenance_id" in data and data["provenance_id"]
    ), "provenance_id missing in response"

    today = next(tmp_log_dir.glob("*.jsonl"), None)
    assert today is not None, f"No provenance log written in {tmp_log_dir}"
    rows = _read_all_jsonl(today)
    assert any(row.get("request_id") == data["provenance_id"] for row in rows)


def test_log_includes_env_paths(tmp_log_dir):
    client = TestClient(app)
    payload = {
        "amount": 50000,
        "past_awards_count": 1,
        "is_sanctioned": False,
        "adverse_media_count": 1,
    }
    r = client.post("/v1/score", json=payload)
    assert r.status_code == 200
    today = next(tmp_log_dir.glob("*.jsonl"))
    rows = _read_all_jsonl(today)
    row = rows[-1]
    for k in [
        "features_path",
        "graph_metrics_path",
        "wb_ineligible_path",
        "ocds_tenders_path",
        "ocds_awards_path",
    ]:
        assert k in row
