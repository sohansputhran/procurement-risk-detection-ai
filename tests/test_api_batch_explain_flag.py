from __future__ import annotations
import pandas as pd
from fastapi.testclient import TestClient

from procurement_risk_detection_ai.app.api.main import app
from procurement_risk_detection_ai.app.api import batch as batch_module

client = TestClient(app)


def _stub_features():
    # Minimal features needed by the heuristic/model paths
    return pd.DataFrame(
        {
            "award_id": ["A1", "A2"],
            "supplier_id": ["S1", "S2"],
            "repeat_winner_ratio": [0.2, 0.8],
            "amount_zscore_by_category": [0.5, 2.5],
            "award_concentration_by_buyer": [0.1, 0.7],
            "near_threshold_flag": [0, 1],
            "time_to_award_days": [10, 60],
        }
    )


def test_explain_false_list_in(monkeypatch):
    monkeypatch.setattr(
        batch_module, "_load_parquet_cached", lambda _: _stub_features()
    )

    payload = [{"award_id": "A1"}, {"award_id": "A2"}]
    resp = client.post("/v1/score/batch?explain=false", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2
    for item in data:
        assert "award_id" in item
        # core assertion: no explanations when explain=false
        assert item.get("top_factors") == []


def test_explain_false_envelope_in(monkeypatch):
    monkeypatch.setattr(
        batch_module, "_load_parquet_cached", lambda _: _stub_features()
    )

    payload = {"items": [{"award_id": "A1"}, {"award_id": "A2"}]}
    resp = client.post("/v1/score/batch?explain=false", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict) and "items" in data
    items = data["items"]
    assert len(items) == 2
    for item in items:
        assert "award_id" in item
        assert item.get("top_factors") == []
