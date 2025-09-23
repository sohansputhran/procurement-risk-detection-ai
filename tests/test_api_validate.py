from __future__ import annotations
import pandas as pd
from fastapi.testclient import TestClient

from procurement_risk_detection_ai.app.api.main import app
from procurement_risk_detection_ai.app.api import batch as batch_module

client = TestClient(app)


def test_validate_list_in(monkeypatch):
    # Stub features parquet to contain A1 only
    def fake_load_parquet_cached(path: str) -> pd.DataFrame:
        return pd.DataFrame({"award_id": ["A1"], "repeat_winner_ratio": [0.5]})

    monkeypatch.setattr(batch_module, "_load_parquet_cached", fake_load_parquet_cached)

    payload = [{"award_id": "A1"}, {"award_id": "A2"}]
    resp = client.post("/v1/score/validate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2
    # A1 valid, A2 not found
    assert (
        data[0]["award_id"] == "A1"
        and data[0]["valid"] is True
        and data[0]["error"] is None
    )
    assert (
        data[1]["award_id"] == "A2"
        and data[1]["valid"] is False
        and "not found" in data[1]["error"]
    )
