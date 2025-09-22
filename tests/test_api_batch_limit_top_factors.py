from __future__ import annotations
from fastapi.testclient import TestClient

from procurement_risk_detection_ai.app.api.main import app

client = TestClient(app)


def test_batch_limit_top_factors_list_in(monkeypatch):
    # Minimal payload with two awards; list-in → list-out
    payload = [{"award_id": "A1"}, {"award_id": "A2"}]

    # Call with a strict factor limit
    resp = client.post("/v1/score/batch?limit_top_factors=3", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2

    # Each item should have <= 3 factors
    for item in data:
        assert "top_factors" in item
        assert isinstance(item["top_factors"], list)
        assert len(item["top_factors"]) <= 3
