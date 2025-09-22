from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from procurement_risk_detection_ai.app.api.main import app
from procurement_risk_detection_ai.app.api import batch as batch_module


def test_batch_limit_top_factors_list_in(monkeypatch):
    # Force heuristic path so no model artifacts are required
    monkeypatch.setattr(batch_module, "is_model_available", lambda: False)

    # Replace the feature join with a minimal, deterministic stub
    def fake_join_features(df_in: pd.DataFrame, join_graph: bool) -> pd.DataFrame:
        df = df_in.copy()
        n = len(df)
        # Minimal set of columns used by the heuristic scorer
        df["supplier_id"] = [f"S{i+1}" for i in range(n)]
        df["near_threshold_flag"] = [1, 0][:n] + [0] * max(0, n - 2)
        df["repeat_winner_ratio"] = [0.9, 0.1][:n] + [0.1] * max(0, n - 2)
        df["amount_zscore_by_category"] = [2.0, -0.1][:n] + [0.0] * max(0, n - 2)
        df["award_concentration_by_buyer"] = [0.8, 0.2][:n] + [0.2] * max(0, n - 2)
        return df

    monkeypatch.setattr(batch_module, "_join_features", fake_join_features)

    client = TestClient(app)

    # List-in → list-out
    payload = [{"award_id": "A1"}, {"award_id": "A2"}]
    resp = client.post("/v1/score/batch?limit_top_factors=3", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2

    # Ensure the explanation length cap is respected
    for item in data:
        assert "top_factors" in item
        assert isinstance(item["top_factors"], list)
        assert len(item["top_factors"]) <= 3
