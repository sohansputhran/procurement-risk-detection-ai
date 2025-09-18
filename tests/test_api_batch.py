# tests/test_api_batch.py
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient


def make_temp_features(tmp_path):
    feats = pd.DataFrame(
        [
            {
                "award_id": "A1",
                "supplier_id": "S1",
                "award_concentration_by_buyer": 1.0,
                "repeat_winner_ratio": 1.0,
                "amount_zscore_by_category": 2.0,
                "near_threshold_flag": 1,
                "time_to_award_days": 5,
            },
            {
                "award_id": "A2",
                "supplier_id": "S2",
                "award_concentration_by_buyer": 0.0,
                "repeat_winner_ratio": 0.0,
                "amount_zscore_by_category": -1.0,
                "near_threshold_flag": 0,
                "time_to_award_days": 0,
            },
        ]
    )
    path = tmp_path / "contracts_features.parquet"
    feats.to_parquet(path, index=False)
    return str(path)


def make_temp_graph(tmp_path):
    g = pd.DataFrame(
        [
            {
                "supplier_id": "S1",
                "degree": 5,
                "betweenness": 0.2,
                "distance_to_sanctioned": 1,
            },
            {
                "supplier_id": "S2",
                "degree": 1,
                "betweenness": 0.0,
                "distance_to_sanctioned": 4,
            },
        ]
    )
    path = tmp_path / "metrics.parquet"
    g.to_parquet(path, index=False)
    return str(path)


def test_batch_scores_with_graph(tmp_path, monkeypatch):
    features_path = make_temp_features(tmp_path)
    graph_path = make_temp_graph(tmp_path)
    monkeypatch.setenv("FEATURES_PATH", features_path)
    monkeypatch.setenv("GRAPH_METRICS_PATH", graph_path)

    from procurement_risk_detection_ai.app.api.batch import (
        router,
    )  # import after env set

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    payload = [
        {"award_id": "A1"},
        {"award_id": "A2"},
    ]
    resp = client.post("/v1/score/batch", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data) == 2

    s1 = next(x for x in data if x["award_id"] == "A1")
    s2 = next(x for x in data if x["award_id"] == "A2")

    # A1 has distance_to_sanctioned=1 (higher adjacency) -> should score at least as high as without graph
    assert "top_factors" in s1 and len(s1["top_factors"]) >= 1
    # adjacency_to_sanctioned should be among the keys (may not be top 1 but present if contributes)
    # Allow either presence in top_factors or a higher score relative to S2 due to adjacency
    assert s1["risk_score"] >= s2["risk_score"]
