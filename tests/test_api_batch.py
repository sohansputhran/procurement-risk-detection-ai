# tests/test_api_batch.py
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient


# Point the API to a temp features parquet before importing the router
def make_temp_features(tmp_path):
    feats = pd.DataFrame(
        [
            {
                "award_id": "A1",
                "award_concentration_by_buyer": 1.0,
                "repeat_winner_ratio": 1.0,
                "amount_zscore_by_category": 2.0,  # becomes ~0.833 norm
                "near_threshold_flag": 1,
                "time_to_award_days": 5,
            },
            {
                "award_id": "A2",
                "award_concentration_by_buyer": 0.0,
                "repeat_winner_ratio": 0.0,
                "amount_zscore_by_category": -1.0,  # becomes ~0.333 norm
                "near_threshold_flag": 0,
                "time_to_award_days": 0,
            },
        ]
    )
    path = tmp_path / "contracts_features.parquet"
    feats.to_parquet(path, index=False)
    return str(path)


def test_batch_scores(tmp_path, monkeypatch):
    features_path = make_temp_features(tmp_path)
    monkeypatch.setenv("FEATURES_PATH", features_path)

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
    # A1 should score higher than A2
    s1 = next(x for x in data if x["award_id"] == "A1")["risk_score"]
    s2 = next(x for x in data if x["award_id"] == "A2")["risk_score"]
    assert s1 > s2
    # top_factors present
    assert "top_factors" in data[0] and len(data[0]["top_factors"]) >= 1
