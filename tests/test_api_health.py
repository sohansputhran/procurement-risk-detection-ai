# tests/test_api_health.py
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _mk_parquet(path, rows=3, cols=None):
    cols = cols or {"x": [1] * rows}
    df = pd.DataFrame(cols)
    import pathlib

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_health_counts(tmp_path, monkeypatch):
    # create tiny parquets
    features = tmp_path / "features.parquet"
    graph = tmp_path / "graph.parquet"
    wb = tmp_path / "inel.parquet"
    tenders = tmp_path / "tenders.parquet"
    awards = tmp_path / "awards.parquet"

    _mk_parquet(features, rows=5)
    _mk_parquet(graph, rows=2)
    _mk_parquet(wb, rows=7)
    _mk_parquet(tenders, rows=11)
    _mk_parquet(awards, rows=13)

    monkeypatch.setenv("FEATURES_PATH", str(features))
    monkeypatch.setenv("GRAPH_METRICS_PATH", str(graph))
    monkeypatch.setenv("WB_INELIGIBLE_PATH", str(wb))
    monkeypatch.setenv("OCDS_TENDERS_PATH", str(tenders))
    monkeypatch.setenv("OCDS_AWARDS_PATH", str(awards))

    from procurement_risk_detection_ai.app.api.health import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200, r.text
    data = r.json()
    ds = data["datasets"]
    assert ds["features"]["available"] is True and ds["features"]["rows"] == 5
    assert ds["graph_metrics"]["rows"] == 2
    assert ds["wb_ineligible"]["rows"] == 7
    assert ds["ocds_tenders"]["rows"] == 11
    assert ds["ocds_awards"]["rows"] == 13
