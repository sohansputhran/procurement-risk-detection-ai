# tests/test_graph_utils.py
import pandas as pd
from procurement_risk_detection_ai.graph.graph_utils import compute_metrics


def test_compute_metrics_basic(tmp_path):
    # Build tiny tenders/awards and sanctions parquet
    tenders = pd.DataFrame(
        [
            {"tender_id": "T1", "buyer_id": "B1", "buyer_name": "Ministry of Works"},
            {"tender_id": "T2", "buyer_id": "B1", "buyer_name": "Ministry of Works"},
        ]
    )
    awards = pd.DataFrame(
        [
            {
                "award_id": "A1",
                "tender_id": "T1",
                "supplier_id": "S-sanctioned",
                "supplier_name": "ACME Ltd.",
                "amount": 10000,
            },
            {
                "award_id": "A2",
                "tender_id": "T1",
                "supplier_id": "S2",
                "supplier_name": "Beta Corp",
                "amount": 5000,
            },
            {
                "award_id": "A3",
                "tender_id": "T2",
                "supplier_id": "S3",
                "supplier_name": "Gamma LLC",
                "amount": 8000,
            },
        ]
    )
    sanctions = pd.DataFrame([{"name": "ACME Ltd.", "normalized_name": "acme ltd"}])

    tenders_path = tmp_path / "tenders.parquet"
    awards_path = tmp_path / "awards.parquet"
    sanctions_path = tmp_path / "ineligible.parquet"
    tenders.to_parquet(tenders_path, index=False)
    awards.to_parquet(awards_path, index=False)
    sanctions.to_parquet(sanctions_path, index=False)

    metrics, _ = compute_metrics(
        tenders_path=str(tenders_path),
        awards_path=str(awards_path),
        sanctions_path=str(sanctions_path),
        out_dir=str(tmp_path),
        ego_supplier_id=None,
        compute_betweenness=False,  # faster for test
    )

    # Columns exist
    for col in [
        "supplier_id",
        "supplier_name",
        "degree",
        "betweenness",
        "distance_to_sanctioned",
    ]:
        assert col in metrics.columns

    # Sanctioned supplier should have distance 0
    d0 = metrics.loc[
        metrics["supplier_id"] == "S-sanctioned", "distance_to_sanctioned"
    ].iloc[0]
    assert d0 == 0

    # Supplier S2 shares buyer with sanctioned supplier -> distance 2 in bipartite graph
    d2 = metrics.loc[metrics["supplier_id"] == "S2", "distance_to_sanctioned"].iloc[0]
    assert d2 == 2

    # Degrees should be >=1
    assert metrics["degree"].min() >= 1
