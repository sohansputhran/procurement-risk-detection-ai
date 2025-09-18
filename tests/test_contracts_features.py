# tests/test_contracts_features.py
import pandas as pd
from procurement_risk_detection_ai.pipelines.features.contracts_features import (
    build_features,
    _near_threshold,
)


def test_near_threshold_basic():
    assert _near_threshold(10000) == 1
    assert _near_threshold(10500) == 1  # within 5%
    assert _near_threshold(16000) == 0


def test_build_features_includes_supplier(tmp_path):
    tenders = pd.DataFrame(
        [
            {
                "tender_id": "T1",
                "buyer_name": "Buyer A",
                "main_category": "goods",
                "tender_date": "2024-01-01T00:00:00Z",
            },
        ]
    )
    awards = pd.DataFrame(
        [
            {
                "award_id": "A1",
                "tender_id": "T1",
                "supplier_id": "S1",
                "supplier_name": "ACME",
                "amount": 10000,
                "date": "2024-01-03T00:00:00Z",
            },
        ]
    )
    tp = tmp_path / "tenders.parquet"
    tenders.to_parquet(tp, index=False)
    ap = tmp_path / "awards.parquet"
    awards.to_parquet(ap, index=False)

    feats = build_features(str(tp), str(ap))
    assert "supplier_id" in feats.columns
    assert feats.loc[feats["award_id"] == "A1", "supplier_id"].iloc[0] == "S1"
