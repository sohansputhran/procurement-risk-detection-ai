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


def test_build_features_minimal(tmp_path):
    # Create tiny curated parquet inputs
    tenders = pd.DataFrame(
        [
            {
                "tender_id": "T1",
                "buyer_name": "Buyer A",
                "main_category": "goods",
                "tender_date": "2024-01-01T00:00:00Z",
            },
            {
                "tender_id": "T2",
                "buyer_name": "Buyer A",
                "main_category": "goods",
                "tender_date": "2024-01-05T00:00:00Z",
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
            {
                "award_id": "A2",
                "tender_id": "T2",
                "supplier_id": "S1",
                "supplier_name": "ACME",
                "amount": 20000,
                "date": "2024-01-20T00:00:00Z",
            },
        ]
    )
    tenders_path = tmp_path / "tenders.parquet"
    awards_path = tmp_path / "awards.parquet"
    tenders.to_parquet(tenders_path, index=False)
    awards.to_parquet(awards_path, index=False)

    feats = build_features(str(tenders_path), str(awards_path))
    assert set(
        [
            "award_id",
            "award_concentration_by_buyer",
            "repeat_winner_ratio",
            "amount_zscore_by_category",
            "near_threshold_flag",
            "time_to_award_days",
        ]
    ).issubset(set(feats.columns))
    # Both awards to same supplier by same buyer -> concentration=1.0
    assert (
        feats.loc[feats["award_id"] == "A1", "award_concentration_by_buyer"].iloc[0]
        == 1.0
    )
    # Near threshold flag true for 10k
    assert feats.loc[feats["award_id"] == "A1", "near_threshold_flag"].iloc[0] == 1
    # time_to_award_days positive
    assert feats["time_to_award_days"].min() >= 0
