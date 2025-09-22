# tests/test_evaluate_baseline.py
from __future__ import annotations
import pandas as pd

from procurement_risk_detection_ai.models.evaluate_baseline import evaluate


def test_evaluate_baseline_on_synthetic():
    # Synthetic rows with the expected feature columns
    df = pd.DataFrame(
        [
            {
                "award_concentration_by_buyer": 0.9,
                "repeat_winner_ratio": 0.95,
                "amount_zscore_by_category": 3.0,
                "near_threshold_flag": 1,
                "time_to_award_days": 5,
            },
            {
                "award_concentration_by_buyer": 0.1,
                "repeat_winner_ratio": 0.05,
                "amount_zscore_by_category": -0.2,
                "near_threshold_flag": 0,
                "time_to_award_days": 60,
            },
            {
                "award_concentration_by_buyer": 0.6,
                "repeat_winner_ratio": 0.7,
                "amount_zscore_by_category": 1.8,
                "near_threshold_flag": 0,
                "time_to_award_days": 20,
            },
        ]
    )

    feature_cols = [
        "award_concentration_by_buyer",
        "repeat_winner_ratio",
        "amount_zscore_by_category",
        "near_threshold_flag",
        "time_to_award_days",
    ]

    metrics = evaluate(df, feature_cols)
    # Basic shape checks
    assert set(
        ["auc_roc", "avg_precision", "brier", "log_loss", "pos_rate", "n"]
    ).issubset(metrics.keys())
    assert 0 < metrics["n"] == len(df)
    # Probabilities should produce finite metrics
    assert 0.0 <= metrics["brier"] <= 1.0
