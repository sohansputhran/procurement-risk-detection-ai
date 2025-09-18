import pandas as pd
from procurement_risk_detection_ai.app.ui.utils import (
    results_json_to_dataframe,
    df_to_csv_bytes,
)


def test_results_json_to_dataframe_flattens_top_factors():
    items = [
        {
            "award_id": "A1",
            "supplier_id": "S1",
            "risk_score": 0.87,
            "top_factors": [
                {"name": "repeat_winner_ratio", "value": 0.9},
                {"name": "amount_zscore_by_category", "value": 2.1},
            ],
        },
        {
            "award_id": "A2",
            "supplier_id": "S2",
            "risk_score": 0.12,
            "top_factors": [{"name": "time_to_award_days", "value": 95}],
        },
        {
            "award_id": "A3",
            "supplier_id": "S3",
            "risk_score": 0.55,
            "top_factors": None,  # missing/None should be handled gracefully
        },
    ]

    df = results_json_to_dataframe(items)
    assert not df.empty
    # max top_factors length across rows is 2 -> expect two name/value pairs
    expected_cols = {
        "award_id",
        "supplier_id",
        "risk_score",
        "top_factor_1_name",
        "top_factor_1_value",
        "top_factor_2_name",
        "top_factor_2_value",
    }
    assert expected_cols.issubset(
        set(df.columns)
    ), f"Missing expected columns: {expected_cols - set(df.columns)}"

    # Row-wise checks
    r1 = df.loc[df["award_id"] == "A1"].iloc[0]
    assert r1["top_factor_1_name"] == "repeat_winner_ratio"
    assert r1["top_factor_1_value"] == 0.9
    assert r1["top_factor_2_name"] == "amount_zscore_by_category"
    assert r1["top_factor_2_value"] == 2.1

    r2 = df.loc[df["award_id"] == "A2"].iloc[0]
    assert r2["top_factor_1_name"] == "time_to_award_days"
    assert r2["top_factor_1_value"] == 95
    assert r2["top_factor_2_name"] == ""  # filled as empty string for missing
    assert r2["top_factor_2_value"] == ""

    r3 = df.loc[df["award_id"] == "A3"].iloc[0]
    assert r3["top_factor_1_name"] == ""  # no factors present at all
    assert r3["top_factor_1_value"] == ""


def test_df_to_csv_bytes_roundtrip():
    df = pd.DataFrame(
        {
            "award_id": ["A1", "A2"],
            "risk_score": [0.1, 0.2],
            "top_factor_1_name": ["f1", ""],
            "top_factor_1_value": [1.23, ""],
        }
    )
    csv_bytes = df_to_csv_bytes(df)
    assert isinstance(csv_bytes, (bytes, bytearray))
    # quick smoke check: encoded CSV should contain header fields
    text = csv_bytes.decode("utf-8-sig")
    assert "award_id,risk_score,top_factor_1_name,top_factor_1_value" in text
    assert "A1,0.1,f1,1.23" in text
