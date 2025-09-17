# tests/test_wb_ineligible.py
import pandas as pd
from procurement_risk_detection_ai.pipelines.ingestion.wb_ineligible import (
    normalize_excel,
)


def test_normalize_excel_schema_and_dates():
    # Minimal synthetic Excel-like DataFrame
    df = pd.DataFrame(
        [
            {
                "Firm Name": "ACME Ltd.",
                "Country": "Nigeria",
                "Grounds": "Fraudulent Practice",
                "From": "29-JUN-2016",
                "To": "29-JUN-2022",
            },
            {
                "Firm Name": "Contoso S.A.",
                "Country": "Peru",
                "Grounds": "Collusive Practice",
                "From": "January 5, 2024",
                "To": "Ongoing",
            },
        ]
    )
    out = normalize_excel(df, source_url="https://example.org/debarred.xlsx")
    assert list(out.columns) == [
        "name",
        "normalized_name",
        "country",
        "grounds",
        "start_date",
        "end_date",
        "source_url",
        "updated_at",
    ]
    assert out.iloc[0]["name"] == "ACME Ltd."
    assert out.iloc[0]["start_date"] == "2016-06-29"
    assert out.iloc[0]["end_date"] == "2022-06-29"
    assert (
        out.iloc[1]["end_date"] is None or out.iloc[1]["end_date"] == ""
    )  # Ongoing becomes None
    assert out.iloc[1]["normalized_name"] == "contoso s.a."
