from procurement_risk_detection_ai.pipelines.ingestion.wb_projects import normalize


def test_normalize_schema_stable():
    sample = [
        {
            "id": "P123456",
            "project_name": "Sample",
            "regionname": "AFRICA",
            "countryshortname": ["Kenya"],
            "countrycode": ["KE"],
            "projectstatusdisplay": "Active",
            "totalcommamt": "1000000",
            "totalamt": "1000000",
            "approvalfy": "2024",
            "board_approval_month": "June",
            "p2a_updated_date": "2025-09-01 00:00:00.0",
        }
    ]
    df = normalize(sample)
    assert set(df.columns) == {
        "id",
        "project_name",
        "regionname",
        "countryshortname",
        "countrycode",
        "projectstatusdisplay",
        "totalcommamt",
        "totalamt",
        "approvalfy",
        "board_approval_month",
        "p2a_updated_date",
    }
    assert df.iloc[0]["countryshortname"] == "Kenya"
