# tests/test_ocds_loader.py
from procurement_risk_detection_ai.pipelines.ingestion.ocds_loader import (
    normalize_from_releases,
)


def test_normalize_from_releases_minimal():
    # Minimal OCDS-like structure with one release, one tender, one award, one supplier
    releases = [
        {
            "ocid": "ocds-12345",
            "id": "1",
            "date": "2024-06-01T00:00:00Z",
            "tender": {
                "id": "T-001",
                "mainProcurementCategory": "goods",
                "procurementMethod": "open",
                "status": "complete",
                "value": {"amount": 100000, "currency": "USD"},
                "items": [
                    {"classification": {"id": "30124500"}},
                    {"classification": {"id": "30122000"}},
                ],
            },
            "awards": [
                {
                    "id": "A-001",
                    "date": "2024-06-10T00:00:00Z",
                    "status": "active",
                    "value": {"amount": 95000, "currency": "USD"},
                    "suppliers": [{"id": "S-001", "name": "ACME Ltd."}],
                }
            ],
            "buyer": {"id": "B-001", "name": "Ministry of Works"},
            "parties": [
                {
                    "id": "S-001",
                    "name": "ACME Ltd.",
                    "roles": ["supplier"],
                    "address": {"countryName": "Kenya"},
                }
            ],
        }
    ]

    tenders, awards, suppliers = normalize_from_releases(releases)
    # Check columns exist
    assert set(
        [
            "tender_id",
            "ocid",
            "buyer_id",
            "buyer_name",
            "main_category",
            "method",
            "status",
            "value_amount",
            "value_currency",
            "cpv_ids",
            "tender_date",
        ]
    ).issubset(set(tenders.columns))
    assert set(
        [
            "award_id",
            "ocid",
            "tender_id",
            "supplier_id",
            "supplier_name",
            "amount",
            "currency",
            "date",
            "status",
        ]
    ).issubset(set(awards.columns))
    assert set(["supplier_id", "name", "country"]).issubset(set(suppliers.columns))

    # Validate values
    assert tenders.iloc[0]["tender_id"] == "T-001"
    assert awards.iloc[0]["award_id"] == "A-001"
    assert awards.iloc[0]["supplier_name"] == "ACME Ltd."
    assert suppliers.iloc[0]["name"] == "ACME Ltd."
    assert suppliers.iloc[0]["country"] == "Kenya"
