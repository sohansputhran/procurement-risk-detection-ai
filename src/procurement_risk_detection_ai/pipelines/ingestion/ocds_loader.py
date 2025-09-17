# src/procurement_risk_detection_ai/pipelines/ingestion/ocds_loader.py
"""
OCDS Release Package loader (public-only).

Loads an OCDS Release Package (local path or URL), then normalizes key tables:
- tenders.parquet
- awards.parquet
- suppliers.parquet

Usage examples:
  # Local JSON file (optionally .json.gz)
  python -m procurement_risk_detection_ai.pipelines.ingestion.ocds_loader --path data/raw/ocds/sample.json --out-dir data

  # URL
  python -m procurement_risk_detection_ai.pipelines.ingestion.ocds_loader --url https://example.org/ocds/releases.json --out-dir data

  # Print-only (no writes)
  python -m procurement_risk_detection_ai.pipelines.ingestion.ocds_loader --path sample.json --print-only
"""

# URL for the Data: https://data.open-contracting.org/en/publication/155/download?name=full.jsonl.gz

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


def _bytes_to_package(data: bytes, is_jsonl: bool) -> dict:
    """
    Convert raw bytes into a release package dict.
    - If is_jsonl=True: parse line-by-line into {"releases":[...]}.
    - Else: parse as a single JSON object and return as-is if it already
      has a 'releases' key; or wrap a single release into that shape.
    """
    text = data.decode("utf-8", errors="ignore")
    if is_jsonl:
        releases = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            releases.append(json.loads(line))
        return {"releases": releases}
    else:
        obj = json.loads(text)
        if "releases" in obj and isinstance(obj["releases"], list):
            return obj
        # Fallback: if it's a single release, wrap it
        if isinstance(obj, dict) and obj.get("ocid"):
            return {"releases": [obj]}
        return obj


def _read_package_from_file(path: str) -> Dict:
    lower = path.lower()
    is_gz = lower.endswith(".gz")
    is_jsonl = lower.endswith(".jsonl") or lower.endswith(".jsonl.gz")
    if is_gz:
        with gzip.open(path, "rb") as f:
            data = f.read()
        return _bytes_to_package(data, is_jsonl=is_jsonl)
    else:
        with open(path, "rb") as f:
            data = f.read()
        return _bytes_to_package(data, is_jsonl=is_jsonl)


def _read_package_from_url(url: str) -> Dict:
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    lower = url.lower()
    ctype = (r.headers.get("Content-Type") or "").lower()
    # Decide gzip & jsonl from URL or headers
    is_gz = lower.endswith(".gz") or "gzip" in ctype
    is_jsonl = (
        lower.endswith(".jsonl") or lower.endswith(".jsonl.gz") or "jsonl" in lower
    )
    data = r.content
    if is_gz:
        try:
            data = gzip.decompress(data)
        except OSError:
            # Some servers already decompress; fall back to raw content
            pass
    return _bytes_to_package(data, is_jsonl=is_jsonl)


def _hash_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _get_first(xs: Iterable, default=None):
    for x in xs:
        return x
    return default


def _party_country(party: Dict) -> Optional[str]:
    addr = party.get("address") or {}
    # Prefer countryName if present; else region or addressLocality
    return addr.get("countryName") or addr.get("region") or addr.get("addressLocality")


def normalize_from_releases(
    releases: List[Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Normalize a list of OCDS releases into (tenders, awards, suppliers) DataFrames.
    - tenders: tender_id, ocid, buyer_id, buyer_name, main_category, method, status, value_amount, value_currency, cpv_ids (list as ';'-joined), tender_date
    - awards:  award_id, ocid, tender_id, supplier_id (first), supplier_name (first), amount, currency, date, status
    - suppliers: supplier_id, name, country
    """
    tenders_rows = []
    awards_rows = []
    suppliers_map: Dict[str, Dict] = {}

    for rel in releases:
        ocid = rel.get("ocid")
        # release_id = rel.get("id")
        tender = rel.get("tender") or {}
        buyer = (rel.get("buyer") or {}) or {}
        parties = rel.get("parties") or []

        # Build a map of parties for supplier lookup
        party_by_id = {p.get("id"): p for p in parties if p.get("id")}

        # --- Tenders ---
        tender_id = tender.get("id") or (f"{ocid}-tender")
        # CPV codes: tender.classification.id or tender.items[*].classification.id
        cpv_ids = []
        if isinstance(tender.get("classification"), dict):
            cid = tender["classification"].get("id")
            if cid:
                cpv_ids.append(str(cid))
        for it in tender.get("items", []) or []:
            cls = it.get("classification") or {}
            cid = cls.get("id")
            if cid:
                cpv_ids.append(str(cid))
        cpv_ids = sorted(set(cpv_ids))
        value = tender.get("value") or {}
        tenders_rows.append(
            {
                "tender_id": str(tender_id),
                "ocid": str(ocid) if ocid else None,
                "buyer_id": (buyer.get("id") if isinstance(buyer, dict) else None),
                "buyer_name": (buyer.get("name") if isinstance(buyer, dict) else None),
                "main_category": tender.get("mainProcurementCategory"),
                "method": tender.get("procurementMethod"),
                "status": tender.get("status"),
                "value_amount": value.get("amount"),
                "value_currency": value.get("currency"),
                "cpv_ids": ";".join(cpv_ids) if cpv_ids else None,
                "tender_date": rel.get("date")
                or tender.get("tenderPeriod", {}).get("startDate"),
            }
        )

        # --- Awards ---
        for idx, aw in enumerate(rel.get("awards") or []):
            award_id = aw.get("id") or f"{ocid}-award-{idx:04d}"
            amount = (aw.get("value") or {}).get("amount")
            currency = (aw.get("value") or {}).get("currency")
            date = aw.get("date")
            status = aw.get("status")
            # Flatten first supplier (if multiple, keep first)
            supplier_name = None
            supplier_id = None
            suppliers_field = aw.get("suppliers") or []
            if suppliers_field:
                s0 = suppliers_field[0]
                supplier_name = s0.get("name")
                supplier_id = s0.get("id") or _hash_id(
                    (supplier_name or "unknown") + f"|{ocid}"
                )
                # Record supplier in suppliers_map
                country = None
                if "id" in s0 and s0["id"] in party_by_id:
                    country = _party_country(party_by_id[s0["id"]])
                suppliers_map.setdefault(
                    supplier_id,
                    {
                        "supplier_id": supplier_id,
                        "name": supplier_name,
                        "country": country,
                    },
                )
            else:
                # Fall back: check parties with role supplier
                sup_parties = [
                    p for p in parties if "roles" in p and "supplier" in p["roles"]
                ]
                if sup_parties:
                    p0 = sup_parties[0]
                    supplier_id = p0.get("id") or _hash_id(
                        (p0.get("name") or "unknown") + f"|{ocid}"
                    )
                    supplier_name = p0.get("name")
                    country = _party_country(p0)
                    suppliers_map.setdefault(
                        supplier_id,
                        {
                            "supplier_id": supplier_id,
                            "name": supplier_name,
                            "country": country,
                        },
                    )

            awards_rows.append(
                {
                    "award_id": str(award_id),
                    "ocid": str(ocid) if ocid else None,
                    "tender_id": str(tender_id),
                    "supplier_id": supplier_id,
                    "supplier_name": supplier_name,
                    "amount": amount,
                    "currency": currency,
                    "date": date,
                    "status": status,
                }
            )

        # --- Suppliers (from parties) ---
        for p in parties:
            if "roles" in p and "supplier" in (p.get("roles") or []):
                sid = p.get("id") or _hash_id((p.get("name") or "unknown") + f"|{ocid}")
                suppliers_map.setdefault(
                    sid,
                    {
                        "supplier_id": sid,
                        "name": p.get("name"),
                        "country": _party_country(p),
                    },
                )

    tenders_df = pd.DataFrame(
        tenders_rows,
        columns=[
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
        ],
    )
    awards_df = pd.DataFrame(
        awards_rows,
        columns=[
            "award_id",
            "ocid",
            "tender_id",
            "supplier_id",
            "supplier_name",
            "amount",
            "currency",
            "date",
            "status",
        ],
    )
    suppliers_df = pd.DataFrame(
        list(suppliers_map.values()), columns=["supplier_id", "name", "country"]
    )
    return tenders_df, awards_df, suppliers_df


def _save_outputs(
    tenders: pd.DataFrame, awards: pd.DataFrame, suppliers: pd.DataFrame, out_dir: str
) -> Dict[str, str]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    curated = os.path.join(out_dir, "curated", "ocds")
    raw = os.path.join(out_dir, "raw", "ocds")
    os.makedirs(curated, exist_ok=True)
    os.makedirs(raw, exist_ok=True)

    paths = {
        "tenders": os.path.join(curated, "tenders.parquet"),
        "awards": os.path.join(curated, "awards.parquet"),
        "suppliers": os.path.join(curated, "suppliers.parquet"),
    }
    tenders.to_parquet(paths["tenders"], index=False)
    awards.to_parquet(paths["awards"], index=False)
    suppliers.to_parquet(paths["suppliers"], index=False)

    # Small raw snapshot (counts)
    with open(os.path.join(raw, f"ocds_counts_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": ts,
                "tenders": int(len(tenders)),
                "awards": int(len(awards)),
                "suppliers": int(len(suppliers)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return paths


def main():
    ap = argparse.ArgumentParser(
        description="Load an OCDS Release Package → tenders/awards/suppliers parquet (public data)."
    )
    ap.add_argument(
        "--path",
        type=str,
        default=None,
        help="Local .json or .json.gz path to a Release Package",
    )
    ap.add_argument(
        "--url", type=str, default=None, help="HTTP(S) URL to a Release Package"
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Output directory root (default: data)",
    )
    ap.add_argument(
        "--print-only",
        action="store_true",
        help="Print head() and row counts; do not write parquet",
    )
    ap.add_argument(
        "--take", type=int, default=None, help="Limit to first N releases (debug)"
    )
    args = ap.parse_args()

    if not args.path and not args.url:
        raise SystemExit("Provide either --path or --url")

    package = (
        _read_package_from_file(args.path)
        if args.path
        else _read_package_from_url(args.url)
    )

    releases = package.get("releases") or []
    if args.take:
        releases = releases[: args.take]

    tenders, awards, suppliers = normalize_from_releases(releases)

    if args.print_only:
        print("Tenders:")
        print(tenders.head().to_markdown(index=False))
        print("\nAwards:")
        print(awards.head().to_markdown(index=False))
        print("\nSuppliers:")
        print(suppliers.head().to_markdown(index=False))
        print(
            f"\nCounts: tenders={len(tenders)} awards={len(awards)} suppliers={len(suppliers)}"
        )
        return

    paths = _save_outputs(tenders, awards, suppliers, args.out_dir)
    print("Wrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
