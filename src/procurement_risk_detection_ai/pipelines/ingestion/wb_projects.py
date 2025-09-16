from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests

WB_SEARCH_URL = "https://search.worldbank.org/api/v2/projects"
USER_AGENT = "procurement-risk-detection-ai/0.1 (public data ingestion)"

# Minimal set of fields that are stable & useful
FIELDS = [
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
]


def _request(
    session: requests.Session, params: Dict, retries: int = 3, backoff: float = 0.8
) -> Dict:
    err = None
    for i in range(retries):
        try:
            r = session.get(
                WB_SEARCH_URL,
                params={**params, "format": "json"},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            err = e
            time.sleep(backoff * (2**i))
    raise RuntimeError(f"World Bank API request failed after {retries} retries: {err}")


def _page(session: requests.Session, rows: int, offset: int) -> Tuple[Dict, int]:
    """
    Returns (projects_dict, returned_count)
    Response JSON shape: { 'projects': { 'PXXXXX': {...}, ... }, 'total': N, ... }
    """
    data = _request(session, params={"rows": rows, "os": offset})
    projects = data.get("projects") or {}
    return projects, len(projects)


def fetch_all(
    rows: int = 500, max_pages: int = 40, if_modified_since: str | None = None
) -> List[Dict]:
    """
    Streams all projects using pagination via the search API.
    - rows: page size (World Bank allows up to ~5k but 500 is safe)
    - max_pages: guardrail
    - if_modified_since: RFC1123 date string to enable cache-friendly pulls (best effort)
    """
    headers = {"User-Agent": USER_AGENT}
    if if_modified_since:
        headers["If-Modified-Since"] = if_modified_since

    s = requests.Session()
    s.headers.update(headers)

    offset = 0
    out: List[Dict] = []
    for page in range(max_pages):
        projects, n = _page(s, rows=rows, offset=offset)
        if n == 0:
            break
        # API returns dict keyed by project id; flatten values
        out.extend(projects.values())
        offset += rows
    return out


def normalize(records: Iterable[Dict]) -> pd.DataFrame:
    """Normalize raw API records to a tidy DataFrame with stable columns."""
    if not records:
        return pd.DataFrame(columns=FIELDS)

    df = pd.json_normalize(records)
    # Coerce list fields to scalar where needed
    for col in ("countryshortname", "countrycode", "source"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) and x else x)

    # Keep only the fields we care about (when present)
    for f in FIELDS:
        if f not in df.columns:
            df[f] = pd.NA
    return df[FIELDS].copy()


def save_outputs(df: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = os.path.join(out_dir, "raw", "worldbank")
    curated_dir = os.path.join(out_dir, "curated", "worldbank")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(curated_dir, exist_ok=True)

    raw_path = os.path.join(raw_dir, f"projects_{ts}.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    parquet_path = os.path.join(curated_dir, "projects.parquet")
    df.to_parquet(parquet_path, index=False)

    return {"raw": raw_path, "parquet": parquet_path}


def main():
    p = argparse.ArgumentParser(
        description="Ingest World Bank Projects (public API) → jsonl + parquet"
    )
    p.add_argument("--rows", type=int, default=500)
    p.add_argument("--max-pages", type=int, default=40)
    p.add_argument("--out-dir", type=str, default="data")
    p.add_argument("--print-only", action="store_true")
    args = p.parse_args()

    records = fetch_all(rows=args.rows, max_pages=args.max_pages)
    df = normalize(records)

    if args.print_only:
        print(df.head().to_markdown(index=False))
        return

    paths = save_outputs(df, args.out_dir)
    print(f"Wrote {len(df):,} records")
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
