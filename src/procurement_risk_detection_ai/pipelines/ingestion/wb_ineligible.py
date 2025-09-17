# src/procurement_risk_detection_ai/pipelines/ingestion/wb_ineligible.py
"""
World Bank Ineligible (Sanctions) ingester (public-only).

Strategy (robust, no credentials required):
1) If --xlsx-url is provided, download that Excel and parse it with pandas.
2) Otherwise fetch the public "Debarred Firms & Individuals" page and attempt to
   discover a direct Excel link (.xlsx) and parse it.
3) Normalize to a tidy parquet with these columns:
   - name (original)
   - normalized_name (lowercase/trimmed, collapsed whitespace)
   - country
   - grounds
   - start_date (ISO-8601)
   - end_date (ISO-8601 or null for "indefinite"/ongoing)
   - source_url (the specific Excel used)
   - updated_at (ingestion timestamp in UTC)

Notes:
- The official landing page updates frequently and typically links to a downloadable Excel.
- We avoid scraping the dynamic HTML table; the Excel is the canonical downloadable artifact.
- If discovery fails (e.g., page structure changes), pass --xlsx-url manually.

Usage:
    python -m procurement_risk_detection_ai.pipelines.ingestion.wb_ineligible --out-dir data
    python -m procurement_risk_detection_ai.pipelines.ingestion.wb_ineligible --xlsx-url https://.../DebarredFirms.xlsx --out-dir data
"""
from __future__ import annotations

import argparse
import io
import os
import re
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

WB_DEBARRED_PAGE = (
    "https://www.worldbank.org/en/projects-operations/procurement/debarred-firms"
)
USER_AGENT = "procurement-risk-detection-ai/0.2 (public data ingestion; sanctions)"
XL_EXTS = (".xlsx", ".xls")


def try_html_table(page_url: str) -> pd.DataFrame | None:
    try:
        tables = pd.read_html(page_url, flavor="lxml")  # needs lxml installed
    except Exception:
        return None
    # Pick the first table that has any of these columns
    candidates = ("Firm", "Firm Name", "Name", "Entity")
    for t in tables:
        cols = {str(c).strip() for c in t.columns}
        if any(any(c in col for col in cols) for c in candidates):
            return t
    return None


def _http_get(
    url: str, session: Optional[requests.Session] = None, **kwargs
) -> requests.Response:
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", USER_AGENT)
    resp = s.get(url, timeout=60, **kwargs)
    resp.raise_for_status()
    return resp


def discover_xlsx_url(page_url: str = WB_DEBARRED_PAGE) -> Optional[str]:
    """
    Best-effort: scan the landing page for a direct .xlsx link.
    Returns the first absolute URL that looks like an Excel file.
    """
    try:
        r = _http_get(page_url)
    except Exception:
        return None

    # Find all href="..."
    hrefs = re.findall(r'href=["\\\']([^"\\\']+)["\\\']', r.text, flags=re.IGNORECASE)
    # Heuristic ranking: prefer URLs that contain 'debar' / 'ineligible' / 'firms'
    candidates = []
    for h in hrefs:
        if any(h.lower().endswith(ext) for ext in XL_EXTS):
            score = 0
            lower = h.lower()
            for kw in ("debar", "ineligible", "firms", "sanction"):
                if kw in lower:
                    score += 1
            # Make absolute
            abs_url = h if bool(urlparse(h).netloc) else urljoin(page_url, h)
            candidates.append((score, abs_url))

    if not candidates:
        return None

    # Highest score first
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def _coalesce_first(df: pd.DataFrame, possible_cols: Iterable[str]) -> Optional[str]:
    for c in possible_cols:
        if c in df.columns:
            return c
    return None


def _norm_whitespace(s: Optional[str]) -> Optional[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    return re.sub(r"\\s+", " ", str(s)).strip()


def _norm_name(s: Optional[str]) -> Optional[str]:
    s2 = _norm_whitespace(s)
    return s2.lower() if s2 is not None else None


def _parse_date(s: object) -> Optional[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    txt = str(s).strip()
    # Common cases: '29-JUN-2016', 'January 5, 2024', '2019-01-31', 'Ongoing'
    if txt.lower() in {"ongoing", "indefinite", "n/a", "na", ""}:
        return None
    try:
        dt = pd.to_datetime(
            txt, dayfirst=False, errors="coerce", infer_datetime_format=True
        )
    except Exception:
        dt = None
    if dt is None or pd.isna(dt):
        # Try dayfirst=True as a fallback
        try:
            dt = pd.to_datetime(
                txt, dayfirst=True, errors="coerce", infer_datetime_format=True
            )
        except Exception:
            dt = None
    if dt is None or pd.isna(dt):
        return None
    # Return ISO date (no time)
    return pd.Timestamp(dt).date().isoformat()


def normalize_excel(df: pd.DataFrame, source_url: str) -> pd.DataFrame:
    """
    Normalize an Excel sheet into the standard schema.
    The Excel may have headers like:
      - 'Firm Name', 'Name of Firm', 'Name'
      - 'Country'
      - 'From', 'Ineligibility From', 'Ineligibility Period From'
      - 'To', 'Ineligibility To', 'Ineligibility Period To'
      - 'Grounds'
      - 'Address'
    """
    # Standardize column names for matching
    df = df.copy()
    df.columns = [re.sub(r"\\s+", " ", str(c)).strip() for c in df.columns]

    name_col = _coalesce_first(
        df, ["Firm Name", "Name of Firm", "Name", "Entity", "Firm"]
    )
    country_col = _coalesce_first(df, ["Country", "Country of Origin"])
    grounds_col = _coalesce_first(df, ["Grounds", "Ground", "Basis"])
    from_col = _coalesce_first(
        df, ["From", "Ineligibility From", "Ineligibility Period From", "Start Date"]
    )
    to_col = _coalesce_first(
        df, ["To", "Ineligibility To", "Ineligibility Period To", "End Date"]
    )

    # Some versions bundle the period as a single column; try to split on " - "
    period_col = None
    if from_col is None and to_col is None:
        period_col = _coalesce_first(
            df, ["Ineligibility Period", "Period", "Ineligibility"]
        )

    # Build output frame
    out = pd.DataFrame()
    if name_col is None:
        # Fallback: find the first text-like column
        name_col = df.columns[0]
    out["name"] = df[name_col].astype(str).map(_norm_whitespace)
    out["normalized_name"] = out["name"].map(_norm_name)

    if country_col and country_col in df.columns:
        out["country"] = df[country_col].astype(str).map(_norm_whitespace)
    else:
        out["country"] = None

    if grounds_col and grounds_col in df.columns:
        out["grounds"] = df[grounds_col].astype(str).map(_norm_whitespace)
    else:
        out["grounds"] = None

    # Dates
    if period_col and period_col in df.columns:
        # Expect strings like "29-JUN-2016  -  29-JUN-2022" or "1-Jan-2020 to Ongoing"
        start_vals, end_vals = [], []
        for txt in df[period_col].astype(str).fillna(""):
            m = re.split(r"\\s+-\\s+|\\s+to\\s+|\\s+–\\s+", txt, flags=re.IGNORECASE)
            start_vals.append(_parse_date(m[0]) if m else None)
            end_vals.append(_parse_date(m[1]) if len(m) > 1 else None)
        out["start_date"] = start_vals
        out["end_date"] = end_vals
    else:
        out["start_date"] = df[from_col].map(_parse_date) if from_col else None
        out["end_date"] = df[to_col].map(_parse_date) if to_col else None

    out["source_url"] = source_url
    out["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Drop rows without a name
    out = out[out["name"].notna() & (out["name"].str.len() > 0)].reset_index(drop=True)

    # Deduplicate on normalized name + dates
    out = out.drop_duplicates(subset=["normalized_name", "start_date", "end_date"])

    return out[
        [
            "name",
            "normalized_name",
            "country",
            "grounds",
            "start_date",
            "end_date",
            "source_url",
            "updated_at",
        ]
    ]


def save_outputs(df: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = os.path.join(out_dir, "raw", "worldbank")
    curated_dir = os.path.join(out_dir, "curated", "worldbank")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(curated_dir, exist_ok=True)

    # Raw JSONL snapshot (cleaned rows as ingested)
    raw_path = os.path.join(raw_dir, f"ineligible_{ts}.jsonl")
    df.to_json(raw_path, orient="records", lines=True, force_ascii=False)

    # Curated parquet (stable path)
    parquet_path = os.path.join(curated_dir, "ineligible.parquet")
    df.to_parquet(parquet_path, index=False)

    return {"raw": raw_path, "parquet": parquet_path}


def main():
    ap = argparse.ArgumentParser(
        description="Ingest World Bank Ineligible (sanctions) list → parquet (public source)."
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="output directory root (default: data)",
    )
    ap.add_argument(
        "--xlsx-url",
        type=str,
        default=None,
        help="direct Excel URL (optional override)",
    )
    ap.add_argument(
        "--xlsx-file", type=str, default=None, help="local .xlsx path (downloaded file)"
    )
    ap.add_argument(
        "--page-url",
        type=str,
        default=WB_DEBARRED_PAGE,
        help="landing page to discover xlsx (optional)",
    )
    ap.add_argument(
        "--sheet-index", type=int, default=0, help="Excel sheet index (default: 0)"
    )
    ap.add_argument(
        "--print-only", action="store_true", help="print head() and exit without saving"
    )
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Guard against placeholder URLs like <...>
    if args.xlsx_url and any(ch in args.xlsx_url for ch in "<>"):
        raise SystemExit(
            "Invalid --xlsx-url: remove angle brackets and paste the real https://... .xlsx URL."
        )

    # Priority: local file > direct URL > discover from page
    if args.xlsx_file:
        if not os.path.exists(args.xlsx_file):
            raise SystemExit(f"--xlsx-file not found: {args.xlsx_file}")
        df_excel = pd.read_excel(
            args.xlsx_file, sheet_name=args.sheet_index, dtype=str, engine="openpyxl"
        )
        df_norm = normalize_excel(
            df_excel, source_url=f"file://{os.path.abspath(args.xlsx_file)}"
        )
    else:
        xlsx_url = args.xlsx_url or discover_xlsx_url(args.page_url)
        if not xlsx_url:
            raise SystemExit(
                "Could not auto-discover an Excel link from the landing page. "
                "Provide --xlsx-file PATH (downloaded .xlsx) or --xlsx-url https://... .xlsx"
            )
        # Download Excel
        resp = _http_get(xlsx_url, session=session, stream=True)
        content = resp.content
        # Some links may actually be HTML redirectors. If response is HTML, try to find a real .xlsx link inside.
        ctype = resp.headers.get("Content-Type", "").lower()
        if "text/html" in ctype and content:
            text = content.decode("utf-8", errors="ignore")
            inner_links = re.findall(
                r'href=["\']([^"\']+\.xlsx)["\']', text, flags=re.IGNORECASE
            )
            if inner_links:
                xlsx_url = (
                    inner_links[0]
                    if re.match(r"^https?://", inner_links[0], re.I)
                    else urljoin(xlsx_url, inner_links[0])
                )
                resp = _http_get(xlsx_url, session=session)
                content = resp.content
        # Parse Excel (selected sheet)
        df_excel = pd.read_excel(
            io.BytesIO(content), sheet_name=args.sheet_index, dtype=str
        )
        df_norm = normalize_excel(df_excel, source_url=xlsx_url)

    if args.print_only:
        print(df_norm.head(10).to_markdown(index=False))
        print(f"Rows: {len(df_norm)}")
        return

    outputs = save_outputs(df_norm, args.out_dir)
    print(f"Wrote {len(df_norm):,} rows")
    for k, v in outputs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
