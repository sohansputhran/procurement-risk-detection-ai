# src/procurement_risk_detection_ai/pipelines/features/contracts_features.py
"""
Feature Seeds v1 for contract/award-level risk signals.

Inputs (from curated OCDS parquet):
- data/curated/ocds/tenders.parquet  (tender_id, buyer_name, main_category, tender_date)
- data/curated/ocds/awards.parquet   (award_id, tender_id, supplier_id, supplier_name, amount, currency, date, status)

Outputs:
- data/feature_store/contracts_features.parquet, keyed by award_id with:
  - supplier_id: for joining graph metrics
  - award_concentration_by_buyer: share of a buyer's awards that went to this supplier (0..1)
  - repeat_winner_ratio: same metric (alias kept for clarity)
  - amount_zscore_by_category: z-score of award amount within main_category
  - near_threshold_flag: 1 if amount within ±5% of {10k, 50k, 250k, 1M}
  - time_to_award_days: (award_date - tender_date) in days, clipped at [0, 3650]
"""
from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd


GENERIC_THRESHOLDS = [10_000, 50_000, 250_000, 1_000_000]
NEAR_PCT = 0.05  # 5% window


def _parse_date(s: object) -> Optional[pd.Timestamp]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    try:
        ts = pd.to_datetime(s, utc=True, errors="coerce", infer_datetime_format=True)
        if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
            return ts
        return None
    except Exception:
        return None


def _near_threshold(
    amount: Optional[float],
    thresholds: Iterable[float] = GENERIC_THRESHOLDS,
    pct: float = NEAR_PCT,
) -> int:
    if amount is None or (
        isinstance(amount, float) and (np.isnan(amount) or amount <= 0)
    ):
        return 0
    for t in thresholds:
        lo, hi = (1 - pct) * t, (1 + pct) * t
        if lo <= amount <= hi:
            return 1
    return 0


def _zscore(series: pd.Series) -> pd.Series:
    s = series.astype("float64")
    mu = s.mean()
    sigma = s.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sigma


def build_features(tenders_path: str, awards_path: str) -> pd.DataFrame:
    tenders = pd.read_parquet(tenders_path)
    awards = pd.read_parquet(awards_path)

    # Basic cleaning
    tenders["buyer_name"] = (
        tenders.get("buyer_name", pd.Series([None] * len(tenders)))
        .astype("string")
        .str.strip()
    )
    tenders["main_category"] = (
        tenders.get("main_category", pd.Series([None] * len(tenders)))
        .astype("string")
        .str.strip()
    )
    tenders["tender_date"] = tenders.get(
        "tender_date", pd.Series([None] * len(tenders))
    ).apply(_parse_date)

    awards["supplier_id"] = (
        awards.get("supplier_id", pd.Series([None] * len(awards)))
        .astype("string")
        .str.strip()
    )
    awards["supplier_name"] = (
        awards.get("supplier_name", pd.Series([None] * len(awards)))
        .astype("string")
        .str.strip()
    )
    awards["amount"] = pd.to_numeric(
        awards.get("amount", pd.Series([None] * len(awards))), errors="coerce"
    )
    awards["date"] = awards.get("date", pd.Series([None] * len(awards))).apply(
        _parse_date
    )

    # Join awards -> tenders for buyer & category & tender_date
    df = awards.merge(
        tenders[["tender_id", "buyer_name", "main_category", "tender_date"]],
        on="tender_id",
        how="left",
    )

    # --- award_concentration_by_buyer & repeat_winner_ratio ---
    grp = (
        df.groupby(["buyer_name", "supplier_id"], dropna=False)
        .size()
        .rename("awards_by_pair")
        .reset_index()
    )
    buyer_totals = (
        df.groupby("buyer_name", dropna=False)
        .size()
        .rename("awards_by_buyer")
        .reset_index()
    )
    conc = grp.merge(buyer_totals, on="buyer_name", how="left")
    conc["share"] = conc["awards_by_pair"] / conc["awards_by_buyer"].replace(0, np.nan)
    conc = conc.replace([np.inf, -np.inf], np.nan).fillna({"share": 0.0})
    df = df.merge(
        conc[["buyer_name", "supplier_id", "share"]],
        on=["buyer_name", "supplier_id"],
        how="left",
    )
    df["award_concentration_by_buyer"] = df["share"].fillna(0.0)
    df["repeat_winner_ratio"] = df["award_concentration_by_buyer"]  # alias for clarity

    # --- amount_zscore_by_category ---
    df["amount_zscore_by_category"] = (
        df.groupby("main_category", dropna=False)["amount"]
        .transform(_zscore)
        .fillna(0.0)
    )

    # --- near_threshold_flag ---
    df["near_threshold_flag"] = df["amount"].apply(_near_threshold).astype(int)

    # --- time_to_award_days ---
    delta = (df["date"] - df["tender_date"]).dt.days
    df["time_to_award_days"] = delta.clip(lower=0).fillna(0).astype(int)
    df["time_to_award_days"] = df["time_to_award_days"].clip(upper=3650)

    features = (
        df[
            [
                "award_id",
                "supplier_id",  # <-- for graph join
                "award_concentration_by_buyer",
                "repeat_winner_ratio",
                "amount_zscore_by_category",
                "near_threshold_flag",
                "time_to_award_days",
            ]
        ]
        .drop_duplicates(subset=["award_id"])
        .reset_index(drop=True)
    )

    return features


def save_features(features: pd.DataFrame, out_path: str) -> str:
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    features.to_parquet(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Build contract-level feature seeds from curated OCDS parquet."
    )
    ap.add_argument("--tenders", type=str, default="data/curated/ocds/tenders.parquet")
    ap.add_argument("--awards", type=str, default="data/curated/ocds/awards.parquet")
    ap.add_argument(
        "--out", type=str, default="data/feature_store/contracts_features.parquet"
    )
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    feats = build_features(args.tenders, args.awards)

    if args.print_only:
        print(feats.head().to_markdown(index=False))
        print(f"Rows: {len(feats)}")
        return

    path = save_features(feats, args.out)
    print(f"Wrote features: {path}  (rows={len(feats)})")


if __name__ == "__main__":
    main()
