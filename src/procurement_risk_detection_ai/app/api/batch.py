# src/procurement_risk_detection_ai/app/api/batch.py
from __future__ import annotations

import os
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# Defaults; read env at request time
DEFAULT_FEATURES_PATH = "data/feature_store/contracts_features.parquet"
DEFAULT_GRAPH_METRICS_PATH = "data/graph/metrics.parquet"


def _get_features_path() -> str:
    return os.environ.get("FEATURES_PATH", DEFAULT_FEATURES_PATH)


def _get_graph_metrics_path() -> str:
    return os.environ.get("GRAPH_METRICS_PATH", DEFAULT_GRAPH_METRICS_PATH)


# ---- Schemas
class BatchItem(BaseModel):
    award_id: str = Field(..., description="Award identifier, used to join features")
    amount: Optional[float] = Field(None, description="Optional amount override")
    supplier_name: Optional[str] = None
    buyer_name: Optional[str] = None


class BatchResponseItem(BaseModel):
    award_id: str
    risk_score: float
    top_factors: Dict[str, float]
    provenance: Dict[str, str]
    warnings: Optional[List[str]] = None


# ---- Utils
def _load_parquet(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def _norm_amount_z(z: float) -> float:
    z = _safe_float(z, 0.0)
    z = max(-3.0, min(3.0, z))
    return (z + 3.0) / 6.0


def _norm_time_days(days: float, cap: int = 365) -> float:
    v = _safe_float(days, 0.0)
    if v <= 0:
        return 0.0
    return min(v / cap, 1.0)


def _adjacency_from_distance(dist: Optional[float]) -> float:
    d = _safe_float(dist, 0.0)
    if d <= 0:
        return 1.0
    return max(0.0, 1.0 - 0.2 * d)


def _score_row(
    row: pd.Series, graph_row: Optional[pd.Series] = None
) -> Tuple[float, Dict[str, float]]:
    f = {
        "award_concentration_by_buyer": _safe_float(
            row.get("award_concentration_by_buyer"), 0.0
        ),
        "repeat_winner_ratio": _safe_float(row.get("repeat_winner_ratio"), 0.0),
        "amount_zscore_by_category_norm": _norm_amount_z(
            row.get("amount_zscore_by_category")
        ),
        "near_threshold_flag": _safe_float(row.get("near_threshold_flag"), 0.0),
        "time_to_award_days_norm": _norm_time_days(row.get("time_to_award_days")),
    }
    if graph_row is not None:
        f["adjacency_to_sanctioned"] = _adjacency_from_distance(
            graph_row.get("distance_to_sanctioned")
        )
    else:
        f["adjacency_to_sanctioned"] = 0.0

    w = {
        "award_concentration_by_buyer": 0.30,
        "repeat_winner_ratio": 0.10,
        "amount_zscore_by_category_norm": 0.20,
        "near_threshold_flag": 0.15,
        "time_to_award_days_norm": 0.10,
        "adjacency_to_sanctioned": 0.15,
    }

    contrib = {k: f[k] * w[k] for k in w.keys()}
    for k, v in list(contrib.items()):
        contrib[k] = 0.0 if (v is None or not math.isfinite(float(v))) else float(v)
    score = sum(contrib.values())
    score = 0.0 if (score is None or not math.isfinite(float(score))) else float(score)
    score = min(1.0, max(0.0, round(score, 6)))
    top = dict(sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:3])
    return score, top


@router.post("/v1/score/batch", response_model=List[BatchResponseItem])
def score_batch(items: List[BatchItem]):
    if not items:
        raise HTTPException(status_code=400, detail="Empty request payload.")

    # Load features
    features_path = _get_features_path()
    feats = _load_parquet(features_path)
    if feats is None or feats.empty or "award_id" not in feats.columns:
        raise HTTPException(
            status_code=503,
            detail=f"Features not available at '{features_path}'. Build features first.",
        )
    feats = feats.copy()
    feats["award_id"] = feats["award_id"].astype(str)

    # Load graph metrics
    graph_path = _get_graph_metrics_path()
    graph = _load_parquet(graph_path)

    # Build request df
    def _to_dict(x):
        return x.model_dump() if hasattr(x, "model_dump") else x.dict()

    df_in = pd.DataFrame([_to_dict(i) for i in items])
    df_in["award_id"] = df_in["award_id"].astype(str)

    # Merge features (award_id)
    merged = df_in.merge(feats, on="award_id", how="left", suffixes=("", "_feats"))

    if (
        graph is not None
        and not graph.empty
        and "supplier_id" in graph.columns
        and "supplier_id" in merged.columns
    ):
        graph = graph.copy()
        graph["supplier_id"] = graph["supplier_id"].astype(str)
        merged["supplier_id"] = merged["supplier_id"].astype(str)
        merged = merged.merge(
            graph, on="supplier_id", how="left", suffixes=("", "_graph")
        )

    responses: List[BatchResponseItem] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    features_rows = len(feats)

    for _, r in merged.iterrows():
        warnings: List[str] = []
        if pd.isna(r.get("award_concentration_by_buyer")):
            warnings.append("features_missing_for_award")

        # build "graph_row" view when distance column exists
        graph_row = r if "distance_to_sanctioned" in r.index else None
        score, top = _score_row(r, graph_row)

        prov = {
            "features": features_path,
            "features_rows": str(features_rows),
            "graph_metrics": (
                graph_path if (graph is not None and not graph.empty) else ""
            ),
            "ts": now,
        }

        responses.append(
            BatchResponseItem(
                award_id=str(r["award_id"]),
                risk_score=round(float(score), 4),
                top_factors={k: round(float(v), 4) for k, v in top.items()},
                provenance=prov,
                warnings=warnings or None,
            )
        )
    return responses
