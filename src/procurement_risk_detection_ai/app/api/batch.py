from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import APIRouter, Body, Query
from pydantic import BaseModel, Field

from procurement_risk_detection_ai.app.services.scoring import (
    is_model_available,
    score_row_with_explanations,
)
from procurement_risk_detection_ai.app.api.provenance import log_provenance

router = APIRouter()


# ----------------------------- Pydantic models -----------------------------


class BatchItem(BaseModel):
    award_id: str = Field(..., description="Award id; must exist in features parquet")
    supplier_id: Optional[str] = None


class BatchRequest(BaseModel):
    items: List[BatchItem]


class BatchResponseItem(BaseModel):
    award_id: str
    supplier_id: Optional[str] = None
    risk_score: float
    top_factors: List[Dict[str, Any]]


class BatchResponse(BaseModel):
    items: List[BatchResponseItem]
    provenance_id: Optional[str] = None
    used_model: bool = True


# ----------------------------- IO helpers -----------------------------


def _load_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


# Cache for the HOT features parquet to avoid re-reading on each request
_FEATURES_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "df": None}


def _load_parquet_cached(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    mtime = p.stat().st_mtime
    if _FEATURES_CACHE["path"] == str(p) and _FEATURES_CACHE["mtime"] == mtime:
        # Return a copy to avoid accidental mutation by downstream ops
        return _FEATURES_CACHE["df"].copy()
    df = pd.read_parquet(str(p))
    _FEATURES_CACHE.update({"path": str(p), "mtime": mtime, "df": df})
    return df.copy()


def _join_features(df_in: pd.DataFrame, join_graph: bool) -> pd.DataFrame:
    features_path = os.getenv(
        "FEATURES_PATH", "data/feature_store/contracts_features.parquet"
    )
    graph_path = os.getenv("GRAPH_METRICS_PATH", "data/graph/metrics.parquet")

    # HOT path uses cache
    df_feat = _load_parquet_cached(features_path)
    if df_feat.empty:
        return pd.DataFrame()
    if "award_id" not in df_feat.columns:
        raise RuntimeError("features parquet missing 'award_id' column")

    # left-join on award_id (includes our _input_row_id carried from df_in)
    df = df_in.merge(df_feat, how="left", on="award_id", suffixes=("", "_feat"))

    # fill supplier_id from features if missing
    if "supplier_id" in df.columns and "supplier_id_feat" in df.columns:
        df["supplier_id"] = (
            df["supplier_id"].fillna(df["supplier_id_feat"]).astype(object)
        )
        df = df.drop(columns=[c for c in ["supplier_id_feat"] if c in df.columns])

    # optional graph join; ensure one row per supplier_id
    if join_graph and os.path.exists(graph_path):
        df_graph = _load_parquet(graph_path)  # smaller/infrequent → no cache needed
        if not df_graph.empty and "supplier_id" in df_graph.columns:
            # enforce 1:1 supplier_id cardinality before merge
            df_graph = df_graph.groupby("supplier_id", as_index=False).first()
            df = df.merge(
                df_graph, how="left", on="supplier_id", suffixes=("", "_graph")
            )

    return df


# ----------------------------- Route -----------------------------


@router.post(
    "/v1/score/batch", response_model=Union[List[BatchResponseItem], BatchResponse]
)
def batch_score(
    payload: Any = Body(...),
    join_graph: bool = Query(
        False, description="Join supplier graph metrics if available."
    ),
    limit_top_factors: int = Query(
        5, ge=1, le=20, description="Max number of explanation factors to include."
    ),
):
    """
    Accepts either:
      1) {"items":[{...}]}  → returns an envelope {items, provenance_id, used_model}
      2) plain list [{...}] → returns a plain list

    Preserves 1:1 cardinality between input rows and outputs even if joins would fan out.
    """
    started = __import__("time").time()

    # Normalize input and remember original shape for mirroring
    input_was_list = isinstance(payload, list)
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        items = []

    # Tag input rows so we preserve 1:1 cardinality no matter what joins do
    df_in = pd.DataFrame(items)
    if df_in.empty or "award_id" not in df_in.columns:
        prov_id = log_provenance(
            endpoint="/v1/score/batch",
            payload=payload,
            started_at=started,
            status="error",
            error="missing award_id",
            num_items=0,
        )
        envelope = BatchResponse(
            items=[], provenance_id=prov_id, used_model=is_model_available()
        )
        return envelope.items if input_was_list else envelope

    df_in = df_in.copy()
    df_in["_input_row_id"] = range(len(df_in))  # stable row id, preserves input order

    # Join features (+ optional graph). This must carry _input_row_id through the merge.
    df_joined = _join_features(df_in, join_graph=join_graph)
    if df_joined.empty:
        prov_id = log_provenance(
            endpoint="/v1/score/batch",
            payload=payload,
            started_at=started,
            status="error",
            error="no features found",
            num_items=0,
        )
        envelope = BatchResponse(
            items=[], provenance_id=prov_id, used_model=is_model_available()
        )
        return envelope.items if input_was_list else envelope

    # --- ENFORCE 1:1 CARDINALITY ---
    # If joins created duplicates, collapse to the first row per input id.
    if "_input_row_id" in df_joined.columns:
        df_joined = (
            df_joined.sort_values("_input_row_id")
            .groupby("_input_row_id", as_index=False)
            .first()
        )

    out_items: List[Dict[str, Any]] = []
    used_model_flag = False

    if is_model_available():
        used_model_flag = True
        for _, row in df_joined.iterrows():
            scored = score_row_with_explanations(row, top_k=limit_top_factors)
            out_items.append(
                {
                    "award_id": row.get("award_id"),
                    "supplier_id": row.get("supplier_id"),
                    "risk_score": scored["risk_score"],
                    "top_factors": scored["top_factors"],
                }
            )
    else:
        # Fallback heuristic (kept consistent; trimmed to limit_top_factors)
        factor_candidates = [
            "repeat_winner_ratio",
            "amount_zscore_by_category",
            "award_concentration_by_buyer",
            "near_threshold_flag",
            "time_to_award_days",
        ]
        for _, row in df_joined.iterrows():
            score = 0.0
            score += 0.4 * float(row.get("near_threshold_flag", 0) == 1)
            score += 0.3 * float(row.get("repeat_winner_ratio", 0) or 0.0)
            score += 0.2 * max(
                0.0,
                min(1.0, float(row.get("amount_zscore_by_category", 0) or 0.0) / 3.0),
            )
            score += 0.1 * max(
                0.0, min(1.0, float(row.get("award_concentration_by_buyer", 0) or 0.0))
            )
            score = max(0.0, min(1.0, score))

            tf: List[Dict[str, Any]] = []
            for name in factor_candidates[:limit_top_factors]:
                tf.append({"name": name, "value": row.get(name, None)})

            out_items.append(
                {
                    "award_id": row.get("award_id"),
                    "supplier_id": row.get("supplier_id"),
                    "risk_score": round(score, 6),
                    "top_factors": tf,
                }
            )

    prov_id = log_provenance(
        endpoint="/v1/score/batch",
        payload=payload,
        started_at=started,
        status="ok",
        num_items=len(out_items),
    )

    # JSON-safe: normalize None/NaN
    for it in out_items:
        for k, v in list(it.items()):
            if isinstance(v, float) and pd.isna(v):
                it[k] = None

    # Mirror input shape in response
    if input_was_list:
        return out_items
    else:
        return BatchResponse(
            items=out_items, provenance_id=prov_id, used_model=used_model_flag
        )
