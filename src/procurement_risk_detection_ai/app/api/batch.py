from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Query, Body
from pydantic import BaseModel, Field

from procurement_risk_detection_ai.app.services.scoring import (
    is_model_available,
    score_row_with_explanations,
)
from procurement_risk_detection_ai.app.api.provenance import log_provenance


_FEATURES_CACHE = {"path": None, "mtime": None, "df": None}

router = APIRouter()


# ----------------- IO MODELS -----------------


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


# ----------------- HELPERS -----------------

_FEATURES_CACHE = {"path": None, "mtime": None, "df": None}


def _load_parquet(path: str) -> pd.DataFrame:
    """Non-cached fallback (kept for graph file)."""
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_parquet_cached(path: str) -> pd.DataFrame:
    """Cache features parquet by (path, mtime) to avoid repeated disk reads."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    mtime = p.stat().st_mtime
    if _FEATURES_CACHE["path"] == str(p) and _FEATURES_CACHE["mtime"] == mtime:
        # Return a copy to avoid accidental mutation
        return _FEATURES_CACHE["df"].copy()
    df = pd.read_parquet(str(p))
    _FEATURES_CACHE.update({"path": str(p), "mtime": mtime, "df": df})
    return df.copy()


def _join_features(df_in: pd.DataFrame, join_graph: bool) -> pd.DataFrame:
    features_path = os.getenv(
        "FEATURES_PATH", "data/feature_store/contracts_features.parquet"
    )
    graph_path = os.getenv("GRAPH_METRICS_PATH", "data/graph/metrics.parquet")

    # Use cached load for features (hot path)
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
        # Graph file is smaller/infrequent; normal load is fine
        df_graph = _load_parquet(graph_path)
        if not df_graph.empty and "supplier_id" in df_graph.columns:
            df_graph = df_graph.groupby("supplier_id", as_index=False).first()
            df = df.merge(
                df_graph, how="left", on="supplier_id", suffixes=("", "_graph")
            )

    return df


def _normalize_payload(payload: Any) -> tuple[List[Dict[str, Any]], bool]:
    """
    Returns (items, list_in)
    - list_in=True  -> caller posted a plain list
    - list_in=False -> caller posted {"items":[...]}
    """
    if isinstance(payload, list):
        return payload, True
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return payload["items"], False
    return [], False


# ----------------- ENDPOINT -----------------


@router.post(
    "/v1/score/batch", response_model=Union[List[BatchResponseItem], BatchResponse]
)
def batch_score(payload: Any = Body(...), join_graph: bool = Query(False)):
    """
    Accepts either:
      1) plain list:       [{...}, {...}]              -> returns a plain list
      2) object envelope:  {"items":[{...}, {...}]}    -> returns an envelope with metadata
    """
    started = __import__("time").time()

    items, list_in = _normalize_payload(payload)
    df_in = pd.DataFrame(items)

    # Validate input awards
    if df_in.empty or "award_id" not in df_in.columns:
        prov_id = log_provenance(
            endpoint="/v1/score/batch",
            payload=payload,
            started_at=started,
            status="error",
            error="missing award_id",
            num_items=0,
        )
        # Mirror shape on error as well
        if list_in:
            return []
        return BatchResponse(
            items=[], provenance_id=prov_id, used_model=is_model_available()
        )

    # Stable input order + 1:1 cardinality guarantee
    df_in = df_in.copy()
    df_in["_input_row_id"] = range(len(df_in))

    # Join features (+ optional graph). Must carry _input_row_id through.
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
        if list_in:
            return []
        return BatchResponse(
            items=[], provenance_id=prov_id, used_model=is_model_available()
        )

    # Enforce 1 output row per input row even if joins create fanout
    if "_input_row_id" in df_joined.columns:
        df_joined = (
            df_joined.sort_values(["_input_row_id"])
            .groupby("_input_row_id", as_index=False)
            .first()
        )

    out_items: List[BatchResponseItem] = []
    used_model_flag = False

    if is_model_available():
        used_model_flag = True
        for _, row in df_joined.iterrows():
            scored = score_row_with_explanations(row, top_k=5)
            out_items.append(
                BatchResponseItem(
                    award_id=str(row.get("award_id")),
                    supplier_id=(
                        str(row.get("supplier_id"))
                        if pd.notna(row.get("supplier_id"))
                        else None
                    ),
                    risk_score=float(scored["risk_score"]),
                    top_factors=scored["top_factors"],
                )
            )
    else:
        # Fallback heuristic
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

            out_items.append(
                BatchResponseItem(
                    award_id=str(row.get("award_id")),
                    supplier_id=(
                        str(row.get("supplier_id"))
                        if pd.notna(row.get("supplier_id"))
                        else None
                    ),
                    risk_score=round(float(score), 6),
                    top_factors=[
                        {
                            "name": "repeat_winner_ratio",
                            "value": row.get("repeat_winner_ratio", None),
                        },
                        {
                            "name": "amount_zscore_by_category",
                            "value": row.get("amount_zscore_by_category", None),
                        },
                    ],
                )
            )

    prov_id = log_provenance(
        endpoint="/v1/score/batch",
        payload=payload,
        started_at=started,
        status="ok",
        num_items=len(out_items),
    )

    # Mirror caller shape:
    if list_in:
        # Return a plain list (so len(response) == len(input))
        return [it.model_dump() for it in out_items]

    # Return envelope with metadata
    return BatchResponse(
        items=out_items, provenance_id=prov_id, used_model=used_model_flag
    )
