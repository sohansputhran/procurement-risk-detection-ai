from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from procurement_risk_detection_ai.app.services.scoring import (
    is_model_available,
    score_row_with_explanations,
)
from procurement_risk_detection_ai.app.api.provenance import log_provenance

router = APIRouter()


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


def _load_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def _join_features(df_in: pd.DataFrame, join_graph: bool) -> pd.DataFrame:
    features_path = os.getenv(
        "FEATURES_PATH", "data/feature_store/contracts_features.parquet"
    )
    graph_path = os.getenv("GRAPH_METRICS_PATH", "data/graph/metrics.parquet")
    df_feat = _load_parquet(features_path)
    if df_feat.empty:
        return pd.DataFrame()

    if "award_id" not in df_feat.columns:
        raise RuntimeError("features parquet missing 'award_id' column")

    df = df_in.merge(df_feat, how="left", on="award_id", suffixes=("", "_feat"))

    if "supplier_id" in df.columns and "supplier_id_feat" in df.columns:
        df["supplier_id"] = (
            df["supplier_id"].fillna(df["supplier_id_feat"]).astype(object)
        )
        df = df.drop(columns=[c for c in ["supplier_id_feat"] if c in df.columns])

    if join_graph and os.path.exists(graph_path):
        df_graph = _load_parquet(graph_path)
        if not df_graph.empty and "supplier_id" in df_graph.columns:
            df = df.merge(
                df_graph, how="left", on="supplier_id", suffixes=("", "_graph")
            )

    return df


@router.post("/v1/score/batch", response_model=BatchResponse)
def batch_score(payload: Any, join_graph: bool = Query(False)):
    """
    Accepts either:
      1) {"items":[{...}]}
      2) plain list [{...}]
    """
    started = __import__("time").time()
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        items = []

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
        return BatchResponse(
            items=[], provenance_id=prov_id, used_model=is_model_available()
        )

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
        return BatchResponse(
            items=[], provenance_id=prov_id, used_model=is_model_available()
        )

    out_items: List[Dict[str, Any]] = []
    used_model_flag = False

    if is_model_available():
        used_model_flag = True
        for _, row in df_joined.iterrows():
            scored = score_row_with_explanations(row, top_k=5)
            out_items.append(
                {
                    "award_id": row.get("award_id"),
                    "supplier_id": row.get("supplier_id"),
                    "risk_score": scored["risk_score"],
                    "top_factors": scored["top_factors"],
                }
            )
    else:
        for _, row in df_joined.iterrows():
            score = 0.0
            score += 0.4 * float(row.get("near_threshold_flag", 0) == 1)
            score += 0.3 * (float(row.get("repeat_winner_ratio", 0)))
            score += 0.2 * max(
                0.0, min(1.0, float(row.get("amount_zscore_by_category", 0)) / 3.0)
            )
            score += 0.1 * max(
                0.0, min(1.0, float(row.get("award_concentration_by_buyer", 0)))
            )
            score = max(0.0, min(1.0, score))
            out_items.append(
                {
                    "award_id": row.get("award_id"),
                    "supplier_id": row.get("supplier_id"),
                    "risk_score": round(score, 6),
                    "top_factors": [
                        {
                            "name": "repeat_winner_ratio",
                            "value": row.get("repeat_winner_ratio", None),
                        },
                        {
                            "name": "amount_zscore_by_category",
                            "value": row.get("amount_zscore_by_category", None),
                        },
                    ],
                }
            )

    prov_id = log_provenance(
        endpoint="/v1/score/batch",
        payload=payload,
        started_at=started,
        status="ok",
        num_items=len(out_items),
    )

    for it in out_items:
        for k, v in list(it.items()):
            if v is None:
                it[k] = None

    return BatchResponse(
        items=out_items, provenance_id=prov_id, used_model=used_model_flag
    )
