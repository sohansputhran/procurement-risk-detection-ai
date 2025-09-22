from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import APIRouter, Body, Query
from pydantic import BaseModel, Field, ValidationError, validator

from procurement_risk_detection_ai.app.services.scoring import (
    is_model_available,
    score_row_with_explanations,
)
from procurement_risk_detection_ai.app.api.provenance import log_provenance
from procurement_risk_detection_ai.models.baseline_model import (
    load_model_meta,
    band_from_score,
)

router = APIRouter()


# ----------------------------- Pydantic models -----------------------------


class BatchItem(BaseModel):
    award_id: str = Field(..., description="Award id; must exist in features parquet")
    supplier_id: Optional[str] = None

    @validator("award_id")
    def _non_empty(cls, v: str) -> str:
        if v is None or str(v).strip() == "":
            raise ValueError("award_id must be a non-empty string")
        return str(v)


class BatchRequest(BaseModel):
    items: List[BatchItem]


class BatchResponseItem(BaseModel):
    award_id: Optional[str] = None
    supplier_id: Optional[str] = None
    risk_score: Optional[float] = None
    risk_band: Optional[str] = None
    top_factors: List[Dict[str, Any]] = []
    error: Optional[str] = None


class BatchResponse(BaseModel):
    items: List[BatchResponseItem]
    provenance_id: Optional[str] = None
    used_model: bool = True


# ----------------------------- IO helpers & cache -----------------------------


def _load_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


_FEATURES_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "df": None}
_FEATURES_CACHE_DISABLED = os.getenv("FEATURES_CACHE_DISABLE", "false").lower() in (
    "1",
    "true",
    "yes",
)


def _load_parquet_cached(path: str) -> pd.DataFrame:
    if _FEATURES_CACHE_DISABLED:
        return _load_parquet(path)
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    mtime = p.stat().st_mtime
    if _FEATURES_CACHE["path"] == str(p) and _FEATURES_CACHE["mtime"] == mtime:
        return _FEATURES_CACHE["df"].copy()
    df = pd.read_parquet(str(p))
    _FEATURES_CACHE.update({"path": str(p), "mtime": mtime, "df": df})
    return df.copy()


def _join_features(df_in: pd.DataFrame, join_graph: bool) -> pd.DataFrame:
    features_path = os.getenv(
        "FEATURES_PATH", "data/feature_store/contracts_features.parquet"
    )
    graph_path = os.getenv("GRAPH_METRICS_PATH", "data/graph/metrics.parquet")

    df_feat = _load_parquet_cached(features_path)
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
    Accepts:
      1) {"items":[{...}]}  → returns an envelope {items, provenance_id, used_model}
      2) plain list [{...}] → returns a plain list
    Preserves 1:1 input→output, annotating invalid rows with `error`.
    """
    started = __import__("time").time()

    input_was_list = isinstance(payload, list)
    items_raw = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items_raw, list):
        items_raw = []

    # Validate each row but keep 1:1 cardinality
    validated: List[Dict[str, Any]] = []
    errors_by_idx: Dict[int, str] = {}
    for i, obj in enumerate(items_raw):
        try:
            # Accept only dicts; other types become validation errors
            model = BatchItem.parse_obj(obj if isinstance(obj, dict) else {})
            validated.append(
                {"award_id": model.award_id, "supplier_id": model.supplier_id}
            )
        except ValidationError as ve:
            validated.append(
                {
                    "award_id": (
                        (obj or {}).get("award_id") if isinstance(obj, dict) else None
                    )
                }
            )
            errors_by_idx[i] = "; ".join(err["msg"] for err in ve.errors())

    # If everything invalid, return per-item errors (shape mirrored)
    if len(errors_by_idx) == len(validated):
        out_items = [
            {
                "award_id": row.get("award_id"),
                "supplier_id": row.get("supplier_id"),
                "risk_score": None,
                "risk_band": None,
                "top_factors": [],
                "error": errors_by_idx.get(idx, "invalid input"),
            }
            for idx, row in enumerate(validated)
        ]
        prov_id = log_provenance(
            endpoint="/v1/score/batch",
            payload=payload,
            started_at=started,
            status="error",
            error="validation",
            num_items=len(out_items),
        )
        return (
            out_items
            if input_was_list
            else BatchResponse(
                items=out_items, provenance_id=prov_id, used_model=is_model_available()
            )
        )

    # Proceed with valid rows; carry _input_row_id to reassemble output order
    df_in = pd.DataFrame(validated)
    df_in["_input_row_id"] = range(len(df_in))
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
        # Fall back to per-item errors for the valid ones
        out_items = []
        for idx, row in enumerate(validated):
            msg = errors_by_idx.get(idx) or "no features found"
            out_items.append(
                {
                    "award_id": row.get("award_id"),
                    "supplier_id": row.get("supplier_id"),
                    "risk_score": None,
                    "risk_band": None,
                    "top_factors": [],
                    "error": msg,
                }
            )
        return (
            out_items
            if input_was_list
            else BatchResponse(
                items=out_items, provenance_id=prov_id, used_model=is_model_available()
            )
        )

    if "_input_row_id" in df_joined.columns:
        df_joined = (
            df_joined.sort_values("_input_row_id")
            .groupby("_input_row_id", as_index=False)
            .first()
        )

    # Load risk-band thresholds from model meta (fallbacks inside band_from_score)
    meta = load_model_meta() or {}
    thresholds = meta.get("risk_band_thresholds") or None

    out_items: List[Dict[str, Any]] = []
    used_model_flag = False

    if is_model_available():
        used_model_flag = True
        for idx, row in df_joined.iterrows():
            if idx in errors_by_idx:
                out_items.append(
                    {
                        "award_id": row.get("award_id"),
                        "supplier_id": row.get("supplier_id"),
                        "risk_score": None,
                        "risk_band": None,
                        "top_factors": [],
                        "error": errors_by_idx[idx],
                    }
                )
                continue
            scored = score_row_with_explanations(row, top_k=limit_top_factors)
            rb = band_from_score(scored["risk_score"], thresholds)
            out_items.append(
                {
                    "award_id": row.get("award_id"),
                    "supplier_id": row.get("supplier_id"),
                    "risk_score": scored["risk_score"],
                    "risk_band": rb,
                    "top_factors": scored["top_factors"],
                }
            )
    else:
        factor_candidates = [
            "repeat_winner_ratio",
            "amount_zscore_by_category",
            "award_concentration_by_buyer",
            "near_threshold_flag",
            "time_to_award_days",
        ]
        for idx, row in df_joined.iterrows():
            if idx in errors_by_idx:
                out_items.append(
                    {
                        "award_id": row.get("award_id"),
                        "supplier_id": row.get("supplier_id"),
                        "risk_score": None,
                        "risk_band": None,
                        "top_factors": [],
                        "error": errors_by_idx[idx],
                    }
                )
                continue
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
            tf = [
                {"name": name, "value": row.get(name, None)}
                for name in factor_candidates[:limit_top_factors]
            ]
            rb = band_from_score(score, thresholds)
            out_items.append(
                {
                    "award_id": row.get("award_id"),
                    "supplier_id": row.get("supplier_id"),
                    "risk_score": round(score, 6),
                    "risk_band": rb,
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

    # Clean NaNs
    for it in out_items:
        for k, v in list(it.items()):
            if isinstance(v, float) and pd.isna(v):
                it[k] = None

    return (
        out_items
        if input_was_list
        else BatchResponse(
            items=out_items, provenance_id=prov_id, used_model=used_model_flag
        )
    )
