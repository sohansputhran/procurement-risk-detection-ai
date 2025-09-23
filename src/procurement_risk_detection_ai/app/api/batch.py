from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import APIRouter, Body, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field


from procurement_risk_detection_ai.app.services.scoring import (
    is_model_available,
    score_row_with_explanations,
    score_batch_vectorized,  # NEW
)
from procurement_risk_detection_ai.app.api.provenance import log_provenance
from procurement_risk_detection_ai.models.baseline_model import (
    load_model_meta,
    band_from_score,
)

# ---- Pydantic v1/v2 compatible field validator alias ----
try:  # Pydantic v2
    from pydantic import field_validator  # type: ignore
except Exception:  # Pydantic v1 fallback
    from pydantic import validator as _v1_validator  # type: ignore

    def field_validator(field_name: str, *, mode: str = "after"):
        pre = mode == "before"

        def _decorator(fn):
            return _v1_validator(field_name, pre=pre, allow_reuse=True)(fn)

        return _decorator


# ---- Optional Prometheus metrics (no-op if lib missing) ----
_PROM_OK = True
try:
    from prometheus_client import Counter, Histogram, generate_latest

    REQ_TOTAL = Counter("api_requests_total", "API requests", ["endpoint", "status"])
    LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])
    ITEMS = Counter(
        "batch_scored_items_total",
        "Items scored in batch",
        ["used_model", "join_graph", "explain"],
    )
except Exception:  # pragma: no cover
    _PROM_OK = False

    class _Noop:
        def labels(self, *_, **__):
            return self

        def inc(self, *_):
            pass

        def observe(self, *_):
            pass

    REQ_TOTAL = LATENCY = ITEMS = _Noop()  # type: ignore

    def generate_latest():  # type: ignore
        return b"# metrics disabled\n"


router = APIRouter()

# ----------------------------- Env helpers (no settings import) -----------------------------


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "t")


def _env_int(name: str, default_val: int) -> int:
    try:
        return int(os.getenv(name, str(default_val)))
    except Exception:
        return default_val


def _env_bytes(name: str, default_val: int) -> int:
    try:
        return int(os.getenv(name, str(default_val)))
    except Exception:
        return default_val


def _paths() -> Dict[str, str]:
    return {
        "FEATURES_PATH": os.getenv(
            "FEATURES_PATH", "data/feature_store/contracts_features.parquet"
        ),
        "GRAPH_METRICS_PATH": os.getenv(
            "GRAPH_METRICS_PATH", "data/graph/metrics.parquet"
        ),
    }


def _limits() -> Dict[str, int]:
    return {
        "DEFAULT_TOP_K": _env_int("DEFAULT_TOP_K", 5),
        "MAX_BATCH_ITEMS": _env_int("MAX_BATCH_ITEMS", 1000),
    }


# ----------------------------- Pydantic models -----------------------------


class BatchItem(BaseModel):
    award_id: str = Field(..., description="Award id; must exist in features parquet")
    supplier_id: Optional[str] = None

    @field_validator("award_id")
    @classmethod
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


def _features_cache_disabled() -> bool:
    return _env_bool("FEATURES_CACHE_DISABLE", False)


def _load_parquet_cached(path: str) -> pd.DataFrame:
    if _features_cache_disabled():
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
    P = _paths()
    features_path = P["FEATURES_PATH"]
    graph_path = P["GRAPH_METRICS_PATH"]

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
        df_graph = _load_parquet(graph_path)
        if not df_graph.empty and "supplier_id" in df_graph.columns:
            df_graph = df_graph.groupby("supplier_id", as_index=False).first()
            df = df.merge(
                df_graph, how="left", on="supplier_id", suffixes=("", "_graph")
            )

    return df


# ----------------------------- /metrics -----------------------------


@router.get("/metrics")
def metrics():
    # Return Prometheus exposition format or 503 if disabled
    body = generate_latest()
    status = 200 if _PROM_OK else 503
    return PlainTextResponse(
        body, status_code=status, media_type="text/plain; version=0.0.4; charset=utf-8"
    )


# ----------------------------- Score route -----------------------------


@router.post(
    "/v1/score/batch", response_model=Union[List[BatchResponseItem], BatchResponse]
)
def batch_score(
    request: Request,
    payload: Any = Body(...),
    join_graph: bool = Query(
        False, description="Join supplier graph metrics if available."
    ),
    # Default None -> resolved from env per-request; keeps tests deterministic
    limit_top_factors: Optional[int] = Query(
        None, ge=1, le=20, description="Max number of explanation factors to include."
    ),
    explain: bool = Query(
        True, description="If false, skip per-item top_factors generation."
    ),
):
    """
    Accepts:
      1) {"items":[{...}]}  → returns an envelope {items, provenance_id, used_model}
      2) plain list [{...}] → returns a plain list
    Preserves 1:1 input→output, annotating invalid rows with `error`.
    """
    started = time.time()
    endpoint_name = "score_batch"

    # Optional raw size guard via Content-Length (best-effort)
    max_bytes = _env_bytes("MAX_REQUEST_BYTES", 1_048_576)  # 1 MiB default
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > max_bytes:
        # Shape mirroring with errors
        input_was_list = isinstance(payload, list)
        items_raw = payload.get("items") if isinstance(payload, dict) else payload
        if not isinstance(items_raw, list):
            items_raw = []
        out_items = [
            {
                "award_id": (
                    (it or {}).get("award_id") if isinstance(it, dict) else None
                ),
                "supplier_id": (
                    (it or {}).get("supplier_id") if isinstance(it, dict) else None
                ),
                "risk_score": None,
                "risk_band": None,
                "top_factors": [],
                "error": "request too large",
            }
            for it in items_raw
        ]
        prov_id = log_provenance(
            endpoint="/v1/score/batch",
            payload={"_too_large": True},
            started_at=started,
            status="error",
            error="request too large",
            num_items=len(out_items),
        )
        REQ_TOTAL.labels(endpoint=endpoint_name, status="413").inc()
        LATENCY.labels(endpoint=endpoint_name).observe(time.time() - started)
        return (
            out_items
            if input_was_list
            else BatchResponse(
                items=out_items, provenance_id=prov_id, used_model=is_model_available()
            )
        )

    L = _limits()
    if limit_top_factors is None:
        limit_top_factors = L["DEFAULT_TOP_K"]
    effective_top_k = 0 if not explain else int(limit_top_factors)

    # Normalize input and remember original shape for mirroring
    input_was_list = isinstance(payload, list)
    items_raw = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items_raw, list):
        items_raw = []

    # Soft guard for very large batches
    if len(items_raw) > L["MAX_BATCH_ITEMS"]:
        items_raw = items_raw[: L["MAX_BATCH_ITEMS"]]

    # Validate each row but keep 1:1 cardinality
    validated: List[Dict[str, Any]] = []
    errors_by_idx: Dict[int, str] = {}
    for i, obj in enumerate(items_raw):
        try:
            model = BatchItem.parse_obj(obj if isinstance(obj, dict) else {})
            validated.append(
                {"award_id": model.award_id, "supplier_id": model.supplier_id}
            )
        except Exception as ve:
            # Pydantic v1/v2: stringify error
            msg = str(ve)
            validated.append(
                {
                    "award_id": (
                        (obj or {}).get("award_id") if isinstance(obj, dict) else None
                    )
                }
            )
            errors_by_idx[i] = msg

    # If everything invalid, return per-item errors (shape mirrored)
    if len(errors_by_idx) == len(validated) and len(validated) > 0:
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
        REQ_TOTAL.labels(endpoint=endpoint_name, status="400").inc()
        LATENCY.labels(endpoint=endpoint_name).observe(time.time() - started)
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
        REQ_TOTAL.labels(endpoint=endpoint_name, status="404").inc()
        LATENCY.labels(endpoint=endpoint_name).observe(time.time() - started)
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

    # ----------- Fast vectorized path when model available & explain==False -----------
    if is_model_available() and not explain:
        used_model_flag = True
        # Vectorized probability prediction (optionally calibrated if available)
        probs = score_batch_vectorized(df_joined)
        for (_, row), p in zip(df_joined.iterrows(), probs):
            rb = band_from_score(float(p), thresholds)
            out_items.append(
                {
                    "award_id": row.get("award_id"),
                    "supplier_id": row.get("supplier_id"),
                    "risk_score": float(p),
                    "risk_band": rb,
                    "top_factors": [],
                }
            )

    else:
        # ----------- Existing paths (model + explanations OR heuristic) -----------
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
                scored = score_row_with_explanations(row, top_k=int(effective_top_k))
                rb = band_from_score(scored["risk_score"], thresholds)
                out_items.append(
                    {
                        "award_id": row.get("award_id"),
                        "supplier_id": row.get("supplier_id"),
                        "risk_score": scored["risk_score"],
                        "risk_band": rb,
                        "top_factors": scored["top_factors"] if explain else [],
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
                    min(
                        1.0, float(row.get("amount_zscore_by_category", 0) or 0.0) / 3.0
                    ),
                )
                score += 0.1 * max(
                    0.0,
                    min(1.0, float(row.get("award_concentration_by_buyer", 0) or 0.0)),
                )
                score = max(0.0, min(1.0, score))
                tf = (
                    []
                    if not explain
                    else [
                        {"name": name, "value": row.get(name, None)}
                        for name in factor_candidates[: int(effective_top_k)]
                    ]
                )
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

    # Metrics
    REQ_TOTAL.labels(endpoint=endpoint_name, status="200").inc()
    LATENCY.labels(endpoint=endpoint_name).observe(time.time() - started)
    ITEMS.labels(
        used_model=str(used_model_flag).lower(),
        join_graph=str(join_graph).lower(),
        explain=str(explain).lower(),
    ).inc(len(out_items))

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


# ----------------------------- Validate-only route -----------------------------


class ValidateResponseItem(BaseModel):
    award_id: Optional[str] = None
    supplier_id: Optional[str] = None
    valid: bool = False
    error: Optional[str] = None


class ValidateResponse(BaseModel):
    items: List[ValidateResponseItem]
    provenance_id: Optional[str] = None


@router.post(
    "/v1/score/validate",
    response_model=Union[List[ValidateResponseItem], ValidateResponse],
)
def batch_validate(payload: Any = Body(...)):
    """
    Validate-only: checks schema (award_id non-empty) and that award_id exists in FEATURES_PATH.
    Mirrors input shape (list→list, envelope→envelope). Does NOT score.
    """
    started = time.time()
    # Normalize input and remember original shape for mirroring
    input_was_list = isinstance(payload, list)
    items_raw = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items_raw, list):
        items_raw = []

    # Schema validation (same behavior as batch_score)
    validated: List[Dict[str, Any]] = []
    errors_by_idx: Dict[int, str] = {}
    for i, obj in enumerate(items_raw):
        try:
            model = BatchItem.parse_obj(obj if isinstance(obj, dict) else {})
            validated.append(
                {"award_id": model.award_id, "supplier_id": model.supplier_id}
            )
        except Exception as ve:
            validated.append(
                {
                    "award_id": (
                        (obj or {}).get("award_id") if isinstance(obj, dict) else None
                    ),
                    "supplier_id": (
                        (obj or {}).get("supplier_id")
                        if isinstance(obj, dict)
                        else None
                    ),
                }
            )
            errors_by_idx[i] = str(ve)

    # Feature existence check
    P = _paths()
    df_feat = _load_parquet_cached(P["FEATURES_PATH"])
    existing_ids = set()
    if not df_feat.empty and "award_id" in df_feat.columns:
        try:
            existing_ids = set(df_feat["award_id"].astype(str).dropna().tolist())
        except Exception:
            pass

    out_items: List[Dict[str, Any]] = []
    for idx, row in enumerate(validated):
        err = errors_by_idx.get(idx)
        aid = row.get("award_id")
        if err:
            out_items.append(
                {
                    "award_id": aid,
                    "supplier_id": row.get("supplier_id"),
                    "valid": False,
                    "error": err,
                }
            )
            continue
        if not existing_ids:
            out_items.append(
                {
                    "award_id": aid,
                    "supplier_id": row.get("supplier_id"),
                    "valid": False,
                    "error": "no features found",
                }
            )
        elif str(aid) not in existing_ids:
            out_items.append(
                {
                    "award_id": aid,
                    "supplier_id": row.get("supplier_id"),
                    "valid": False,
                    "error": "award_id not found in features",
                }
            )
        else:
            out_items.append(
                {
                    "award_id": aid,
                    "supplier_id": row.get("supplier_id"),
                    "valid": True,
                    "error": None,
                }
            )

    prov_id = log_provenance(
        endpoint="/v1/score/validate",
        payload=payload,
        started_at=started,
        status="ok",
        num_items=len(out_items),
    )
    REQ_TOTAL.labels(endpoint="score_validate", status="200").inc()
    LATENCY.labels(endpoint="score_validate").observe(time.time() - started)
    return (
        out_items
        if input_was_list
        else ValidateResponse(items=out_items, provenance_id=prov_id)
    )
