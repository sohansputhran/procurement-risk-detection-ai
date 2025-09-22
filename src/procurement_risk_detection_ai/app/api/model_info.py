from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

MODELS_DIR = os.getenv("MODELS_DIR", "models")
MODEL_PATH = os.getenv("MODEL_PATH", f"{MODELS_DIR}/baseline_logreg.joblib")
META_PATH = os.getenv("MODEL_META_PATH", f"{MODELS_DIR}/baseline_logreg_meta.json")
METRICS_DIR = os.getenv("METRICS_DIR", "reports/metrics")


@dataclass
class _TopWeight:
    name: str
    weight: float
    abs_weight: float


class ModelWeights(BaseModel):
    n_features: Optional[int] = None
    top_weights: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top absolute coefficient weights with feature names.",
    )


class EvaluationInfo(BaseModel):
    path: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    available: bool
    model_path: Optional[str] = None
    meta_path: Optional[str] = None
    trained_at: Optional[str] = None
    feature_cols: Optional[List[str]] = None
    class_weight: Optional[Dict[str, float]] = None
    weights: ModelWeights = Field(default_factory=ModelWeights)
    evaluation: Optional[EvaluationInfo] = None


def _read_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_metrics_path() -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(METRICS_DIR, "baseline_metrics_*.json")))
    return paths[-1] if paths else None


def _compute_top_weights(
    model_path: str, feature_cols: Optional[List[str]]
) -> ModelWeights:
    # Return empty if we can’t compute safely
    try:
        import joblib  # type: ignore
    except Exception:
        return ModelWeights()

    p = Path(model_path)
    if not p.exists():
        return ModelWeights()

    try:
        lr = joblib.load(str(p))
    except Exception:
        return ModelWeights()

    # Expect scikit LogisticRegression with shape (1, n_features)
    coef = getattr(lr, "coef_", None)
    if coef is None:
        return ModelWeights()

    coef = coef[0]
    n_features = len(coef)
    names = (
        feature_cols
        if feature_cols and len(feature_cols) == n_features
        else [f"f{i}" for i in range(n_features)]
    )

    weights = [
        _TopWeight(name=names[i], weight=float(coef[i]), abs_weight=float(abs(coef[i])))
        for i in range(n_features)
    ]
    weights.sort(key=lambda w: w.abs_weight, reverse=True)
    top = [asdict(w) for w in weights[: min(10, len(weights))]]

    return ModelWeights(n_features=n_features, top_weights=top)


@router.get("/v1/model/info", response_model=ModelInfo)
def get_model_info() -> ModelInfo:
    model_exists = Path(MODEL_PATH).exists()
    meta = _read_json_if_exists(META_PATH) or {}

    feature_cols: Optional[List[str]] = meta.get("feature_cols")
    trained_at: Optional[str] = meta.get("trained_at") or meta.get("timestamp")
    class_weight = meta.get("class_weight")

    weights = _compute_top_weights(MODEL_PATH, feature_cols)

    # Latest evaluation metrics (if any)
    eval_path = _latest_metrics_path()
    eval_metrics = _read_json_if_exists(eval_path) if eval_path else None
    evaluation = (
        EvaluationInfo(path=eval_path, metrics=eval_metrics or {})
        if eval_path
        else None
    )

    return ModelInfo(
        available=bool(model_exists),
        model_path=MODEL_PATH if model_exists else None,
        meta_path=META_PATH if Path(META_PATH).exists() else None,
        trained_at=trained_at,
        feature_cols=feature_cols,
        class_weight=class_weight,
        weights=weights,
        evaluation=evaluation,
    )
