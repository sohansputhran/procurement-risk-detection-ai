from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd

from procurement_risk_detection_ai.models.baseline_model import (
    load_model,
    predict_proba_and_contrib,
)

_LOADED = load_model()


def is_model_available() -> bool:
    return _LOADED is not None


def score_row_with_explanations(row: pd.Series, top_k: int = 5) -> Dict[str, Any]:
    if _LOADED is None:
        raise RuntimeError("Model not loaded")
    prob, contribs = predict_proba_and_contrib(_LOADED.model, _LOADED.feature_cols, row)
    top = []
    for name, value, c in contribs[:top_k]:
        top.append({"name": name, "value": value, "contribution": c})
    return {"risk_score": round(prob, 6), "top_factors": top}


def get_feature_columns() -> List[str]:
    if _LOADED is None:
        return []
    return list(_LOADED.feature_cols)
