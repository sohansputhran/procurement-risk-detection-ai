from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# Local baseline helpers
from procurement_risk_detection_ai.models import baseline_model as bm


# ----------------------------- Paths & availability -----------------------------


def _model_path() -> str:
    return os.getenv("MODEL_PATH", "models/baseline_logreg.joblib")


def _meta_path() -> str:
    return os.getenv("MODEL_META_PATH", "models/baseline_logreg_meta.json")


def _calibrator_path(meta: Optional[Dict[str, Any]] = None) -> str:
    # meta can override via 'calibrator_path'; env can override both
    envp = os.getenv("MODEL_CALIBRATOR_PATH")
    if envp:
        return envp
    if meta and isinstance(meta.get("calibrator_path"), str):
        return str(meta["calibrator_path"])
    return "models/baseline_calibrator.joblib"


def is_model_available() -> bool:
    try:
        import os

        return os.path.exists(_model_path())
    except Exception:
        return False


# ----------------------------- Per-row scoring with explanations -----------------------------


def score_row_with_explanations(row: pd.Series, top_k: int = 5) -> Dict[str, Any]:
    """
    Returns:
      {
        "risk_score": float in [0,1],
        "top_factors": [{"name": str, "value": Any, "contribution": float}, ...][:top_k]
      }
    """
    meta = bm.load_model_meta() or {}
    feature_cols = meta.get("feature_cols") or bm.FEATURE_COLS_DEFAULT
    # baseline_model provides explanation-ready function for linear model
    prob, contribs = bm.predict_proba_and_contrib_linear_explanations(
        row, feature_cols=feature_cols
    )
    # contribs is a list of dicts sorted by |contribution|
    return {
        "risk_score": float(prob),
        "top_factors": contribs[: max(0, int(top_k))] if top_k else [],
    }


# ----------------------------- Vectorized scoring for batches -----------------------------


def _prepare_matrix(df: pd.DataFrame, feature_cols: Iterable[str]) -> np.ndarray:
    X = df.reindex(columns=list(feature_cols), fill_value=np.nan)
    X = (
        X.apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float, copy=False)
    )
    return X


def score_batch_vectorized(df_joined: pd.DataFrame) -> List[float]:
    """
    Fast path used when explain=False:
    - Loads meta → feature_cols
    - Uses either calibrator (if present) or raw LogisticRegression to predict_proba
    - Returns a list of probabilities aligned to df_joined order
    """
    import joblib  # local import to avoid hard dep if unused

    meta = bm.load_model_meta() or {}
    feature_cols = meta.get("feature_cols") or bm.FEATURE_COLS_DEFAULT

    # Build matrix
    X = _prepare_matrix(df_joined, feature_cols)

    # Prefer calibrator if it exists (cv='prefit' saved wrapper)
    cal_path = _calibrator_path(meta)
    if os.path.exists(cal_path):
        calibrator = joblib.load(cal_path)
        probs = calibrator.predict_proba(X)[:, 1]
        return [float(max(0.0, min(1.0, p))) for p in probs]

    # Fallback to raw LR
    if not os.path.exists(_model_path()):
        # Should not happen if caller checks is_model_available
        return [0.0] * len(df_joined)

    lr = joblib.load(_model_path())
    probs = lr.predict_proba(X)[:, 1]
    return [float(max(0.0, min(1.0, p))) for p in probs]
