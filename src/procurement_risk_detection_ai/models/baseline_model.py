from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib  # type: ignore
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURE_COLS_DEFAULT = [
    "award_concentration_by_buyer",
    "repeat_winner_ratio",
    "amount_zscore_by_category",
    "near_threshold_flag",
    "time_to_award_days",
]

MODELS_DIR = os.getenv("MODELS_DIR", "models_data")
MODEL_PATH = os.getenv("MODEL_PATH", f"{MODELS_DIR}/baseline_logreg.joblib")
META_PATH = os.getenv("MODEL_META_PATH", f"{MODELS_DIR}/baseline_logreg_meta.json")


@dataclass
class LoadedModel:
    model: LogisticRegression
    feature_cols: List[str]


def select_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in feature_cols if c in df.columns]
    return df[cols].copy()


def make_proxy_label(df: pd.DataFrame) -> pd.Series:
    # Heuristic proxy label
    z = (df.get("amount_zscore_by_category") or 0).fillna(0)
    rep = (df.get("repeat_winner_ratio") or 0).fillna(0)
    near = (df.get("near_threshold_flag") or 0).fillna(0)
    conc = (df.get("award_concentration_by_buyer") or 0).fillna(0)
    score = (
        (z > 1.5).astype(int)
        | (rep > 0.8).astype(int)
        | (near == 1).astype(int)
        | (conc > 0.7).astype(int)
    )
    return score.astype(int)


def _write_json(path: str, data: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_json(path: str) -> Optional[Dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _compute_risk_band_thresholds(
    p: np.ndarray, q_low: float = 0.33, q_med: float = 0.66
) -> Dict[str, float]:
    q1 = float(np.quantile(p, q_low))
    q2 = float(np.quantile(p, q_med))
    # low <= low_max, medium <= medium_max, else high
    return {"low_max": q1, "medium_max": q2}


def _band_from_score(score: float, thresholds: Dict[str, float]) -> str:
    if score <= thresholds.get("low_max", 0.33):
        return "low"
    if score <= thresholds.get("medium_max", 0.66):
        return "medium"
    return "high"


def fit_baseline(df: pd.DataFrame, feature_cols: List[str]) -> LogisticRegression:
    X = select_features(df, feature_cols).to_numpy(dtype=float)
    y = make_proxy_label(df).to_numpy(dtype=int)

    clf = LogisticRegression(max_iter=200, n_jobs=None)
    clf.fit(X, y)

    # Persist model
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    # Compute training-set predictions for thresholds
    try:
        p = clf.predict_proba(X)[:, 1]
        thresholds = _compute_risk_band_thresholds(p)
    except Exception:
        thresholds = {"low_max": 0.33, "medium_max": 0.66}

    meta = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "feature_cols": feature_cols,
        "class_weight": getattr(clf, "class_weight", None),
        "risk_band_thresholds": thresholds,
        "model_path": MODEL_PATH,
    }
    _write_json(META_PATH, meta)
    return clf


def load_model() -> Optional[LoadedModel]:
    p_model = Path(MODEL_PATH)
    p_meta = Path(META_PATH)
    if not p_model.exists() or not p_meta.exists():
        return None
    try:
        clf = joblib.load(str(p_model))
        meta = _read_json(str(p_meta)) or {}
        feature_cols = meta.get("feature_cols") or FEATURE_COLS_DEFAULT
        return LoadedModel(model=clf, feature_cols=feature_cols)
    except Exception:
        return None


def load_model_meta() -> Optional[Dict]:
    return _read_json(META_PATH)


def band_from_score(score: float, thresholds: Optional[Dict[str, float]]) -> str:
    if not thresholds:
        thresholds = {"low_max": 0.33, "medium_max": 0.66}
    return _band_from_score(float(score), thresholds)
