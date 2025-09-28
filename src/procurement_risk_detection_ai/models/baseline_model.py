from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib  # type: ignore
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Default feature order used by training/inference (only those present are used at train)
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


# ----------------------------- I/O helpers -----------------------------


def _write_json(path: str, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ----------------------------- Feature utils -----------------------------


def select_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in feature_cols if c in df.columns]
    return df[cols].copy()


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """Safe column getter returning a Series with the DataFrame's index."""
    if name in df.columns:
        return df[name]
    # return constant series matching df index
    return pd.Series(default, index=df.index)


# ----------------------------- Proxy label (for baseline) -----------------------------


def make_proxy_label(df: pd.DataFrame) -> pd.Series:
    # Heuristic proxy label:
    # risky if: high z-score OR high repeat OR near-threshold OR high concentration
    z = _col(df, "amount_zscore_by_category").fillna(0).astype(float)
    rep = _col(df, "repeat_winner_ratio").fillna(0).astype(float)
    near = _col(df, "near_threshold_flag").fillna(0).astype(float)
    conc = _col(df, "award_concentration_by_buyer").fillna(0).astype(float)

    score = (
        (z > 1.5).astype(int)
        | (rep > 0.8).astype(int)
        | (near == 1).astype(int)
        | (conc > 0.7).astype(int)
    )
    return score.astype(int)


# ----------------------------- Train / Load -----------------------------


def fit_baseline(df: pd.DataFrame, feature_cols: List[str]) -> LogisticRegression:
    """Train logistic regression on proxy labels and persist model + meta."""
    X = select_features(df, feature_cols).to_numpy(dtype=float)
    y = make_proxy_label(df).to_numpy(dtype=int)

    clf = LogisticRegression(max_iter=200, n_jobs=None)
    clf.fit(X, y)

    # Persist model
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    # Compute training-set predictions for thresholds (robust fallback on error)
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

    # --- Optional calibration (keeps LR artifact type unchanged) ---
    # Gate with env to avoid surprising test changes
    calibrate = str(os.getenv("CALIBRATE_PROBS", "false")).lower() in (
        "1",
        "true",
        "yes",
        "y",
        "t",
    )
    calibrator_path = "models/baseline_calibrator.joblib"
    calibration_method = "sigmoid"  # Platt scaling

    if calibrate:
        try:
            from sklearn.calibration import CalibratedClassifierCV

            # Important: cv='prefit' so we wrap the already-fitted LR
            calibrator = CalibratedClassifierCV(
                base_estimator=clf, cv="prefit", method=calibration_method
            )
            calibrator.fit(X, y)

            joblib.dump(calibrator, calibrator_path)
            meta["calibrator_path"] = calibrator_path
            meta["calibration_method"] = calibration_method
        except Exception as e:
            # Non-fatal: keep training successful even if calibration fails
            meta["calibration_error"] = str(e)

    _write_json(META_PATH, meta)  # overwrite with calibrator info if any
    return clf


def load_model() -> Optional[LoadedModel]:
    """Load model + meta; returns None if either artifact is missing or invalid."""
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


def load_model_meta() -> Optional[Dict[str, Any]]:
    return _read_json(META_PATH)


# ----------------------------- Risk bands -----------------------------


def _compute_risk_band_thresholds(
    p: np.ndarray, q_low: float = 0.33, q_med: float = 0.66
) -> Dict[str, float]:
    q1 = float(np.quantile(p, q_low))
    q2 = float(np.quantile(p, q_med))
    return {"low_max": q1, "medium_max": q2}


def _band_from_score(score: float, thresholds: Dict[str, float]) -> str:
    if score <= thresholds.get("low_max", 0.33):
        return "low"
    if score <= thresholds.get("medium_max", 0.66):
        return "medium"
    return "high"


def band_from_score(score: float, thresholds: Optional[Dict[str, float]]) -> str:
    if not thresholds:
        thresholds = {"low_max": 0.33, "medium_max": 0.66}
    return _band_from_score(float(score), thresholds)


# ----------------------------- Inference + linear contributions -----------------------------


def _to_feature_vector(
    features: Union[pd.Series, Dict[str, Any], np.ndarray],
    feature_cols: List[str],
) -> np.ndarray:
    """Create a numeric vector aligned to feature_cols."""
    if isinstance(features, np.ndarray):
        vec = features.astype(float)
        if vec.shape[0] != len(feature_cols):
            raise ValueError("ndarray length does not match feature_cols length")
        return vec

    if isinstance(features, pd.DataFrame):
        # if a single-row DataFrame sneaks in, use the first row
        if len(features) == 0:
            raise TypeError("features DataFrame is empty")
        features = features.iloc[0]

    if isinstance(features, pd.Series):
        return np.array(
            [float(features.get(c, 0.0)) for c in feature_cols], dtype=float
        )

    if isinstance(features, dict):
        return np.array(
            [float(features.get(c, 0.0)) for c in feature_cols], dtype=float
        )

    raise TypeError("Unsupported features type; expected Series, dict, or ndarray")


def _predict_proba_internal(model: LogisticRegression, x: np.ndarray) -> float:
    coef = np.asarray(model.coef_[0], dtype=float)
    intercept = float(getattr(model, "intercept_", [0.0])[0])
    # align lengths if needed
    n = min(coef.shape[0], x.shape[0])
    coef = coef[:n]
    x = x[:n]
    logit = float(intercept + float(np.dot(coef, x)))
    prob = float(1.0 / (1.0 + np.exp(-logit)))
    return prob


def predict_proba_and_contrib(
    model: LogisticRegression,
    a: Union[List[str], pd.Series, Dict[str, Any], np.ndarray],
    b: Optional[Union[List[str], pd.Series, Dict[str, Any], np.ndarray]] = None,
) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Flexible signature to support both:
      - predict_proba_and_contrib(model, FEATURE_COLS_DEFAULT, row)
      - predict_proba_and_contrib(model, row, FEATURE_COLS_DEFAULT)

    Returns:
        prob: float in [0,1]
        contributions: List[(name, contribution)] sorted by |contribution| desc
    """
    # Disambiguate args by type
    if isinstance(a, list) and all(isinstance(x, str) for x in a):
        feature_cols = a
        features = b  # type: ignore
    else:
        features = a  # type: ignore
        feature_cols = b  # type: ignore

    if not isinstance(feature_cols, list) or not all(
        isinstance(x, str) for x in feature_cols
    ):
        raise TypeError("feature_cols must be a List[str]")

    x = _to_feature_vector(features, feature_cols)
    # Guard against coef length mismatch
    coef = np.asarray(model.coef_[0], dtype=float)
    n = min(len(coef), len(x))
    coef = coef[:n]
    x = x[:n]
    feature_cols = feature_cols[:n]

    prob = _predict_proba_internal(model, x)

    contribs: List[Tuple[str, float]] = []
    for name, value, w in zip(feature_cols, x, coef):
        contribs.append((name, float(w * value)))

    contribs.sort(key=lambda t: abs(t[1]), reverse=True)
    return prob, contribs


def predict_proba_and_contrib_linear_explanations(
    model: LogisticRegression,
    features: Union[pd.Series, Dict[str, Any], np.ndarray],
    feature_cols: List[str],
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper that returns the API-style payload:
    { "risk_score": p, "top_factors": [ {name, value, contribution}, ... ] }
    """
    x = _to_feature_vector(features, feature_cols)
    prob = _predict_proba_internal(model, x)
    coef = np.asarray(model.coef_[0], dtype=float)
    n = min(len(coef), len(x))
    coef = coef[:n]
    x = x[:n]
    feature_cols = feature_cols[:n]

    contribs: List[Dict[str, Any]] = []
    for name, value, w in zip(feature_cols, x, coef):
        contribs.append(
            {"name": name, "value": float(value), "contribution": float(w * value)}
        )

    contribs.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    if top_k is not None:
        contribs = contribs[: int(top_k)]
    return {"risk_score": prob, "top_factors": contribs}
