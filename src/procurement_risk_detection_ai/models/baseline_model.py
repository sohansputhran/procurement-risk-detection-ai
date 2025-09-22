from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


MODEL_DIR = Path("models_data")
MODEL_PATH = MODEL_DIR / "baseline_logreg.joblib"
META_PATH = MODEL_DIR / "baseline_logreg.meta.json"

FEATURE_COLS_DEFAULT = [
    "award_concentration_by_buyer",
    "repeat_winner_ratio",
    "amount_zscore_by_category",
    "near_threshold_flag",
    "time_to_award_days",
]


@dataclass
class LoadedModel:
    model: LogisticRegression
    feature_cols: List[str]
    intercept: float


def ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model: LogisticRegression, feature_cols: List[str]) -> None:
    ensure_model_dir()
    joblib.dump(model, MODEL_PATH)
    meta = {
        "feature_cols": feature_cols,
        "intercept": float(model.intercept_[0]),
        "coef": model.coef_[0].tolist(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_model() -> LoadedModel | None:
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    feature_cols = meta["feature_cols"]
    intercept = float(meta["intercept"])
    return LoadedModel(model=model, feature_cols=feature_cols, intercept=intercept)


def make_proxy_label(df: pd.DataFrame) -> pd.Series:
    nt = (df.get("near_threshold_flag", 0) == 1).astype(int)
    az = (df.get("amount_zscore_by_category", 0) > 2.0).astype(int)
    rw = (df.get("repeat_winner_ratio", 0) > 0.8).astype(int)
    ac = (df.get("award_concentration_by_buyer", 0) > 0.6).astype(int)
    y = (nt | az | rw | ac).astype(int)
    return y


def select_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df.reindex(columns=feature_cols, fill_value=0)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def fit_baseline(
    df_features: pd.DataFrame, feature_cols: List[str] | None = None
) -> LogisticRegression:
    if feature_cols is None:
        feature_cols = FEATURE_COLS_DEFAULT
    X = select_features(df_features, feature_cols).values
    y = make_proxy_label(df_features).values
    if y.sum() == 0:  # avoid degenerate labels
        y = (
            (df_features.get("amount_zscore_by_category", 0).rank(pct=True) > 0.9)
            .astype(int)
            .values
        )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    save_model(clf, feature_cols)
    return clf  # <-- critical fix


def predict_proba_and_contrib(
    model: LogisticRegression, feature_cols: List[str], row: pd.Series
) -> Tuple[float, List[Tuple[str, float, float]]]:
    x = select_features(pd.DataFrame([row]), feature_cols).iloc[0].values.astype(float)
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    logit = float(intercept + np.dot(coef, x))
    prob = float(1.0 / (1.0 + np.exp(-logit)))
    contribs = [(f, float(v), float(c * v)) for f, v, c in zip(feature_cols, x, coef)]
    contribs.sort(key=lambda t: abs(t[2]), reverse=True)
    return prob, contribs
