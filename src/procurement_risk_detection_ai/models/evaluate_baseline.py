# src/procurement_risk_detection_ai/models/evaluate_baseline.py
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

from procurement_risk_detection_ai.models.baseline_model import (
    FEATURE_COLS_DEFAULT,
    fit_baseline,
    load_model,
    make_proxy_label,
    select_features,
)


# ------------ Core evaluation (imported by tests) ------------


def evaluate(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    """
    Train (or load) the baseline and compute metrics using proxy labels.
    Returns a dict with standard classification metrics.
    """
    # Build proxy labels and feature matrix
    y = make_proxy_label(df).astype(int).to_numpy()
    X = select_features(df, feature_cols).to_numpy(dtype=float)

    # Load model if present; otherwise fit & persist
    loaded = load_model()
    clf = loaded.model if loaded is not None else None
    if clf is None:
        clf = fit_baseline(df, feature_cols)

    # Predict proba for positive class
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)[:, 1]
    else:
        logits = clf.decision_function(X)  # type: ignore[attr-defined]
        p = 1.0 / (1.0 + np.exp(-logits))

    # Avoid degenerate labels
    if y.max() == y.min():
        y = y.copy()
        y[0] = 1 - y[0]

    metrics = {
        "auc_roc": float(roc_auc_score(y, p)),
        "avg_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "pos_rate": float(y.mean()),
        "n": int(len(y)),
    }
    return metrics


# ------------ CLI helpers (file IO) ------------


def _read_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception as e:
            raise RuntimeError(
                f"Could not read parquet at '{path}'. Install 'pyarrow' or 'fastparquet'. Original error: {e}"
            )


def _read_features_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    if p.suffix.lower() in (".parquet", ".pq"):
        return _read_parquet(str(p))
    if p.suffix.lower() == ".csv":
        return pd.read_csv(str(p))
    return _read_parquet(str(p))


def _pick_feature_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    if not cols or cols == ["auto"]:
        return [c for c in FEATURE_COLS_DEFAULT if c in df.columns]
    return cols


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate baseline model on features.")
    ap.add_argument(
        "--features",
        default=os.getenv(
            "FEATURES_PATH", "data/feature_store/contracts_features.parquet"
        ),
        help="Path to features parquet/csv.",
    )
    ap.add_argument(
        "--features-cols",
        nargs="*",
        default=FEATURE_COLS_DEFAULT,
        help="Feature columns to use; pass 'auto' to infer present defaults.",
    )
    ap.add_argument(
        "--out-dir",
        default="reports/metrics",
        help="Directory to write metrics JSON artifact.",
    )
    args = ap.parse_args()

    df = _read_features_any(args.features)
    if df.empty:
        print("[eval][error] features file has 0 rows.")
        return 3

    feature_cols = _pick_feature_cols(df, args.features_cols)
    if not feature_cols:
        print("[eval][error] No usable feature columns found.")
        return 4

    m = evaluate(df, feature_cols)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out_dir) / f"baseline_metrics_{ts}.json"
    out_path.write_text(
        json.dumps({"feature_cols": feature_cols, **m}, indent=2), encoding="utf-8"
    )

    print(f"[eval] metrics written to {out_path}")
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
