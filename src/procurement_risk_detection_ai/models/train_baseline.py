# procurement_risk_detection_ai/models/train_baseline.py
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd

from procurement_risk_detection_ai.models.baseline_model import (
    FEATURE_COLS_DEFAULT,
    fit_baseline,
    make_proxy_label,
    select_features,
    load_model,
)


def _read_parquet(path: str) -> pd.DataFrame:
    # Try pyarrow, then fastparquet, with friendly errors for Windows users
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception as e:
            raise RuntimeError(
                f"Could not read parquet at '{path}'. Install one of: 'pyarrow' or 'fastparquet'. Original error: {e}"
            )


def _read_features_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    if p.suffix.lower() in [".parquet", ".pq"]:
        return _read_parquet(str(p))
    if p.suffix.lower() == ".csv":
        return pd.read_csv(str(p))
    # Default to parquet if unknown; better error message if it fails
    return _read_parquet(str(p))


def _pick_feature_cols(df: pd.DataFrame, args_cols: List[str]) -> List[str]:
    if not args_cols or args_cols == ["auto"]:
        present = [c for c in FEATURE_COLS_DEFAULT if c in df.columns]
        return present
    return args_cols


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train baseline logistic regression on contracts features."
    )
    ap.add_argument(
        "--features",
        default=os.getenv(
            "FEATURES_PATH", "data/feature_store/contracts_features.parquet"
        ),
        help="Path to features parquet/csv (can also be set via FEATURES_PATH).",
    )
    ap.add_argument(
        "--features-cols",
        nargs="*",
        default=FEATURE_COLS_DEFAULT,
        help="Feature columns to use. Pass 'auto' to use those present in the file.",
    )
    args = ap.parse_args()

    print(f"[train] loading features: {args.features}")
    try:
        df = _read_features_any(args.features)
    except Exception as e:
        print(f"[train][error] {e}")
        return 2

    if df.empty:
        print("[train][error] features file loaded but contains 0 rows.")
        return 3

    feature_cols = _pick_feature_cols(df, args.features_cols)
    if not feature_cols:
        print(
            "[train][error] No usable feature columns found. "
            f"Expected overlap with: {FEATURE_COLS_DEFAULT}"
        )
        return 4

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[train][warn] Missing columns will be filled with 0: {missing}")

    print(
        f"[train] rows={len(df)}, using {len(feature_cols)} feature(s): {feature_cols}"
    )

    # Quick diagnostics: proxy label balance (helps sanity-check the baseline)
    try:
        y_proxy = make_proxy_label(df)
        pos = int(y_proxy.sum())
        neg = int((1 - y_proxy).sum())
        print(
            f"[train] proxy label balance -> pos={pos}, neg={neg} (total={len(y_proxy)})"
        )
    except Exception as e:
        print(f"[train][warn] could not compute proxy label balance: {e}")

    # Show a tiny feature sample (first row) for visibility
    try:
        x_sample = select_features(df.head(1), feature_cols)
        print("[train] sample feature row (head=1):")
        print(x_sample.to_string(index=False))
    except Exception:
        pass

    # Train + persist (fit_baseline also saves artifacts)
    try:
        fit_baseline(df, feature_cols)
    except Exception as e:
        print(f"[train][error] training failed: {e}")
        return 5

    # Verify artifacts and print quick model summary
    loaded = load_model()
    if loaded is None:
        print("[train][error] model artifacts not found after training.")
        return 6

    coef = getattr(loaded.model, "coef_", None)
    if coef is not None:
        coef_row = coef[0]
        pairs = list(zip(loaded.feature_cols, coef_row))
        pairs.sort(key=lambda t: abs(t[1]), reverse=True)
        top = ", ".join(f"{n}={w:+.4f}" for n, w in pairs[:5])
        print(f"[train] top weights: {top}")

    print("[train] saved model to models_data/baseline_logreg.joblib and meta json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
