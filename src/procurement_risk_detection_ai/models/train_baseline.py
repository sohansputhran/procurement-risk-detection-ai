from __future__ import annotations
import argparse
import os
import sys
import pandas as pd

from procurement_risk_detection_ai.models.baseline_model import (
    FEATURE_COLS_DEFAULT,
    fit_baseline,
    load_model_meta,
)

from procurement_risk_detection_ai.models.evaluate_baseline import (
    evaluate as eval_metrics_func,
)


def _read_parquet_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception as e:
            raise RuntimeError(
                f"Could not read parquet at '{path}'. Install 'pyarrow' or 'fastparquet'. Original error: {e}"
            )


def main():
    ap = argparse.ArgumentParser(
        description="Train baseline logistic regression on contracts features."
    )
    ap.add_argument(
        "--features",
        default=os.getenv(
            "FEATURES_PATH", "data/feature_store/contracts_features.parquet"
        ),
    )
    ap.add_argument("--features-cols", nargs="*", default=FEATURE_COLS_DEFAULT)
    args = ap.parse_args()

    print(f"[train] loading features: {args.features}")
    df = _read_parquet_any(args.features)
    print(f"[train] rows={len(df)}")

    feature_cols = (
        args.features_cols
        if args.features_cols and args.features_cols != ["auto"]
        else [c for c in FEATURE_COLS_DEFAULT if c in df.columns]
    )
    if not feature_cols:
        print("[train][error] no usable feature columns found")
        return 4

    try:
        fit_baseline(df, feature_cols)
    except Exception as e:
        print(f"[train][error] training failed: {e}")
        return 2

    # Evaluate and append to meta (best-effort; does not fail training)
    try:
        m = eval_metrics_func(df, feature_cols)
        meta = load_model_meta() or {}
        meta["last_evaluation"] = m
        from pathlib import Path
        import json

        meta_path = os.getenv("MODEL_META_PATH", "models/baseline_logreg_meta.json")
        Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("[train] appended last_evaluation to meta JSON")
    except Exception as e:
        print(f"[train][warn] evaluation failed: {e}")

    print("[train] saved model to models/baseline_logreg.joblib and updated meta json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
