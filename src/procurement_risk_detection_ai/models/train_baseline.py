from __future__ import annotations
import argparse
import os
import sys
import pandas as pd

from procurement_risk_detection_ai.models.baseline_model import (
    FEATURE_COLS_DEFAULT,
    fit_baseline,
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
    df = pd.read_parquet(args.features)
    print(f"[train] rows={len(df)}")

    fit_baseline(df, args.features_cols)
    print("[train] saved model to models/baseline_logreg.joblib and meta json")


if __name__ == "__main__":
    sys.exit(main())
