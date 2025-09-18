# src/procurement_risk_detection_ai/app/api/health.py
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter
import pandas as pd

router = APIRouter()

DEFAULT_FEATURES_PATH = "data/feature_store/contracts_features.parquet"
DEFAULT_GRAPH_METRICS_PATH = "data/graph/metrics.parquet"
DEFAULT_WB_INELIGIBLE_PATH = "data/curated/worldbank/ineligible.parquet"
DEFAULT_OCDS_TENDERS_PATH = "data/curated/ocds/tenders.parquet"
DEFAULT_OCDS_AWARDS_PATH = "data/curated/ocds/awards.parquet"


def _get_path(env_key: str, default: str) -> str:
    return os.environ.get(env_key, default)


def _count_rows(path: str) -> Optional[int]:
    if not path or not os.path.exists(path):
        return None
    # Prefer pyarrow metadata to avoid loading the whole file
    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(path)
        return pf.metadata.num_rows
    except Exception:
        try:
            df = pd.read_parquet(path)
            return int(len(df))
        except Exception:
            return None


@router.get("/health")
def health():
    # Version (best-effort)
    try:
        from importlib.metadata import version

        api_version = version("procurement-risk-detection-ai")
    except Exception:
        api_version = "0.0.0-dev"

    features_path = _get_path("FEATURES_PATH", DEFAULT_FEATURES_PATH)
    graph_path = _get_path("GRAPH_METRICS_PATH", DEFAULT_GRAPH_METRICS_PATH)
    wb_inel = _get_path("WB_INELIGIBLE_PATH", DEFAULT_WB_INELIGIBLE_PATH)
    tenders_path = _get_path("OCDS_TENDERS_PATH", DEFAULT_OCDS_TENDERS_PATH)
    awards_path = _get_path("OCDS_AWARDS_PATH", DEFAULT_OCDS_AWARDS_PATH)

    datasets = {
        "features": {
            "path": features_path,
            "rows": _count_rows(features_path),
        },
        "graph_metrics": {
            "path": graph_path,
            "rows": _count_rows(graph_path),
        },
        "wb_ineligible": {
            "path": wb_inel,
            "rows": _count_rows(wb_inel),
        },
        "ocds_tenders": {
            "path": tenders_path,
            "rows": _count_rows(tenders_path),
        },
        "ocds_awards": {
            "path": awards_path,
            "rows": _count_rows(awards_path),
        },
    }

    # add availability flags
    for v in datasets.values():
        v["available"] = v["rows"] is not None

    return {
        "status": "ok",
        "api_version": api_version,
        "datasets": datasets,
    }
