from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    # Data paths
    FEATURES_PATH: str
    GRAPH_METRICS_PATH: str
    WB_INELIGIBLE_PATH: str
    OCDS_TENDERS_PATH: str
    OCDS_AWARDS_PATH: str

    # Models & reports
    MODELS_DIR: str
    MODEL_PATH: str
    MODEL_META_PATH: str
    METRICS_DIR: str

    # API behavior
    FEATURES_CACHE_DISABLE: bool
    DEFAULT_TOP_K: int
    MAX_BATCH_ITEMS: int


def _bool(env_name: str, default: bool = False) -> bool:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    models_dir = os.getenv("MODELS_DIR", "models")
    model_path = os.getenv("MODEL_PATH", f"{models_dir}/baseline_logreg.joblib")
    meta_path = os.getenv("MODEL_META_PATH", f"{models_dir}/baseline_logreg_meta.json")

    return Settings(
        FEATURES_PATH=os.getenv(
            "FEATURES_PATH", "data/feature_store/contracts_features.parquet"
        ),
        GRAPH_METRICS_PATH=os.getenv(
            "GRAPH_METRICS_PATH", "data/graph/metrics.parquet"
        ),
        WB_INELIGIBLE_PATH=os.getenv(
            "WB_INELIGIBLE_PATH", "data/curated/worldbank/ineligible.parquet"
        ),
        OCDS_TENDERS_PATH=os.getenv(
            "OCDS_TENDERS_PATH", "data/curated/ocds/tenders.parquet"
        ),
        OCDS_AWARDS_PATH=os.getenv(
            "OCDS_AWARDS_PATH", "data/curated/ocds/awards.parquet"
        ),
        MODELS_DIR=models_dir,
        MODEL_PATH=model_path,
        MODEL_META_PATH=meta_path,
        METRICS_DIR=os.getenv("METRICS_DIR", "reports/metrics"),
        FEATURES_CACHE_DISABLE=_bool("FEATURES_CACHE_DISABLE", False),
        DEFAULT_TOP_K=int(os.getenv("DEFAULT_TOP_K", "5")),
        MAX_BATCH_ITEMS=int(os.getenv("MAX_BATCH_ITEMS", "1000")),
    )
