from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ProvenanceRecord:
    # Identity
    request_id: str
    endpoint: str
    timestamp_utc: str
    duration_ms: Optional[int] = None

    # Client & host
    client_host: Optional[str] = None
    server_host: str = field(default_factory=lambda: socket.gethostname())

    # Inputs (safe/preview)
    payload_preview: Dict[str, Any] = field(default_factory=dict)
    num_items: Optional[int] = None  # for batch

    # Data sources (env-driven)
    features_path: Optional[str] = None
    graph_metrics_path: Optional[str] = None
    wb_ineligible_path: Optional[str] = None
    ocds_tenders_path: Optional[str] = None
    ocds_awards_path: Optional[str] = None

    # Output/Result meta
    status: str = "ok"
    error: Optional[str] = None


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    if v is None:
        return None
    return str(v)


def _preview_payload(payload: Any, max_chars: int = 4000) -> Dict[str, Any]:
    """
    Return a JSON-serializable preview of the payload with size guardrails.
    - For {"items":[...]}, keep up to first 3 items.
    - For dict: keep as-is but truncate long stringified repr overall.
    """
    try:
        if (
            isinstance(payload, dict)
            and "items" in payload
            and isinstance(payload["items"], list)
        ):
            preview = dict(payload)
            preview["items"] = payload["items"][:3]
        elif isinstance(payload, list):
            preview = payload[:3]
        else:
            preview = payload
        text = json.dumps(preview, default=str)
        if len(text) > max_chars:
            text = text[:max_chars] + "...<truncated>"
        return json.loads(text)
    except Exception:
        # Fall back to string
        return {"preview": str(payload)[:max_chars]}


def _append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def log_provenance(
    *,
    endpoint: str,
    payload: Any,
    started_at: float,
    status: str = "ok",
    error: Optional[str] = None,
    num_items: Optional[int] = None,
    client_host: Optional[str] = None,
) -> str:
    """
    Append one JSONL record to PROVENANCE_LOG_DIR/YYYY-MM-DD.jsonl and return request_id.
    Never raises: any logging failure is swallowed so it can't break API requests.
    """
    request_id = uuid.uuid4().hex
    try:
        log_dir = Path(_env("PROVENANCE_LOG_DIR", "data/logs/provenance"))
        ts = datetime.now(timezone.utc)
        record = ProvenanceRecord(
            request_id=request_id,
            endpoint=endpoint,
            timestamp_utc=ts.isoformat(),
            duration_ms=int((time.time() - started_at) * 1000),
            client_host=client_host,
            payload_preview=_preview_payload(payload),
            num_items=num_items,
            features_path=_env("FEATURES_PATH"),
            graph_metrics_path=_env("GRAPH_METRICS_PATH"),
            wb_ineligible_path=_env("WB_INELIGIBLE_PATH"),
            ocds_tenders_path=_env("OCDS_TENDERS_PATH"),
            ocds_awards_path=_env("OCDS_AWARDS_PATH"),
            status=status,
            error=error,
        )
        log_file = log_dir / f"{ts.date().isoformat()}.jsonl"
        _append_jsonl(log_file, asdict(record))
    except Exception:
        # Intentionally swallow any error — provenance must never take down the request.
        pass
    return request_id
