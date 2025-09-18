from __future__ import annotations

from typing import Any, Dict, List, Sequence
import pandas as pd


def _normalize_top_factors(top_factors: Any) -> List[Dict[str, Any]]:
    """
    Ensure top_factors is always a list[dict]. If missing/None/not-a-list, return [].
    Each factor is expected to have keys like {"name": ..., "value": ...} but we
    won't assume strict schema beyond being a mapping.
    """
    from collections.abc import Sequence as _Seq

    if not isinstance(top_factors, _Seq) or isinstance(top_factors, (str, bytes, dict)):
        return []
    out: List[Dict[str, Any]] = []
    for f in top_factors:
        if isinstance(f, dict):
            out.append(f)
    return out


def results_json_to_dataframe(items: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert /v1/score/batch JSON response into a pandas DataFrame and flatten
    top_factors into deterministic columns:
      - top_factor_1_name, top_factor_1_value
      - top_factor_2_name, top_factor_2_value
      - ...
    The number of pairs is determined by the maximum length of top_factors across rows.
    Other keys (e.g., award_id, supplier_id, risk_score, …) are preserved.
    """
    if not isinstance(items, (list, tuple)):
        return pd.DataFrame()

    # First pass: figure out the max number of top_factors across rows
    max_k = 0
    normalized_rows: List[Dict[str, Any]] = []

    for row in items:
        row = dict(row) if isinstance(row, dict) else {}
        tfs = _normalize_top_factors(row.get("top_factors"))
        max_k = max(max_k, len(tfs))
        row["_top_factors_list"] = tfs  # stash for second pass
        normalized_rows.append(row)

    # Second pass: create flattened columns
    flat_rows: List[Dict[str, Any]] = []
    for row in normalized_rows:
        out = {
            k: v
            for k, v in row.items()
            if k != "_top_factors_list" and k != "top_factors"
        }

        tfs = row.get("_top_factors_list", [])
        # Emit name/value pairs up to max_k; if fewer factors present, leave as None
        for i in range(max_k):
            name_key = f"top_factor_{i+1}_name"
            value_key = f"top_factor_{i+1}_value"
            if i < len(tfs):
                fdict = tfs[i] or {}
                out[name_key] = fdict.get("name")
                out[value_key] = fdict.get("value")
            else:
                out[name_key] = None
                out[value_key] = None

        flat_rows.append(out)

    df = pd.DataFrame(flat_rows)
    # Make CSV-friendly: no NaN/Inf to keep consistency with API's JSON-safe policy
    return df.replace({pd.NA: None}).fillna("")


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert DataFrame to CSV bytes with UTF-8 BOM for Excel friendliness on Windows.
    """
    return df.to_csv(index=False).encode("utf-8-sig")
