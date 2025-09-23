from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st


# ------------------------- Config -------------------------

API_URL_DEFAULT = os.getenv("API_URL", "http://127.0.0.1:8000")
PAGE_TITLE = "Datasets & Batch Scoring"


# ------------------------- Helpers -------------------------


@dataclass
class BatchOptions:
    join_graph: bool
    explain: bool
    limit_top_factors: int


def _api_post_batch(
    api_url: str,
    rows: List[Dict[str, Any]],
    opts: BatchOptions,
    use_envelope: bool = True,
    timeout_s: float = 60.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Calls /v1/score/batch and returns (items, meta).
    - items is always a list of dicts (preserves input order)
    - meta contains provenance_id/used_model when envelope is returned
    """
    params = {
        "join_graph": "true" if opts.join_graph else "false",
        "limit_top_factors": str(max(1, min(20, int(opts.limit_top_factors)))),
        "explain": "true" if opts.explain else "false",
    }

    url = f"{api_url.rstrip('/')}/v1/score/batch"
    try:
        body = {"items": rows} if use_envelope else rows
        resp = requests.post(url, params=params, json=body, timeout=timeout_s)
    except requests.RequestException as e:
        raise RuntimeError(f"Request failed: {e}")

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    if isinstance(data, dict) and "items" in data:
        return list(data.get("items") or []), {
            "provenance_id": data.get("provenance_id"),
            "used_model": data.get("used_model"),
        }
    elif isinstance(data, list):
        return data, {}
    else:
        raise RuntimeError("Unexpected response shape from /v1/score/batch")


def _split_success_errors(
    items: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows with 'error' from successful scored rows."""
    errs, ok = [], []
    for it in items:
        if it.get("error"):
            errs.append(it)
        else:
            ok.append(it)

    df_ok = (
        pd.DataFrame(ok)
        if ok
        else pd.DataFrame(
            columns=[
                "award_id",
                "supplier_id",
                "risk_score",
                "risk_band",
                "top_factors",
            ]
        )
    )
    df_err = (
        pd.DataFrame(errs)
        if errs
        else pd.DataFrame(columns=["award_id", "supplier_id", "error"])
    )
    return df_ok, df_err


def _download_csv(df: pd.DataFrame, filename: str) -> None:
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv",
        type="primary",
    )


def _coerce_award_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Ensure the minimal schema for the API: award_id (str), optional supplier_id."""
    out: List[Dict[str, Any]] = []
    if df.empty:
        return out

    # Normalize column names
    cols = {c.strip().lower(): c for c in df.columns}
    award_col = cols.get("award_id")
    supplier_col = cols.get("supplier_id")

    if not award_col:
        raise ValueError("CSV must include an 'award_id' column")

    for _, r in df.iterrows():
        a = r.get(award_col)
        s = r.get(supplier_col) if supplier_col else None
        if pd.isna(a):
            a = None
        if pd.isna(s):
            s = None
        out.append(
            {
                "award_id": str(a) if a is not None else None,
                "supplier_id": str(s) if s is not None else None,
            }
        )
    return out


def _factor_preview_cell(tf: Any, top_n: int = 3) -> str:
    if not isinstance(tf, list) or not tf:
        return ""
    head = tf[: max(0, top_n)]
    parts = []
    for e in head:
        name = e.get("name")
        val = e.get("value")
        contrib = e.get("contribution", None)
        if contrib is None:
            parts.append(f"{name}={val}")
        else:
            parts.append(f"{name}={val} ({contrib:+.3f})")
    return "; ".join(parts)


# ------------------------- UI -------------------------

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

with st.sidebar:
    st.subheader("API")
    api_url = st.text_input(
        "API URL", value=API_URL_DEFAULT, help="FastAPI server base URL"
    )

    st.subheader("Scoring options")
    join_graph = st.checkbox(
        "Join graph metrics",
        value=False,
        help="If available, joins supplier-level graph metrics",
    )
    explain = st.checkbox(
        "Explain (top factors)",
        value=True,
        help="If off, returns only scores & bands (faster)",
    )
    limit_top_factors = st.slider(
        "Top factors per item",
        min_value=1,
        max_value=20,
        value=5,
        help="Ignored when Explain is off",
    )

    st.caption(
        "Request will use: "
        f"`join_graph={str(join_graph).lower()}`, "
        f"`explain={str(explain).lower()}`, "
        f"`limit_top_factors={(limit_top_factors if explain else 0)}`"
    )

st.markdown(
    "Upload a CSV with at least an **`award_id`** column (optional `supplier_id`)."
)

uploaded = st.file_uploader("CSV file", type=["csv"])
sample_col1, sample_col2 = st.columns(2)
with sample_col1:
    if st.button("Show sample CSV"):
        st.code("award_id,supplier_id\nA1,SUP-1\nA2,SUP-2\n", language="csv")
with sample_col2:
    if st.button("Health check"):
        try:
            resp = requests.get(f"{api_url.rstrip('/')}/health", timeout=15.0)
            st.json(resp.json())
        except Exception as e:
            st.error(f"Health request failed: {e}")

st.divider()

if uploaded is not None:
    try:
        df_upload = pd.read_csv(uploaded)
        st.write("### Preview")
        st.dataframe(df_upload.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Options
    opt = BatchOptions(
        join_graph=join_graph, explain=explain, limit_top_factors=int(limit_top_factors)
    )

    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        use_envelope = st.checkbox(
            "Use envelope body",
            value=True,
            help='{"items": [...]} instead of plain list',
        )
    with colB:
        timeout_s = st.number_input(
            "Timeout (seconds)", min_value=5.0, max_value=300.0, value=60.0
        )

    st.divider()
    if st.button("Score batch", type="primary"):
        try:
            rows = _coerce_award_rows(df_upload)
        except Exception as e:
            st.error(str(e))
            st.stop()

        if not rows:
            st.warning("No input rows to score.")
            st.stop()

        with st.spinner("Scoring…"):
            try:
                items, meta = _api_post_batch(
                    api_url,
                    rows,
                    opt,
                    use_envelope=use_envelope,
                    timeout_s=float(timeout_s),
                )
            except Exception as e:
                st.error(str(e))
                st.stop()

        # Split and display
        df_ok, df_err = _split_success_errors(items)

        if df_err.shape[0] > 0:
            st.error(f"{df_err.shape[0]} row(s) returned errors")
            st.dataframe(df_err, use_container_width=True)
            _download_csv(df_err, "batch_errors.csv")

        if df_ok.shape[0] > 0:
            st.success(f"Scored {df_ok.shape[0]} row(s)")
            # Lightweight factor preview
            if "top_factors" in df_ok.columns:
                df_ok = df_ok.copy()
                df_ok["factors_preview"] = df_ok["top_factors"].apply(
                    lambda x: _factor_preview_cell(x, top_n=3)
                )
            st.dataframe(df_ok, use_container_width=True)

            # Band filter
            bands = sorted([b for b in df_ok["risk_band"].dropna().unique().tolist()])
            if bands:
                sel = st.multiselect(
                    "Filter by risk band", options=bands, default=bands
                )
                if sel:
                    st.dataframe(
                        df_ok[df_ok["risk_band"].isin(sel)], use_container_width=True
                    )

            # Top 20 bar chart by risk score
            if "risk_score" in df_ok.columns:
                chart_df = (
                    df_ok[["award_id", "risk_score"]]
                    .dropna()
                    .sort_values("risk_score", ascending=False)
                    .head(20)
                )
                chart_df = chart_df.set_index("award_id")
                st.bar_chart(chart_df)

            _download_csv(df_ok, "batch_scores.csv")

        if meta:
            st.caption("Response meta")
            st.json(meta)

else:
    st.info("Upload a CSV to begin. Need a template? Click **Show sample CSV** above.")
