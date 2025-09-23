# app/ui/pages/1_Datasets.py
import os
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Datasets & Batch Scoring", layout="wide")

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.title("Datasets & Batch Scoring")

# --- Datasets card ---
st.subheader("Datasets status")
try:
    r = requests.get(f"{API_URL}/health", timeout=10)
    r.raise_for_status()
    health = r.json()
    ds = health.get("datasets", {})
    cols = st.columns(3)
    items = [
        ("Features", "features"),
        ("Graph metrics", "graph_metrics"),
        ("WB Ineligible", "wb_ineligible"),
        ("OCDS Tenders", "ocds_tenders"),
        ("OCDS Awards", "ocds_awards"),
    ]
    for i, (title, key) in enumerate(items):
        c = cols[i % 3]
        info = ds.get(key, {})
        ok = info.get("available", False)
        rows = info.get("rows", 0) if ok else 0
        path = info.get("path", "")
        c.metric(
            label=f"{title} {'✅' if ok else '❌'}", value=f"{rows:,}" if ok else "—"
        )
        if path:
            c.caption(path)
except Exception as e:
    st.error(f"Failed to query /health from {API_URL}. Error: {e}")
    st.stop()

st.divider()

# --- Batch scoring uploader ---
st.subheader("Batch scoring")
st.write(
    "Upload a CSV with an `award_id` column (optional `supplier_id`). "
    "We'll call `/v1/score/batch` and display results. "
    "Toggle 'Join graph metrics' to attach supplier graph features if available."
)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    join_graph = st.toggle("Join graph metrics", value=False)
with c2:
    limit_top_factors = st.slider(
        "Top factors per item", min_value=1, max_value=20, value=5
    )
with c3:
    timeout_sec = st.number_input(
        "Request timeout (sec)", min_value=5, max_value=300, value=60, step=5
    )

up = st.file_uploader("CSV file", type=["csv"], accept_multiple_files=False)
if up is not None:
    try:
        df = pd.read_csv(up)
        if "award_id" not in df.columns:
            # try first column as award_id
            first = df.columns[0]
            st.warning(f"`award_id` column not found; using first column `{first}`.")
            df = df.rename(columns={first: "award_id"})

        # Build list payload; include supplier_id if provided
        def _row_to_item(row: pd.Series) -> Dict[str, Any]:
            item = {"award_id": str(row["award_id"])}
            if "supplier_id" in row and pd.notna(row["supplier_id"]):
                item["supplier_id"] = str(row["supplier_id"])
            return item

        payload: List[Dict[str, Any]] = [_row_to_item(r) for _, r in df.iterrows()]

        # Call API; respect shape mirroring (list-in -> list-out)
        url = f"{API_URL}/v1/score/batch"
        params = {
            "join_graph": str(join_graph).lower(),
            "limit_top_factors": int(limit_top_factors),
        }
        resp = requests.post(url, json=payload, params=params, timeout=timeout_sec)
        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
            st.stop()

        data = resp.json()
        # Support both list and envelope shapes
        if isinstance(data, dict) and "items" in data:
            items = data["items"]
            provenance_id = data.get("provenance_id")
            used_model = data.get("used_model")
        else:
            items = data
            provenance_id = None
            used_model = None

        out = pd.DataFrame(items)

        # Feedback / header
        if used_model is None:
            st.success(f"Scored {len(out)} rows")
        else:
            st.success(
                (
                    f"Scored {len(out)} rows"
                    + (f" • provenance_id={provenance_id}" if provenance_id else "")
                    + (f" • used_model={used_model}" if used_model is not None else "")
                )
            )

        # Errors panel (keep 1:1 shape; show any items with 'error')
        if "error" in out.columns and out["error"].notna().any():
            st.warning("Some rows had validation issues:")
            st.dataframe(
                out.loc[out["error"].notna(), ["award_id", "supplier_id", "error"]],
                use_container_width=True,
            )

        # Optional risk band filter
        band_values = ["low", "medium", "high"]
        selected_bands = st.multiselect(
            "Filter by risk_band", options=band_values, default=band_values
        )
        filtered = out.copy()
        if "risk_band" in filtered.columns:
            filtered = filtered[filtered["risk_band"].isin(selected_bands)]

        # Primary scores table
        score_cols = [
            c
            for c in ["award_id", "supplier_id", "risk_band", "risk_score"]
            if c in filtered.columns
        ]
        if score_cols:
            st.dataframe(
                filtered[score_cols].sort_values("risk_score", ascending=False),
                use_container_width=True,
            )

        # Compact top-factor summary
        if "top_factors" in filtered.columns:
            with st.expander("Show top factor details per award (top N)"):
                rows = []
                for _, r in filtered.iterrows():
                    tf = r.get("top_factors") or []
                    if not isinstance(tf, list):
                        tf = []
                    row_d = {
                        "award_id": r.get("award_id"),
                        "supplier_id": r.get("supplier_id"),
                    }
                    # show up to 3 columns for compactness
                    for i in range(min(3, len(tf))):
                        if isinstance(tf[i], dict):
                            name = tf[i].get("name")
                            contrib = tf[i].get("contribution")
                            value = tf[i].get("value")
                            row_d[f"factor{i+1}"] = (
                                f"{name} (contrib={contrib})"
                                if contrib is not None
                                else f"{name} (value={value})"
                            )
                    rows.append(row_d)
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Simple chart of top 20 by risk_score
        try:
            if "risk_score" in filtered.columns and not filtered.empty:
                top = filtered.sort_values("risk_score", ascending=False).head(20)
                st.bar_chart(top.set_index("award_id")["risk_score"])
        except Exception:
            pass

        # Download scored CSV
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download scored results (CSV)",
            data=csv_bytes,
            file_name="batch_scored_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"Failed to score file: {e}")
else:
    st.info("No file uploaded yet.")
