# app/ui/pages/1_Datasets.py
import os

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
        c.caption(path)
except Exception as e:
    st.error(f"Failed to query /health from {API_URL}. Error: {e}")
    st.stop()

st.divider()

# --- Batch scoring uploader ---
st.subheader("Batch scoring")
st.write(
    "Upload a CSV with an `award_id` column. We'll call `/v1/score/batch` and display results."
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
        payload = [{"award_id": str(x)} for x in df["award_id"].astype(str).tolist()]
        resp = requests.post(f"{API_URL}/v1/score/batch", json=payload, timeout=60)
        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
        else:
            data = resp.json()
            out = pd.DataFrame(data)
            st.success(f"Scored {len(out)} rows")
            st.dataframe(out, use_container_width=True)
            # Simple chart of top 20 by risk_score
            try:
                top = out.sort_values("risk_score", ascending=False).head(20)
                st.bar_chart(top.set_index("award_id")["risk_score"])
            except Exception:
                pass
    except Exception as e:
        st.error(f"Failed to score file: {e}")
else:
    st.info("No file uploaded yet.")
