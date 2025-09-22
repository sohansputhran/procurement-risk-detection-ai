import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model info", layout="wide")
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.title("Model info")

try:
    r = requests.get(f"{API_URL}/v1/model/info", timeout=10)
    r.raise_for_status()
    info = r.json()
except Exception as e:
    st.error(f"Failed to query /v1/model/info from {API_URL}. Error: {e}")
    st.stop()

cols = st.columns(4)
cols[0].metric("Available", "✅" if info.get("available") else "❌")
cols[1].metric("Trained at (UTC)", info.get("trained_at") or "—")
cols[2].metric("Model path", info.get("model_path") or "—")
cols[3].metric("Meta path", info.get("meta_path") or "—")

st.subheader("Features")
feat = info.get("feature_cols") or []
st.write(", ".join(feat) if feat else "—")

st.subheader("Top weights (|coef|)")
weights = info.get("weights", {})
tw = weights.get("top_weights") or []
if tw:
    st.dataframe(pd.DataFrame(tw), use_container_width=True)
else:
    st.info("No weights available.")

st.subheader("Latest evaluation metrics")
ev = info.get("evaluation") or {}
metrics = ev.get("metrics") or {}
if metrics:
    st.json(metrics)
else:
    st.info("No evaluation metrics found. Run the evaluation CLI to generate one.")
