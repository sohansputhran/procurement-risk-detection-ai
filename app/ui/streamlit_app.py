import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Procurement Risk Detection AI", layout="centered")
st.title("Procurement Risk Detection – Demo")

st.caption("This is a minimal demo UI. Enter parameters and submit to score risk.")

with st.form("risk_form"):
    amount = st.number_input(
        "Contract amount (USD)",
        min_value=0.0,
        step=1000.0,
        value=100000.0,
        format="%.2f",
    )
    past_awards = st.number_input(
        "Past awards to the same supplier", min_value=0, step=1, value=3
    )
    is_sanctioned = st.checkbox("Supplier is sanctioned", value=False)
    adverse_media = st.number_input("Adverse media hits", min_value=0, step=1, value=1)
    submitted = st.form_submit_button("Score Risk")

if submitted:
    try:
        payload = {
            "amount": amount,
            "past_awards_count": int(past_awards),
            "is_sanctioned": bool(is_sanctioned),
            "adverse_media_count": int(adverse_media),
        }
        r = requests.post(f"{API_URL}/v1/score", json=payload, timeout=10)
        if r.ok:
            data = r.json()
            st.success(f"Risk score: {data.get('risk_score')}")
            st.caption(data.get("notes", ""))
        else:
            st.error(f"API error: {r.status_code} {r.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
