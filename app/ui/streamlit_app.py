import os
import requests
import streamlit as st
import pandas as pd

# Use the package path for utils (matches utils.py we created)
from app.ui.utils import (
    results_json_to_dataframe,
    df_to_csv_bytes,
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Procurement Risk Detection AI", layout="centered")
st.title("Procurement Risk Detection – Demo")

st.caption("This is a minimal demo UI. Enter parameters and submit to score risk.")

# -----------------------------
# Single-record demo (/v1/score)
# -----------------------------
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

# ----------------------------------------------
# Batch Scoring UI (/v1/score/batch) with fallbacks
# ----------------------------------------------
st.markdown("---")
st.header("Batch Scoring (CSV → API → Download Results)")

st.caption(
    "Upload a CSV containing your batch (e.g., award_id, supplier_id). "
    "The app will try multiple request formats to match the API schema, "
    "display results with flattened top-factor columns, and let you download the CSV."
)

uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])

with st.expander("CSV tips & minimal requirements"):
    st.markdown(
        "- **Required**: `award_id`\n"
        "- **Recommended**: `supplier_id`\n"
        "- Other columns are optional; backend may ignore or use them.\n"
        "- If you see 422 errors, ensure you have the `award_id` column and try the minimal sample below."
    )


def _post_batch_with_fallbacks(api_url: str, records):
    """
    Try several likely server contracts in order:
      1) plain list
      2) {'items': list}
      3) {'records': list}
    Returns (response, payload_variant) where payload_variant is 'list' | 'items' | 'records'.
    """
    import requests

    variants = [
        ("list", records),
        ("items", {"items": records}),
        ("records", {"records": records}),
    ]

    last_exc = None
    for name, payload in variants:
        try:
            resp = requests.post(f"{api_url}/v1/score/batch", json=payload, timeout=60)
            if resp.status_code < 400:
                return resp, name
            # If 422, try the next variant
            if resp.status_code != 422:
                # Non-schema error; surface it immediately.
                return resp, name
        except Exception as e:
            last_exc = e
    if last_exc:
        raise last_exc
    return None, None


if uploaded_csv is not None:
    try:
        df_in = pd.read_csv(uploaded_csv)
        if df_in.empty:
            st.warning("Uploaded CSV is empty.")
        elif "award_id" not in df_in.columns:
            st.error("CSV must include a column named 'award_id'.")
        else:
            st.write("**Preview (first 10 rows):**")
            st.dataframe(df_in.head(10), use_container_width=True)

            records = df_in.to_dict(orient="records")

            with st.spinner("Scoring batch..."):
                resp, variant = _post_batch_with_fallbacks(API_URL, records)

            if resp is None:
                st.error("Request failed and no response was returned.")
            elif not resp.ok:
                st.error(
                    f"API error ({resp.status_code}) using payload variant '{variant}'."
                )
                st.text(resp.text if resp.text else "<no response body>")
            else:
                st.caption(f"Used payload variant: **{variant}**")
                data = resp.json()
                # Accept either a list response or a dict with 'items'
                if isinstance(data, dict) and "items" in data:
                    results_list = data["items"]
                else:
                    results_list = data

                df_results = results_json_to_dataframe(results_list)
                if df_results.empty:
                    st.warning("No results parsed from API response.")
                else:
                    st.success(f"Received {len(df_results)} scored rows.")
                    st.dataframe(df_results, use_container_width=True)

                    csv_bytes = df_to_csv_bytes(df_results)
                    st.download_button(
                        label="⬇️ Download batch scores as CSV",
                        data=csv_bytes,
                        file_name="batch_scores.csv",
                        mime="text/csv",
                    )

    except Exception as e:
        st.error(f"Failed to process batch CSV: {e}")
        st.exception(e)
