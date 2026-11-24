"""Simple web UI for the first URL classification module."""
from pathlib import Path

import streamlit as st
import pandas as pd

from src.predict_url import MODEL_PATH, load_predictor, predict

st.set_page_config(page_title="Cyber Threat Detector", page_icon="🛡️", layout="centered")
st.title("🛡️ Cyber Threat Detector")
st.caption("Phase 1: four-class URL detection (benign, phishing, malware, defacement)")
st.info("Analysis uses the URL text only. The app does not visit, download from, or execute the supplied URL.")

if not Path(MODEL_PATH).exists():
    st.error("No trained model found. First run: python -m src.train_url")
    st.stop()

@st.cache_resource
def get_model():
    return load_predictor()

url = st.text_input("Enter URL", placeholder="https://example.com/login")
if st.button("Analyze URL", type="primary"):
    if not url.strip():
        st.warning("Please enter a URL")
    else:
        try:
            result = predict(url, get_model())
            st.subheader("Predicted URL category")
            st.markdown(f"### {result['label'].title()}")
            st.write("Model class scores (not calibrated probabilities):")
            st.bar_chart(pd.Series(result["scores"], name="Model score"))
            st.warning("This is an experimental URL classifier, not a guarantee that a website is safe or unsafe.")
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Prediction failed. Verify the saved model and input.")

st.divider()
st.caption("Next modules: email phishing, SMS spam, attachment analysis, and unified threat results.")
