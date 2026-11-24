"""Streamlit frontend for the latest registered-domain-separated URL models.

Save next to app.py in CyberThreatDetectionStarter and launch with:
    python -m streamlit run app_domain_compare.py --server.port 8503

Offline lexical URL analysis only. Do not treat outputs as safety verdicts.
"""
from pathlib import Path
from urllib.parse import urlsplit
import json

import joblib
import pandas as pd
import streamlit as st

from src.url_features import FEATURE_COLUMNS, make_feature_frame

ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "Character n-gram Logistic Regression": ROOT / "models/url_domain_char_logreg.joblib",
    "Random Forest": ROOT / "models/url_domain_rf.joblib",
}
REPORT_PATH = ROOT / "reports/url_domain_comparison.json"
CLASSES = ["benign", "defacement", "malware", "phishing"]
SAMPLES = {
    "Google homepage": "https://www.google.com/",
    "LinkedIn homepage": "https://www.linkedin.com/",
    "LinkedIn feed": "https://www.linkedin.com/feed/foryou/",
    "GitHub homepage": "https://github.com/",
    "Wikipedia homepage": "https://www.wikipedia.org/",
}


def normalize_url(raw: str) -> str:
    url = raw.strip()
    if not url or len(url) > 2048 or any(ch.isspace() for ch in url):
        raise ValueError("Enter a valid URL (max 2,048 characters, no whitespace).")
    if "://" not in url:
        url = "https://" + url
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid URL hostname or port.") from exc
    if parsed.scheme not in ("https", "http") or not host or "." not in host:
        raise ValueError("Enter an http:// or https:// URL with a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URLs containing embedded credentials are not supported by this demo.")
    return url


@st.cache_resource(show_spinner=False)
def get_models():
    # Joblib is pickle-based: only load model files trained locally or from a trusted source.
    result = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
    rf = result["Random Forest"]
    if not isinstance(rf, dict) or rf.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Random Forest feature schema mismatch. Rerun training.")
    return result


def predict(model_or_bundle, url: str):
    if isinstance(model_or_bundle, dict):
        model = model_or_bundle["model"]
        X = make_feature_frame([url])
    else:
        model = model_or_bundle
        X = [url]
    label = str(model.predict(X)[0])
    scores = dict(zip([str(c) for c in model.classes_],
                      [float(p) for p in model.predict_proba(X)[0]]))
    return label, scores


st.set_page_config(page_title="Cyber Threat Detector - Domain Model Check", page_icon="🛡️", layout="centered")
st.title("🛡️ URL Threat Detection — Latest Model Check")
st.caption("Registered-domain-separated Random Forest vs Character n-gram Logistic Regression")
st.info("**Offline URL text analysis only.** No site is visited. Model outputs are not proof that a website is safe or malicious.")

missing = [p.name for p in MODEL_PATHS.values() if not p.is_file()]
if missing:
    st.error("Missing trained model(s): " + ", ".join(missing))
    st.code("python -m src.train_url_domain_safe", language="bash")
    st.stop()

try:
    models = get_models()
except Exception as exc:
    st.error(f"Failed to load the model files: {exc}")
    st.stop()

with st.expander("Current registered-domain test results", expanded=False):
    if REPORT_PATH.exists():
        try:
            report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            st.caption("Results below are from the previous held-out evaluation, not the URL entered below.")
            for name, key in [("Logistic Regression", "char_logistic_regression"), ("Random Forest", "random_forest")]:
                info = report.get(key, {})
                stats = info.get("classification_report", {})
                if stats:
                    accuracy = stats.get("accuracy", 0)
                    fa = info.get("benign_false_alert_rate", 0)
                    st.write(f"**{name}** — Accuracy: {accuracy:.1%}; benign false-alert rate: {fa:.1%}")
        except (OSError, ValueError, TypeError):
            st.caption("Could not read the evaluation JSON.")
    else:
        st.caption("No saved evaluation report found.")

st.subheader("Test a URL")
choice = st.selectbox("Optional sample (well-known homepage URLs)", ["Custom URL"] + list(SAMPLES))
default = SAMPLES.get(choice, "https://www.google.com/")
url_raw = st.text_input("Enter URL", value=default, key=f"url_for_{choice}")

if st.button("Analyze URL", type="primary", use_container_width=True):
    try:
        url = normalize_url(url_raw)
        outcomes = {name: predict(model, url) for name, model in models.items()}
    except ValueError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        st.caption("URL analyzed: " + url)
        cols = st.columns(2)
        for col, (name, (label, scores)) in zip(cols, outcomes.items()):
            with col:
                st.caption(name)
                st.metric("Predicted class", label.title())
                st.caption(f"Highest uncalibrated model score: {max(scores.values()):.1%}")
        frame = pd.DataFrame({name: {cls: score.get(cls, 0.0) for cls in CLASSES}
                              for name, (_, score) in outcomes.items()}).reindex(CLASSES)
        st.subheader("Class scores")
        st.bar_chart(frame)
        if len({result[0] for result in outcomes.values()}) > 1:
            st.warning("The models disagree. Treat the result as uncertain.")
        if url in SAMPLES.values() and any(label != "benign" for label, _ in outcomes.values()):
            st.warning("A familiar genuine homepage URL was flagged as a threat: this is a likely false positive and needs investigation.")
        st.warning("Scores are **not calibrated probabilities**. URL-only models cannot verify a site's ownership, contents, or safety. Malware and defacement are **URL dataset categories**, not proof of an infection or website compromise.")

st.divider()
st.caption("These domain-separated models are experimental. Do not use the app to decide whether an unknown link is safe to click.")
