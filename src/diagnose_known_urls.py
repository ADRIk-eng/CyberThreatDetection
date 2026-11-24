"""Diagnose false alarms on known web URLs without network access.

Place in CyberThreatDetectionStarter/src/diagnose_known_urls.py
Run: python -m src.diagnose_known_urls

Loads only locally trained models (joblib files are executable deserialization
formats; never load an untrusted model file).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd

from src.train_url import ROOT
from src.url_features import make_feature_frame

DATA_PATH = ROOT / "data/raw/malicious_phish.csv"
MODEL_PATHS = {
    "Grouped Random Forest": ROOT / "models/url_grouped_rf.joblib",
    "Character n-gram Logistic Regression": ROOT / "models/url_grouped_char_logreg.joblib",
}
KNOWN_URLS = [
    "https://www.google.com",
    "https://www.google.com/",
    "https://google.com/",
    "https://www.linkedin.com/feed/foryou/",
    "https://www.wikipedia.org/",
    "https://github.com/",
]
KNOWN_HOSTS = {
    "google.com", "www.google.com", "linkedin.com", "www.linkedin.com",
    "wikipedia.org", "www.wikipedia.org", "github.com", "www.github.com",
}


def hostname(value: str) -> str:
    s = str(value).strip()
    if not s:
        return ""
    try:
        parsed = urlsplit(s if "://" in s else "http://" + s)
        return (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def inspect_dataset() -> None:
    print("\n=== DATASET CHECK ===")
    if not DATA_PATH.exists():
        print(f"Dataset missing: {DATA_PATH}")
        return
    df = pd.read_csv(DATA_PATH, usecols=["url", "type"], dtype="string").dropna()
    df["url"] = df["url"].str.strip()
    df["type"] = df["type"].str.strip().str.lower()
    print("Raw labels / records:")
    print(df["type"].value_counts().to_string())
    # Only parse candidate strings, then require an exact hostname match.
    # A malicious URL can contain 'google.com' in its path or query without
    # being hosted by Google. A substring search alone would be misleading.
    candidates = df["url"].str.contains(
        "google.com|linkedin.com|wikipedia.org|github.com",
        regex=True, case=False, na=False,
    )
    subset = df.loc[candidates, ["url", "type"]].copy()
    subset["hostname"] = subset["url"].map(hostname)
    exact = subset.loc[subset["hostname"].isin(KNOWN_HOSTS)].copy()
    print("\nDataset records with EXACT known hostnames:")
    if exact.empty:
        print("None found. This does not prove model error or explain it.")
    else:
        print(pd.crosstab(exact["hostname"], exact["type"]).to_string())
        conflicts = exact.groupby("url")["type"].nunique()
        print(f"Exact known-host URLs carrying contradictory labels: {(conflicts > 1).sum()}")
    print("\nKnown-brand text appearing ANYWHERE in URL (including decoy paths):")
    print(subset["type"].value_counts().to_string())
    print("Only the exact-hostname table above indicates URLs actually hosted at those names.")


def inspect_models() -> None:
    print("\n=== DIRECT MODEL PREDICTIONS (BYPASS STREAMLIT) ===")
    for label, path in MODEL_PATHS.items():
        print(f"\nModel: {label}")
        if not path.is_file():
            print(f"Missing model: {path}")
            continue
        model_object = joblib.load(path)
        if isinstance(model_object, dict):
            model = model_object["model"]
            X = make_feature_frame(KNOWN_URLS)
        else:
            model = model_object
            X = KNOWN_URLS
        predictions = model.predict(X)
        scores = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        for idx, (url, predicted) in enumerate(zip(KNOWN_URLS, predictions)):
            score_text = ""
            if scores is not None:
                by_class = dict(zip(model.classes_, scores[idx]))
                score_text = f"; top score={max(by_class.values()):.1%} (uncalibrated)"
            print(f"  {url} -> {predicted}{score_text}")
    print("\nDo not use these predictions as a website safety assessment.")


def main() -> None:
    inspect_dataset()
    inspect_models()


if __name__ == "__main__":
    main()
