"""Use the saved model to classify a URL without visiting it."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.url_features import FEATURE_COLUMNS, make_feature_frame

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/url_random_forest.joblib"


def load_predictor(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError("Model missing. Run: python -m src.train_url")
    # Only load models trained locally or obtained from a trusted source.
    bundle = joblib.load(path)
    if bundle["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Feature schema mismatch: retrain the model with this code")
    return bundle


def predict(url: str, bundle: dict) -> dict:
    if not url.strip() or len(url.strip()) > 2048:
        raise ValueError("Enter a URL between 1 and 2048 characters")
    features = make_feature_frame([url])
    classifier = bundle["model"]
    label = str(classifier.predict(features)[0])
    scores = dict(zip(classifier.classes_, (float(p) for p in classifier.predict_proba(features)[0])))
    return {"label": label, "scores": scores, "note": "Uncalibrated model scores; not a safety verdict."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL to classify as text; it will NOT be opened")
    args = parser.parse_args()
    result = predict(args.url, load_predictor())
    print("Predicted category:", result["label"])
    print("Scores (uncalibrated):")
    for label, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
        print(f"  {label}: {score:.1%}")
    print(result["note"])


if __name__ == "__main__":
    main()
