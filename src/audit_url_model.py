"""Audit the existing four-class URL model without retraining or visiting URLs.

Place this file at CyberThreatDetectionStarter/src/audit_url_model.py and run:
    python -m src.audit_url_model

The test split is reconstructed from src/train_url.py defaults.
If you trained with non-default --max-per-class, pass that value here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.predict_url import load_predictor, predict
from src.train_url import LABELS, ROOT, read_dataset
from src.url_features import FEATURE_COLUMNS, make_feature_frame


def hostname(url: str) -> str:
    """Read the host from text only. Never makes HTTP requests."""
    text = str(url).strip()
    parsed = urlsplit(text if "://" in text else "http://" + text)
    return (parsed.hostname or "").lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit URL classifier predictions and split overlap")
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/malicious_phish.csv")
    parser.add_argument("--max-per-class", type=int, default=6000)
    args = parser.parse_args()

    # Uses exactly the same dataset cleaning, random state, and split as train_url.py.
    frame = read_dataset(args.data, args.max_per_class)
    train_urls, test_urls, _, y_test = train_test_split(
        frame["url"].tolist(), frame["type"].tolist(),
        test_size=0.2, random_state=42, stratify=frame["type"].tolist()
    )

    bundle = load_predictor()
    model = bundle["model"]
    x_test = make_feature_frame(test_urls)
    predictions = model.predict(x_test)
    scores = model.predict_proba(x_test)
    class_names = list(model.classes_)
    phishing_index = class_names.index("phishing")

    print("\n=== Classification report: reconstructed held-out test set ===")
    print(classification_report(y_test, predictions, labels=LABELS, zero_division=0))

    matrix = confusion_matrix(y_test, predictions, labels=LABELS)
    print("=== Confusion matrix (rows=actual; columns=predicted) ===")
    print(pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_string())

    actual_benign = pd.Series(y_test).eq("benign")
    false_alarms = actual_benign & pd.Series(predictions).ne("benign")
    print(f"\nBenign URLs incorrectly flagged: {int(false_alarms.sum())}/{int(actual_benign.sum())} "
          f"({false_alarms.mean() / actual_benign.mean():.1%} of actual benign URLs)")

    train_hosts = {hostname(u) for u in train_urls} - {""}
    test_hosts = pd.Series([hostname(u) for u in test_urls])
    repeated = test_hosts.isin(train_hosts)
    print(f"Test URLs with a hostname also seen in training: {int(repeated.sum())}/{len(test_hosts)} "
          f"({repeated.mean():.1%}). This is a warning about related-record overlap, "
          "not proof of incorrect labels.")

    out = pd.DataFrame({
        "actual": y_test,
        "predicted": predictions,
        "hostname": test_hosts,
        "url_length": [len(u) for u in test_urls],
        "phishing_score_uncalibrated": scores[:, phishing_index],
    })
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    out.loc[false_alarms].sort_values("phishing_score_uncalibrated", ascending=False).to_csv(
        report_dir / "url_false_positives.csv", index=False
    )
    print("Saved: reports/url_false_positives.csv (hostnames only; full URLs omitted)")

    print("\n=== Top 12 global feature importances ===")
    print(pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
          .sort_values(ascending=False).head(12).to_string())

    linkedin = "https://www.linkedin.com/feed/foryou/"
    result = predict(linkedin, bundle)
    print("\n=== LinkedIn URL from your screenshot ===")
    print("Predicted:", result["label"])
    print("Uncalibrated scores:", ", ".join(f"{k}={v:.1%}" for k, v in result["scores"].items()))
    print("Lexical input features:")
    print(make_feature_frame([linkedin]).T.to_string(header=False))
    print("\nNote: global feature importance does NOT explain this one prediction. "
          "These results cannot guarantee whether any URL is safe.")


if __name__ == "__main__":
    main()
