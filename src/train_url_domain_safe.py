"""Compare URL classifiers on an unseen *registrable-domain* holdout.

Place inside CyberThreatDetectionStarter/src/, then run:
    python -m pip install tldextract
    python -m src.train_url_domain_safe

This script only reads local CSV data. It never visits URLs and never
replaces your existing app.py or previously saved models.

Why this matters: exact-hostname grouping still lets google.com and
www.google.com fall on opposite sides. Registrable-domain grouping keeps
both in the SAME group.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src.train_url import LABELS, ROOT, read_dataset
from src.url_features import FEATURE_COLUMNS, make_feature_frame


def extract_host(url: str) -> str:
    text = str(url).strip()
    try:
        parts = urlsplit(text if "://" in text else "http://" + text)
        return (parts.hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def registered_domain(url: str, extractor) -> str:
    """Group related hosts under one registrable domain (offline PSL).

    The bundled public-suffix list is not necessarily the latest. Private
    hosting platforms are deliberately grouped at the registrable domain
    rather than individual subdomain level for a conservative holdout.
    """
    host = extract_host(url)
    if not host:
        return ""
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    result = extractor(host)
    if result.domain and result.suffix:
        return f"{result.domain}.{result.suffix}".lower()
    return host


def evaluate(name: str, y_true: pd.Series, predicted, test_hosts: pd.Series,
             test_domains: pd.Series, reports: Path) -> dict:
    y_true = pd.Series(y_true).reset_index(drop=True)
    pred = pd.Series(predicted).reset_index(drop=True)
    print(f"\n=== {name} — unseen registrable domains ===")
    print(classification_report(y_true, pred, labels=LABELS, zero_division=0))
    matrix = confusion_matrix(y_true, pred, labels=LABELS)
    print(pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_string())
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
        reports / f"{name}_confusion.csv"
    )

    benign = y_true.eq("benign")
    false_alerts = benign & pred.ne("benign")
    false_alert_rate = float(false_alerts.sum() / max(1, benign.sum()))
    print(f"Benign false-alert rate: {int(false_alerts.sum())}/{int(benign.sum())} "
          f"({false_alert_rate:.1%})")
    pd.DataFrame({
        "hostname": test_hosts.reset_index(drop=True),
        "registered_domain": test_domains.reset_index(drop=True),
        "actual": y_true,
        "predicted": pred,
    })[false_alerts].to_csv(reports / f"{name}_benign_false_alerts.csv", index=False)

    return {
        "classification_report": classification_report(
            y_true, pred, labels=LABELS, output_dict=True, zero_division=0
        ),
        "benign_false_alert_rate": false_alert_rate,
        "benign_false_alert_count": int(false_alerts.sum()),
        "benign_count": int(benign.sum()),
    }


def main() -> None:
    try:
        import tldextract
    except ImportError as exc:
        raise SystemExit("Missing package 'tldextract'. Run: python -m pip install tldextract") from exc

    parser = argparse.ArgumentParser(description="Unseen-registrable-domain URL classification experiment")
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/malicious_phish.csv")
    parser.add_argument("--max-per-class", type=int, default=6000)
    parser.add_argument("--trees", type=int, default=120)
    args = parser.parse_args()

    # No update request is made for the public suffix list at runtime.
    extractor = tldextract.TLDExtract(suffix_list_urls=())
    frame = read_dataset(args.data, args.max_per_class).copy()
    frame["hostname"] = frame["url"].map(extract_host)
    frame["registered_domain"] = frame["url"].map(lambda s: registered_domain(s, extractor))
    invalid = frame["hostname"].eq("") | frame["registered_domain"].eq("")
    if invalid.any():
        print(f"Skipping {int(invalid.sum())} unparsable URLs")
        frame = frame.loc[~invalid].copy()
    frame = frame.reset_index(drop=True)
    if set(frame["type"]) != set(LABELS):
        raise SystemExit("Invalid sample: at least one class has no usable URLs")

    domain_counts = frame.groupby("type")["registered_domain"].nunique().reindex(LABELS)
    print("Distinct registered domains per class:\n" + domain_counts.to_string())
    if domain_counts.min() < 5:
        raise SystemExit("Need at least five registrable-domain groups in each class")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    chosen = None
    # Choose based on class presence, never on evaluation scores.
    for train_ix, test_ix in splitter.split(frame["url"], frame["type"], frame["registered_domain"]):
        if set(frame.iloc[train_ix]["type"]) == set(LABELS) and set(frame.iloc[test_ix]["type"]) == set(LABELS):
            chosen = train_ix, test_ix
            break
    if chosen is None:
        raise SystemExit("Could not produce a fold containing all classes. Try a larger sample.")
    train_ix, test_ix = chosen
    train = frame.iloc[train_ix].reset_index(drop=True)
    test = frame.iloc[test_ix].reset_index(drop=True)

    overlap = set(train["registered_domain"]) & set(test["registered_domain"])
    if overlap:
        raise AssertionError("Train/test domain leakage detected")

    print(f"\nTraining: {len(train):,} URLs, {train['registered_domain'].nunique():,} domains")
    print(f"Testing:  {len(test):,} URLs, {test['registered_domain'].nunique():,} domains")
    print("Registered-domain overlap: 0")
    print("Training class counts:\n" + train["type"].value_counts().reindex(LABELS, fill_value=0).to_string())
    print("Testing class counts:\n" + test["type"].value_counts().reindex(LABELS, fill_value=0).to_string())
    print("Note: grouping can make the test set and class balance uneven.")

    report_dir, model_dir = ROOT / "reports", ROOT / "models"
    report_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    results = {
        "split": {
            "method": "first valid StratifiedGroupKFold fold; shuffled with random_state=42",
            "group": "registrable domain using bundled Public Suffix List (tldextract)",
            "train_rows": len(train), "test_rows": len(test),
            "train_domains": int(train["registered_domain"].nunique()),
            "test_domains": int(test["registered_domain"].nunique()),
            "domain_overlap": len(overlap),
            "train_class_counts": train["type"].value_counts().to_dict(),
            "test_class_counts": test["type"].value_counts().to_dict(),
        }
    }

    rf = RandomForestClassifier(
        n_estimators=args.trees, max_depth=22, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    print("\nTraining Random Forest on numeric lexical features...")
    rf.fit(make_feature_frame(train["url"].tolist()), train["type"])
    rf_predictions = rf.predict(make_feature_frame(test["url"].tolist()))
    results["random_forest"] = evaluate(
        "url_domain_rf", test["type"], rf_predictions, test["hostname"],
        test["registered_domain"], report_dir
    )
    joblib.dump({
        "model": rf, "feature_columns": FEATURE_COLUMNS,
        "classes": list(rf.classes_),
        "notes": "Trained with registrable-domain holdout; class scores uncalibrated",
    }, model_dir / "url_domain_rf.joblib")

    linear = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                  min_df=2, max_features=40000, lowercase=True)),
        ("classifier", LogisticRegression(max_iter=500, class_weight="balanced",
                                          random_state=42)),
    ])
    print("\nTraining character n-gram Logistic Regression...")
    linear.fit(train["url"].tolist(), train["type"])
    linear_predictions = linear.predict(test["url"].tolist())
    results["char_logistic_regression"] = evaluate(
        "url_domain_char_logreg", test["type"], linear_predictions,
        test["hostname"], test["registered_domain"], report_dir
    )
    joblib.dump(linear, model_dir / "url_domain_char_logreg.joblib")

    destination = report_dir / "url_domain_comparison.json"
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation: {destination}")
    print("The original models and Streamlit apps were NOT modified.")
    print("Do not compare accuracy directly against earlier splits; their test sets differ.")
    print("These models only learn from URL strings; scores are NOT safety guarantees.")


if __name__ == "__main__":
    main()
