"""Evaluate the starter URL classifier with an unseen-hostname holdout.

Place in CyberThreatDetectionStarter/src/train_url_grouped.py, then run:
    python -m src.train_url_grouped

Uses the project's existing URL feature extractor and dataset reader.
No websites are visited. Does NOT replace the model used by app.py.
"""
from __future__ import annotations

import argparse
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


def hostname(raw_url: str) -> str:
    """Find hostname from string only; performs no HTTP requests."""
    text = str(raw_url).strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text if "://" in text else "http://" + text)
        return (parts.hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def save_report(name: str, y_true: pd.Series, pred, hosts: pd.Series, out_dir: Path) -> dict:
    report = classification_report(
        y_true, pred, labels=LABELS, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, pred, labels=LABELS)
    actual_benign = (y_true.to_numpy() == "benign")
    pred_array = pd.Series(pred).to_numpy()
    benign_false_alerts = int((actual_benign & (pred_array != "benign")).sum())
    benign_support = int(actual_benign.sum())

    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
        out_dir / f"{name}_confusion.csv"
    )
    pd.DataFrame({
        "hostname": hosts.to_numpy(),
        "actual": y_true.to_numpy(),
        "predicted": pred_array,
    }).loc[lambda df: (df["actual"] == "benign") & (df["predicted"] != "benign")].to_csv(
        out_dir / f"{name}_benign_false_alerts.csv", index=False
    )

    print(f"\n=== {name}: unseen-hostname test ===")
    print(classification_report(y_true, pred, labels=LABELS, zero_division=0))
    print(pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_string())
    print(
        f"Benign false-alert rate: {benign_false_alerts}/{benign_support} "
        f"({benign_false_alerts / benign_support:.1%})"
    )
    return {
        "classification_report": report,
        "benign_false_alerts": benign_false_alerts,
        "benign_test_count": benign_support,
        "benign_false_alert_rate": benign_false_alerts / benign_support,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RF and char n-gram baseline on unseen hostnames")
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/malicious_phish.csv")
    parser.add_argument("--max-per-class", type=int, default=6000)
    parser.add_argument("--trees", type=int, default=120)
    parser.add_argument("--skip-linear", action="store_true", help="Only train Random Forest")
    args = parser.parse_args()

    df = read_dataset(args.data, args.max_per_class).copy()
    df["hostname"] = df["url"].map(hostname)
    invalid_count = int((df["hostname"] == "").sum())
    if invalid_count:
        print(f"Skipping {invalid_count} URLs without a parseable hostname")
        df = df[df["hostname"] != ""].copy()
    df = df.reset_index(drop=True)

    domain_counts = df.groupby("type")["hostname"].nunique()
    print("\nDistinct hostnames per class:")
    print(domain_counts.to_string())
    min_groups = int(domain_counts.min())
    if min_groups < 2:
        raise ValueError("At least 2 distinct hostnames per class are required for group evaluation")
    n_splits = min(5, min_groups)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Pick the first split containing all classes on both sides. This choice
    # is made BEFORE fitting any model and never by inspecting model scores.
    chosen = None
    for train_idx, test_idx in splitter.split(df["url"], df["type"], df["hostname"]):
        if set(df.iloc[train_idx]["type"]) == set(LABELS) and set(df.iloc[test_idx]["type"]) == set(LABELS):
            chosen = (train_idx, test_idx)
            break
    if chosen is None:
        raise ValueError(
            "Cannot construct a split with all four classes on both sides. "
            "Try --max-per-class 0, or inspect which classes are concentrated in a few hosts."
        )

    train_idx, test_idx = chosen
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    host_overlap = set(train["hostname"]) & set(test["hostname"])
    if host_overlap:
        raise AssertionError("Host overlap was detected; evaluation aborted")

    print(f"\nTraining URLs: {len(train):,} | Testing URLs: {len(test):,}")
    print(f"Training hostnames: {train['hostname'].nunique():,} | Testing hostnames: {test['hostname'].nunique():,}")
    print("Hostname overlap between train and test: 0")
    print("\nTraining class counts:\n" + train["type"].value_counts().reindex(LABELS, fill_value=0).to_string())
    print("\nTesting class counts:\n" + test["type"].value_counts().reindex(LABELS, fill_value=0).to_string())
    print("Note: grouped folds may not contain exactly 20% of rows or equal class counts.")

    out_dir = ROOT / "reports"
    model_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    X_train = make_feature_frame(train["url"].tolist())
    X_test = make_feature_frame(test["url"].tolist())
    forest = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=22,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    print("\nTraining grouped Random Forest...")
    forest.fit(X_train, train["type"])
    rf_pred = forest.predict(X_test)
    results = {
        "split": {
            "method": "StratifiedGroupKFold; first fold containing all classes",
            "group": "exact hostname (not registrable domain)",
            "train_rows": len(train),
            "test_rows": len(test),
            "train_hosts": int(train["hostname"].nunique()),
            "test_hosts": int(test["hostname"].nunique()),
            "hostname_overlap": 0,
            "train_label_counts": train["type"].value_counts().to_dict(),
            "test_label_counts": test["type"].value_counts().to_dict(),
        },
        "random_forest": save_report("url_grouped_rf", test["type"], rf_pred, test["hostname"], out_dir),
    }
    pd.Series(forest.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False).to_csv(
        out_dir / "url_grouped_rf_feature_importance.csv", header=["importance"]
    )
    joblib.dump({
        "model": forest,
        "feature_columns": FEATURE_COLUMNS,
        "classes": list(forest.classes_),
        "notes": "RF trained with exact-hostname grouped holdout; scores uncalibrated.",
    }, model_dir / "url_grouped_rf.joblib")

    if not args.skip_linear:
        print("\nTraining character n-gram Logistic Regression baseline...")
        linear = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5), min_df=2,
                max_features=40000, lowercase=True,
            )),
            ("classifier", LogisticRegression(
                max_iter=500, class_weight="balanced", random_state=42,
            )),
        ])
        linear.fit(train["url"].tolist(), train["type"])
        linear_pred = linear.predict(test["url"].tolist())
        results["char_logistic_regression"] = save_report(
            "url_grouped_char_logreg", test["type"], linear_pred, test["hostname"], out_dir
        )
        joblib.dump(linear, model_dir / "url_grouped_char_logreg.joblib")

    report_path = out_dir / "url_grouped_comparison.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved grouped evaluation:", report_path)
    print("Your current models/url_random_forest.joblib and Streamlit app are unchanged.")
    print("Do NOT compare the new grouped scores directly with the old random-split score as a model improvement test.")
    print("Compare the two grouped models on the same grouped test set instead.")


if __name__ == "__main__":
    main()
