"""Train a four-class Random Forest URL classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.url_features import FEATURE_COLUMNS, make_feature_frame

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["benign", "defacement", "malware", "phishing"]


def read_dataset(path: Path, max_per_class: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run: python -m scripts.download_dataset, or place the CSV there manually."
        )
    frame = pd.read_csv(path, usecols=["url", "type"], dtype="string")
    frame = frame.dropna(subset=["url", "type"]).copy()
    frame["url"] = frame["url"].str.strip()
    frame["type"] = frame["type"].str.strip().str.lower()
    frame = frame[(frame["url"].str.len() > 0) & frame["type"].isin(LABELS)].copy()
    if frame.empty:
        raise ValueError("No rows with the expected labels and URL strings were found")
    # Exclude URLs assigned contradictory labels; don't allow such duplicates to leak.
    inconsistent_urls = frame.groupby("url")["type"].nunique()
    inconsistent_urls = inconsistent_urls[inconsistent_urls > 1].index
    if len(inconsistent_urls):
        frame = frame[~frame["url"].isin(inconsistent_urls)]
    frame = frame.drop_duplicates(subset=["url"])
    if max_per_class > 0:
        groups = []
        for label in LABELS:
            subset = frame[frame["type"] == label]
            groups.append(subset.sample(n=min(len(subset), max_per_class), random_state=42))
        frame = pd.concat(groups, ignore_index=True)
    if set(frame["type"].unique()) != set(LABELS):
        raise ValueError("Dataset must contain all four classes: " + ", ".join(LABELS))
    if (frame["type"].value_counts() < 2).any():
        raise ValueError("Need at least two samples per class")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/malicious_phish.csv")
    parser.add_argument("--max-per-class", type=int, default=6000,
                        help="Class-balanced cap for a fast first experiment; 0 uses all rows")
    parser.add_argument("--trees", type=int, default=120)
    args = parser.parse_args()

    frame = read_dataset(args.data, args.max_per_class)
    print("Training rows per class:\n", frame["type"].value_counts().to_string())
    # Split raw URLs first, then compute the same lexical features for both groups.
    train_urls, test_urls, y_train, y_test = train_test_split(
        frame["url"].tolist(), frame["type"].tolist(),
        test_size=0.2, random_state=42, stratify=frame["type"].tolist()
    )
    print("Extracting numeric URL features...")
    X_train = make_feature_frame(train_urls)
    X_test = make_feature_frame(test_urls)

    model = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=22,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    print("Training Random Forest...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, labels=LABELS, output_dict=True, zero_division=0)
    print(classification_report(y_test, predictions, labels=LABELS, zero_division=0))

    model_dir = ROOT / "models"
    report_dir = ROOT / "reports"
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "classes": model.classes_.tolist(),
        "dataset_name": "sid321axn/malicious-urls-dataset",
        "notes": "URL lexical classifier. Scores are uncalibrated; not a safety guarantee.",
    }
    joblib.dump(bundle, model_dir / "url_random_forest.joblib")
    with (report_dir / "url_metrics.json").open("w") as f:
        json.dump({"classification_report": report,
                   "training_rows": len(train_urls), "test_rows": len(test_urls),
                   "dataset": str(args.data)}, f, indent=2)
    pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False).to_csv(
        report_dir / "url_feature_importance.csv", header=["importance"])
    matrix = confusion_matrix(y_test, predictions, labels=LABELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(ax=ax, xticks_rotation=30)
    fig.tight_layout()
    fig.savefig(report_dir / "url_confusion_matrix.png", dpi=160)
    plt.close(fig)
    print("Saved model to models/url_random_forest.joblib")
    print("Saved evaluation files to reports/")


if __name__ == "__main__":
    main()
