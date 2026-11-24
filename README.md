# Cyber Threat Detection

This starter implements **four-class URL classification** using numeric lexical URL features and a Random Forest. It accepts `url` and `type` from Kaggle's `sid321axn/malicious-urls-dataset`, where `type` is one of `benign`, `phishing`, `malware`, or `defacement`.

**IMPORTANT:** These categories describe the associated *URL*. A malware URL is not the same as scanning an executable; a defacement URL is not proof of a current compromise. Scores are **uncalibrated**; this is an academic prototype, not a real-time security guarantee.

## 1. macOS setup

Open Terminal (in VS Code or macOS), navigate to this project folder:

```bash
cd /path/to/CyberThreatDetectionStarter
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your `python3` is older than 3.10, install a supported Python version, then recreate `.venv`. Python 3.11 or 3.12 is a reasonable choice.

## 2. Get dataset

```bash
python -m scripts.download_dataset
```

If the download requires authentication or fails, manually download the CSV from:
https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset

Extract it and place the CSV here: `data/raw/malicious_phish.csv`.
Required columns: `url`, `type`. Do not open dataset URLs in a browser.

## 3. Unit tests

```bash
python -m unittest discover -s tests -v
```

## 4. Train URL classifier

```bash
python -m src.train_url
```

By default, this takes at most 6,000 labelled URLs per class as a starter experiment. This is intentionally smaller than the full dataset to make a first local run manageable. To use all rows later:

```bash
python -m src.train_url --max-per-class 0
```

Outputs:
- `models/url_random_forest.joblib`
- `reports/url_metrics.json`
- `reports/url_confusion_matrix.png`
- `reports/url_feature_importance.csv`

## 5. Predict in terminal

```bash
python -m src.predict_url 'https://example.com'
```

The prediction reads the URL *as a string*, it never visits it.

## 6. Run Streamlit UI

```bash
python -m streamlit run app.py
```

Usually opens at http://localhost:8501

## 7. Optional FastAPI

In another Terminal tab with virtual environment activated:

```bash
python -m uvicorn api:app --reload
```

Open http://127.0.0.1:8000/docs for testable API documentation.

## Notes on evaluation

- The provided random holdout after duplicate removal is a **baseline only**. It may overestimate real-world performance because related domains/campaigns can occur in both train and test.
- A later milestone is grouping by domain, time-based or external-dataset evaluation, probability calibration, and class-specific error analysis.
- The `suspicious` label is not trained; a future system may report "Needs Review" based on validated uncertainty thresholds.
- Do not share training data or model files in Git without reviewing licensing and size.

## Future work

1. A domain-disjoint evaluation split and character n-gram baseline.
2. URL model comparisons: Random Forest, SVM/linear baseline, XGBoost.
3. Email phishing and SMS spam text pipelines.
4. Static, sandboxed file analysis and web-content features.
5. A unified input router and model-specific predictions in FastAPI.
6. Explainability and deployment.
