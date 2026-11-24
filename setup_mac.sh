#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
printf '\nSetup complete. Next run:\n  source .venv/bin/activate\n  python -m scripts.download_dataset\n  python -m src.train_url\n  python -m streamlit run app.py\n'
