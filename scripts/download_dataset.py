"""Download public Kaggle malicious URLs dataset into data/raw/."""
from pathlib import Path
import shutil

import kagglehub

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data/raw/malicious_phish.csv"


def main() -> None:
    print("Retrieving Kaggle dataset (may require Kaggle login)...")
    dataset_dir = Path(kagglehub.dataset_download("sid321axn/malicious-urls-dataset"))
    files = list(dataset_dir.rglob("*.csv"))
    # Official collection has a CSV with the columns 'url' and 'type'.
    if not files:
        raise FileNotFoundError("No CSV found in downloaded dataset directory")
    preferred = next((p for p in files if p.name == "malicious_phish.csv"), files[0])
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(preferred, DESTINATION)
    print("Dataset copied to:", DESTINATION)
    print("Source file:", preferred)


if __name__ == "__main__":
    main()
