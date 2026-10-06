"""Downloads the Kaggle Arabic reviews dataset into data/raw/arabic_reviews.csv.

Run with: python -m arasent.download_data
Needs: pip install -e ".[data]"  and Kaggle access (works anonymously for public datasets
in most setups; otherwise set KAGGLE_USERNAME / KAGGLE_KEY or run `kagglehub.login()`).
"""
from __future__ import annotations

from arasent.data import download_from_kaggle
from arasent.logging_conf import configure_logging


def main() -> None:
    configure_logging()
    path = download_from_kaggle()
    print(f"Saved dataset to {path}")


if __name__ == "__main__":
    main()
