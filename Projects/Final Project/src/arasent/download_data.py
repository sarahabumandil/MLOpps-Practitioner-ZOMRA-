from __future__ import annotations
from arasent.data import download_from_kaggle
from arasent.logging_conf import configure_logging

def main() -> None:
    configure_logging()
    path = download_from_kaggle()
    print(f"Saved dataset to {path}")
if __name__ == "__main__":
    main()
