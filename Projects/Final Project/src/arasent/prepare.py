"""DVC 'prepare' stage. Writes the train/val split to data/processed/ so it is a
versioned, cacheable DVC output that `train` and `evaluate` depend on — instead of every
stage silently re-deriving its own split from the raw source.
"""
from __future__ import annotations

from pathlib import Path

from arasent.config import settings
from arasent.data import load_train_val_split
from arasent.logging_conf import configure_logging, get_logger

logger = get_logger(__name__)

PROCESSED_DIR = settings.project_root / "data" / "processed"


def main() -> None:
    configure_logging()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df = load_train_val_split()
    train_df.to_parquet(PROCESSED_DIR / "train.parquet")
    val_df.to_parquet(PROCESSED_DIR / "val.parquet")
    logger.info(
        "prepare_complete",
        extra={"n_train": len(train_df), "n_val": len(val_df), "out_dir": str(PROCESSED_DIR)},
    )


if __name__ == "__main__":
    main()
