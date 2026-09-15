"""DVC 'evaluate' stage. Loads the model artifact `train` produced and the val set
`prepare` produced, and writes a metrics JSON — the CI quality gate reads this file.
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from arasent.config import LABELS, settings
from arasent.logging_conf import configure_logging, get_logger
from arasent.model import SentimentClassifier

logger = get_logger(__name__)

PROCESSED_DIR = settings.project_root / "data" / "processed"
REPORTS_DIR = settings.project_root / "reports"


def main() -> None:
    configure_logging()
    val_df = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    clf = SentimentClassifier.load(settings.baseline_model_path)

    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    preds_raw = clf.predict_batch(val_df["text"].tolist())
    preds = [label_to_idx[p["label"]] for p in preds_raw]

    metrics = {
        "accuracy": accuracy_score(val_df["label"], preds),
        "f1_macro": f1_score(val_df["label"], preds, average="macro"),
        "n_eval": len(val_df),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("evaluate_complete", extra=metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
