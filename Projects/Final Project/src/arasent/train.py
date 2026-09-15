"""Trains the baseline SentimentClassifier across a small hyperparameter sweep, logging
every run to MLflow (params, metrics, artifacts, tags), then registers and promotes the
best run to Production in the MLflow Model Registry.

Run with: python -m arasent.train
MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import json
import subprocess

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score

from arasent.config import settings
from arasent.data import load_train_val_split
from arasent.logging_conf import configure_logging, get_logger
from arasent.model import SentimentClassifier

logger = get_logger(__name__)

# The sweep: (C, max_features) pairs — five runs, one model family (matches the
# "≥5 runs" requirement; the AraBERT notebook sweeps lr/batch_size the same way).
SWEEP = [
    {"C": 0.5, "max_features": 5_000},
    {"C": 1.0, "max_features": 5_000},
    {"C": 1.0, "max_features": 20_000},
    {"C": 2.0, "max_features": 20_000},
    {"C": 5.0, "max_features": 40_000},
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def train_and_evaluate() -> dict:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    train_df, val_df = load_train_val_split()
    git_commit = _git_commit()

    best = {"run_id": None, "f1_macro": -1.0, "metrics": None}

    for i, params in enumerate(SWEEP):
        with mlflow.start_run(run_name=f"baseline-sweep-{i}") as run:
            clf = SentimentClassifier().fit(
                train_df["text"].tolist(),
                train_df["label"].tolist(),
                C=params["C"],
                max_features=params["max_features"],
                seed=settings.random_seed,
            )
            val_preds_raw = clf.predict_batch(val_df["text"].tolist())
            label_to_idx = {"negative": 0, "neutral": 1, "positive": 2}
            preds = [label_to_idx[p["label"]] for p in val_preds_raw]

            accuracy = accuracy_score(val_df["label"], preds)
            f1_macro = f1_score(val_df["label"], preds, average="macro")

            mlflow.log_params(
                {**params, "seed": settings.random_seed, "data_version": "synthetic-or-csv-v1"}
            )
            mlflow.log_metrics({"accuracy": accuracy, "f1_macro": f1_macro, "n_train": len(train_df)})
            mlflow.set_tags({"git_commit": git_commit, "author": "dentiligence", "framework": "sklearn"})
            mlflow.sklearn.log_model(clf.model, "model")

            logger.info(
                "sweep_run_complete",
                extra={"run_id": run.info.run_id, "accuracy": accuracy, "f1_macro": f1_macro, **params},
            )

            if f1_macro > best["f1_macro"]:
                best = {
                    "run_id": run.info.run_id,
                    "f1_macro": f1_macro,
                    "metrics": {"accuracy": accuracy, "f1_macro": f1_macro},
                    "classifier": clf,
                }

    # Persist the best model to disk for the API/BentoML/ONNX path.
    best["classifier"].save(settings.baseline_model_path)

    # Register the best run's model and promote it to Production.
    model_uri = f"runs:/{best['run_id']}/model"
    registered = mlflow.register_model(model_uri, settings.mlflow_registry_model_name)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=settings.mlflow_registry_model_name,
        version=registered.version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info(
        "model_registered_and_promoted",
        extra={"registry_name": settings.mlflow_registry_model_name, "version": registered.version},
    )

    return {"best_run_id": best["run_id"], **best["metrics"], "registered_version": registered.version}


def main() -> None:
    configure_logging()
    result = train_and_evaluate()
    reports_dir = settings.project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "train_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
