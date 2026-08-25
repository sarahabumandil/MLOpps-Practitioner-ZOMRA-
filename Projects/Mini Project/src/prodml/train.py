"""Train the baseline ride-duration model and persist it to models/model.pkl.

Entry point: ``prodml-train`` (see pyproject.toml [project.scripts]) or
``python -m prodml.train``.
"""

from __future__ import annotations

import pickle
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from prodml.config import settings
from prodml.data import compute_duration_minutes, load_raw, train_val_split
from prodml.features import dataframe_to_features
from prodml.logging_conf import configure_logging, get_logger

logger = get_logger(__name__)


def timed(func: Callable) -> Callable:
    """Decorator logging the wrapped function's execution time in milliseconds.

    Applied to predict_one in Module 1 as required by the handbook; used
    here on train() too, since "how long does training take" is the first
    number every later optimization step (Module 4) is measured against.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"{func.__name__} finished", extra={"elapsed_ms": round(elapsed_ms, 2)})
        return result

    return wrapper


@timed
def train() -> dict[str, float]:
    df = load_raw()
    df["duration"] = compute_duration_minutes(df)
    df = df[(df["duration"] > 0) & (df["duration"] < 180)].reset_index(drop=True)

    train_df, val_df = train_val_split(df)

    X_train = dataframe_to_features(train_df)
    X_val = dataframe_to_features(val_df)
    y_train = train_df["duration"].to_numpy()
    y_val = val_df["duration"].to_numpy()

    pipeline = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=settings.n_estimators,
                    max_depth=settings.max_depth,
                    random_state=settings.random_seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.model_path, "wb") as f:
        pickle.dump(pipeline, f)

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "n_train": len(train_df),
        "n_val": len(val_df),
    }
    logger.info("training_complete", extra=metrics)
    return metrics


def main() -> None:
    configure_logging()
    metrics = train()
    print(f"Validation MAE: {metrics['mae']:.4f} min  |  RMSE: {metrics['rmse']:.4f} min")
    print(f"Model saved to: {settings.model_path}")


if __name__ == "__main__":
    main()
