"""DurationPredictor — the single seam behind every serving layer.

.load() / .predict_one() / .predict_batch() is the interface the FastAPI
route, the Module 3 BentoML Runner, and the Module 4 optimized/ONNX model
will all sit behind, without callers changing.
"""

from __future__ import annotations

import pickle
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

from prodml.config import settings
from prodml.features import row_to_features
from prodml.logging_conf import get_logger

logger = get_logger(__name__)


def timed(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"{func.__name__} served", extra={"latency_ms": round(elapsed_ms, 3)})
        return result

    return wrapper


class DurationPredictor:
    """Loads a fitted sklearn Pipeline and serves single/batch predictions."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or settings.model_path
        self.pipeline = None
        self.version = settings.model_version

    def load(self) -> DurationPredictor:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No model at {self.model_path}. Run `prodml-train` "
                "(or `python -m prodml.train`) first."
            )
        with open(self.model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        logger.info("model_loaded", extra={"path": str(self.model_path)})
        return self

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    @timed
    def predict_one(self, pu: int | str, do: int | str, trip_distance: float) -> float:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded — call .load() first.")
        if trip_distance > 100:
            logger.warning("input_outside_training_range", extra={"trip_distance": trip_distance})
        features = row_to_features(pu, do, trip_distance)
        pred = self.pipeline.predict([features])[0]
        return float(max(pred, 0.0))

    @timed
    def predict_batch(self, rows: list[dict[str, Any]]) -> list[float]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded — call .load() first.")
        features = [
            row_to_features(r["pu_location_id"], r["do_location_id"], r["trip_distance"])
            for r in rows
        ]
        preds = self.pipeline.predict(features)
        return [float(max(p, 0.0)) for p in preds]
