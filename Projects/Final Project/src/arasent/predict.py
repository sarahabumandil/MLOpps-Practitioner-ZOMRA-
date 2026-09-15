"""Loads the serving model once per process. The backend is a config flag, not a code
change — this is the whole point of the interface shared between model.py and
model_transformer.py.
"""
from __future__ import annotations

import os
from functools import lru_cache

from arasent.config import settings
from arasent.logging_conf import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_predictor():
    backend = os.environ.get("ARASENT_MODEL_BACKEND", "baseline")
    if backend == "transformer":
        from arasent.model_transformer import ArabertSentimentClassifier

        logger.info("loading_transformer_backend")
        return ArabertSentimentClassifier.load(settings.transformer_output_dir)

    from arasent.model import SentimentClassifier

    logger.info("loading_baseline_backend")
    return SentimentClassifier.load(settings.baseline_model_path)
