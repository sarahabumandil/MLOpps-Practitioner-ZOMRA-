"""SentimentClassifier: one class, one clear interface — the same seam pattern used in
Mini Project 1's DurationPredictor. This is the baseline that runs anywhere, right now,
with no GPU and no internet. `model_transformer.py` implements the exact same interface
around a fine-tuned AraBERT checkpoint, so swapping it into the API/BentoML/ONNX path
requires no code changes elsewhere — only a config flag.
"""
from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from arasent.config import LABELS
from arasent.logging_conf import get_logger

logger = get_logger(__name__)


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("timed_call", extra={"function": func.__name__, "elapsed_ms": round(elapsed_ms, 3)})
        return result

    return wrapper


class SentimentClassifier:
    """Baseline: TF-IDF (word + char n-grams, Arabic-friendly) + multinomial Logistic
    Regression. Interface-compatible with the AraBERT model in model_transformer.py.
    """

    framework = "scikit-learn (TF-IDF + LogisticRegression)"

    def __init__(self, vectorizer: TfidfVectorizer | None = None, model: Any = None):
        self.vectorizer = vectorizer
        self.model = model
        self.model_version = "0.1.0-baseline"

    def fit(self, texts: list[str], labels, C: float = 1.0, max_features: int = 20_000, seed: int = 42):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",  # char n-grams are far more robust to Arabic morphology
            ngram_range=(2, 4),
            max_features=max_features,
        )
        X = self.vectorizer.fit_transform(texts)
        # Modern scikit-learn picks multinomial loss automatically for a >2-class
        # problem with the default 'lbfgs' solver — no multi_class= kwarg needed
        # (and newer sklearn versions removed it).
        self.model = LogisticRegression(C=C, max_iter=1000, random_state=seed)
        self.model.fit(X, labels)
        return self

    @timed
    def predict_one(self, text: str) -> dict[str, Any]:
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        idx = int(proba.argmax())
        return {"label": LABELS[idx], "confidence": float(proba[idx])}

    @timed
    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        X = self.vectorizer.transform(texts)
        probas = self.model.predict_proba(X)
        results = []
        for proba in probas:
            idx = int(proba.argmax())
            results.append({"label": LABELS[idx], "confidence": float(proba[idx])})
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vectorizer": self.vectorizer, "model": self.model, "version": self.model_version}, path
        )
        logger.info("model_saved", extra={"path": str(path)})

    @classmethod
    def load(cls, path: Path) -> "SentimentClassifier":
        payload = joblib.load(path)
        instance = cls(vectorizer=payload["vectorizer"], model=payload["model"])
        instance.model_version = payload.get("version", "0.1.0-baseline")
        logger.info("model_loaded", extra={"path": str(path)})
        return instance
