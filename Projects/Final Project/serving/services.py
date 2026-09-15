"""BentoML service — Category 2 serving (Module 3), BentoML 1.4+ new-style API.

Wraps the same SentimentClassifier used by the FastAPI baseline, but BentoML manages
worker concurrency and adaptive batching for us via @bentoml.api(batchable=True),
so inference does not compete with request handling for the GIL the way the raw
FastAPI baseline does.

Run locally:  pip install -e ".[dev,serving]" && bentoml serve serving.service:SentimentService --reload
Build/push:   bentoml build && bentoml containerize arasent_sentiment:latest
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bentoml

from arasent.config import LABELS, settings
from arasent.model import SentimentClassifier


@bentoml.service(
    name="arasent_sentiment",
    resources={"cpu": "1"},
    traffic={"timeout": 10},
)
class SentimentService:
    def __init__(self) -> None:
        self.classifier = SentimentClassifier.load(settings.baseline_model_path)

    @bentoml.api(batchable=True, batch_dim=0, max_batch_size=32, max_latency_ms=200)
    def predict(self, texts: list[str] = ["منتج رائع"]) -> list[dict]:
        """Adaptive micro-batching: up to 32 requests are coalesced into one model
        call if they land within 200ms of each other — the throughput/latency dial
        the handbook asks for."""
        return self.classifier.predict_batch(texts)

    @bentoml.api
    def predict_one(self, text: str = "منتج رائع") -> dict:
        return self.classifier.predict_one(text)

    @bentoml.api
    def health(self) -> dict:
        return {"status": "healthy", "labels": LABELS, "model_version": self.classifier.model_version}
