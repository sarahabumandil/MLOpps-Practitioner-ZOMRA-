"""AraBERT-backed SentimentClassifier — the real Track-1 model.

Implements the exact same interface as arasent.model.SentimentClassifier
(predict_one / predict_batch / save / load / model_version / framework), so it is a
drop-in replacement everywhere else in this repo (API, BentoML runner, ONNX export,
benchmark harness). Nothing outside this file needs to change to switch models — only
`ARASENT_MODEL_BACKEND=transformer` (see api/main.py).

Requires the `transformer` extra: `pip install -e ".[transformer]"`. Fine-tune it first
with notebooks/finetune_arabert_colab.ipynb on a GPU runtime (Colab/Kaggle) — this
environment has no GPU and no HuggingFace Hub access, so the checkpoint must be produced
elsewhere and copied into models/arabert-sentiment/.
"""
from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Any

from arasent.config import LABELS, settings
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


class ArabertSentimentClassifier:
    framework = "PyTorch (AraBERT / bert-base-arabertv02)"

    def __init__(self, tokenizer=None, model=None):
        self.tokenizer = tokenizer
        self.model = model
        self.model_version = "0.1.0-arabert"

    @classmethod
    def load(cls, path: Path | str = settings.transformer_output_dir) -> "ArabertSentimentClassifier":
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The transformer backend needs the 'transformer' extra: "
                'pip install -e ".[transformer]" — and a fine-tuned checkpoint at '
                f"{path} (see notebooks/finetune_arabert_colab.ipynb)."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
        model.eval()
        instance = cls(tokenizer=tokenizer, model=model)
        logger.info("arabert_model_loaded", extra={"path": str(path)})
        return instance

    def _predict_logits(self, texts: list[str]):
        import torch

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.max_seq_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits
        return torch.softmax(logits, dim=-1).numpy()

    @timed
    def predict_one(self, text: str) -> dict[str, Any]:
        proba = self._predict_logits([text])[0]
        idx = int(proba.argmax())
        return {"label": LABELS[idx], "confidence": float(proba[idx])}

    @timed
    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        probas = self._predict_logits(texts)
        return [
            {"label": LABELS[int(p.argmax())], "confidence": float(p.max())} for p in probas
        ]

    def save(self, path: Path | str = settings.transformer_output_dir) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("arabert_model_saved", extra={"path": str(path)})
