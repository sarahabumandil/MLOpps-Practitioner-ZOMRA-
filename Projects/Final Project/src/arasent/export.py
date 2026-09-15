"""Optimization (Module 4) for the baseline model — runs today, no GPU needed.
The AraBERT equivalent (distillation + ONNX INT8) is in
notebooks/finetune_arabert_colab.ipynb, since it needs a GPU runtime.
"""
from __future__ import annotations

import time

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from arasent.config import settings
from arasent.logging_conf import get_logger
from arasent.model import SentimentClassifier

logger = get_logger(__name__)


def export_to_onnx(clf: SentimentClassifier) -> None:
    n_features = len(clf.vectorizer.get_feature_names_out())
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    # zipmap=False: without it, sklearn-onnx wraps class probabilities in a list of
    # dicts (one per row) instead of a plain float array, which is what every runtime
    # consumer here (parity check, BentoML, benchmark harness) expects.
    onnx_model = convert_sklearn(
        clf.model, initial_types=initial_type, options={id(clf.model): {"zipmap": False}}
    )
    settings.onnx_model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info("onnx_export_complete", extra={"path": str(settings.onnx_model_path)})


def _time_calls(fn, n_calls: int = 200) -> dict[str, float]:
    for _ in range(20):  # warmup, discarded
        fn()
    times = []
    for _ in range(n_calls):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return {"mean_ms": float(np.mean(times)), "p95_ms": float(times[int(0.95 * len(times)) - 1])}


def parity_and_latency_check(clf: SentimentClassifier, texts: list[str]) -> dict:
    import onnxruntime as rt

    X = clf.vectorizer.transform(texts).toarray().astype(np.float32)
    sklearn_preds = clf.model.predict_proba(X)

    sess = rt.InferenceSession(str(settings.onnx_model_path))
    input_name = sess.get_inputs()[0].name
    onnx_preds = sess.run(None, {input_name: X})[1]  # [0]=predicted label, [1]=probabilities

    is_close = np.allclose(sklearn_preds, onnx_preds, atol=1e-3)

    sklearn_latency = _time_calls(lambda: clf.model.predict_proba(X[:1]))
    onnx_latency = _time_calls(lambda: sess.run(None, {input_name: X[:1]})[1])

    result = {
        "parity_ok": bool(is_close),
        "max_abs_diff": float(np.max(np.abs(sklearn_preds - onnx_preds))),
        "sklearn_latency_ms": sklearn_latency,
        "onnx_latency_ms": onnx_latency,
    }
    logger.info("parity_and_latency_check", extra=result)
    return result
