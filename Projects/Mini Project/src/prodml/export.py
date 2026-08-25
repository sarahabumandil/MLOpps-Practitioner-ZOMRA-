"""Export the pickled sklearn Pipeline to ONNX, with a parity check against
the original. See reports/module-1.md for the serialization-format decision
table (JSON / Protobuf / Pickle / ONNX) and the benchmark numbers.

Pickle executes arbitrary code on load — never load a .pkl you did not
produce yourself.
"""

from __future__ import annotations

import pickle

import numpy as np
import onnxruntime as rt
from skl2onnx import convert_sklearn

from prodml.config import settings
from prodml.logging_conf import get_logger

logger = get_logger(__name__)


def export_to_onnx() -> None:
    with open(settings.model_path, "rb") as f:
        pipeline = pickle.load(f)

    # DictVectorizer isn't directly convertible from raw dict input, so we
    # export only the downstream regressor: the API's vectorizer step stays
    # in Python, and the fixed-width vectorized output is what ONNX serves.
    from skl2onnx.common.data_types import FloatTensorType

    sample = [{"PU_DO": "1_2", "trip_distance": 1.0}]
    vec = pipeline.named_steps["vectorizer"].transform(sample)
    n_features = vec.shape[1]
    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(pipeline.named_steps["model"], initial_types=initial_types)

    settings.onnx_model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info("onnx_export_complete", extra={"path": str(settings.onnx_model_path)})


def parity_check(n_samples: int = 500) -> float:
    """Compare pickle vs ONNX predictions on n_samples; returns max abs diff."""
    with open(settings.model_path, "rb") as f:
        pipeline = pickle.load(f)

    rng = np.random.default_rng(settings.random_seed)
    samples = [
        {
            "PU_DO": f"{rng.integers(1, 30)}_{rng.integers(1, 30)}",
            "trip_distance": float(rng.gamma(2, 1.8)),
        }
        for _ in range(n_samples)
    ]
    pkl_preds = pipeline.predict(samples)

    vec = pipeline.named_steps["vectorizer"].transform(samples).astype(np.float32).toarray()
    session = rt.InferenceSession(str(settings.onnx_model_path))
    input_name = session.get_inputs()[0].name
    onnx_preds = session.run(None, {input_name: vec})[0].ravel()

    max_diff = float(np.max(np.abs(pkl_preds - onnx_preds)))
    logger.info("parity_check_complete", extra={"max_diff": max_diff, "n_samples": n_samples})
    return max_diff


if __name__ == "__main__":
    export_to_onnx()
    diff = parity_check()
    print(f"Max abs diff (pickle vs ONNX) over 500 samples: {diff:.6f}")
