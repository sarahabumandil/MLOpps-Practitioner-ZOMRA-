from __future__ import annotations

import numpy as np
import pytest

from prodml.export import export_to_onnx, parity_check


@pytest.mark.slow
def test_pickle_onnx_parity(trained_model):
    """Exports the fixture-trained model to ONNX and asserts predictions
    match the pickle model within a tight tolerance — the Step 4 acceptance
    check from the handbook, run at test-suite scale (50 samples here
    instead of 500, to keep the default `pytest` run fast).
    """
    export_to_onnx()
    max_diff = parity_check(n_samples=50)
    assert max_diff < 1e-3, f"pickle vs ONNX predictions diverge by {max_diff}"
    assert np.isfinite(max_diff)
