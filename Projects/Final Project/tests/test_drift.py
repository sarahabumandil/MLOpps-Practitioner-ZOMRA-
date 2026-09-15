import numpy as np

from arasent.monitoring.drift import compute_psi, drift_report


def test_psi_near_zero_for_identical_distributions():
    ref = np.random.default_rng(0).normal(50, 10, 1000).tolist()
    cur = ref.copy()
    assert compute_psi(ref, cur) < 0.01


def test_psi_high_for_clearly_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(50, 10, 1000).tolist()
    cur = rng.normal(90, 10, 1000).tolist()  # large mean shift
    assert compute_psi(ref, cur) > 0.25


def test_drift_report_flags_drifted_when_psi_exceeds_threshold():
    rng = np.random.default_rng(0)
    ref_len = rng.normal(50, 10, 500).tolist()
    cur_len = rng.normal(120, 10, 500).tolist()
    ref_conf = rng.uniform(0.6, 0.9, 500).tolist()
    cur_conf = rng.uniform(0.6, 0.9, 500).tolist()

    report = drift_report(ref_len, cur_len, ref_conf, cur_conf)
    assert report["drifted"] is True
    assert report["text_length_psi"] > report["confidence_score_psi"]
