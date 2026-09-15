"""Drift monitoring — Population Stability Index (PSI) on the two label-free signals the
rubric asks for: text_length and confidence_score. Vocabulary drift (e.g. Ramadan-specific
words, dialect shifts) shows up first as a shift in these two cheap signals, long before
accuracy visibly degrades (ground truth on reviews lags by days).
"""
from __future__ import annotations

import numpy as np

from arasent.logging_conf import get_logger

logger = get_logger(__name__)


def _psi_for_bins(reference: np.ndarray, current: np.ndarray, bins: np.ndarray) -> float:
    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_psi(reference: list[float], current: list[float], n_bins: int = 10) -> float:
    """PSI < 0.1: no significant shift. 0.1-0.25: moderate shift. > 0.25: significant shift
    (this is the alert threshold used in monitoring/alerts and Grafana)."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    bins = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    bins[0], bins[-1] = -np.inf, np.inf  # catch values outside the reference range
    bins = np.unique(bins)
    if len(bins) < 3:
        return 0.0  # reference has no spread — PSI is undefined/meaningless
    return _psi_for_bins(reference, current, bins)


def drift_report(
    reference_lengths: list[float],
    current_lengths: list[float],
    reference_confidences: list[float],
    current_confidences: list[float],
) -> dict:
    text_length_psi = compute_psi(reference_lengths, current_lengths)
    confidence_psi = compute_psi(reference_confidences, current_confidences)
    report = {
        "text_length_psi": text_length_psi,
        "confidence_score_psi": confidence_psi,
        "drifted": max(text_length_psi, confidence_psi) > 0.25,
    }
    logger.info("drift_report", extra=report)
    return report
