"""Drift that a distribution comparison cannot see: error drift and query drift.

    python -m jobs.concept_drift              # both detectors over the stored data
    python -m jobs.concept_drift --by-city    # split the error stream per segment

Sections 1 and 2 of the README compare two distributions and ask "have the
inputs moved?". This file asks the two questions that need a *stream*:

**Has the error started drifting?** Concept drift is a change in the
relationship between inputs and target. Input and output distributions can be
perfectly stationary while the model quietly becomes wrong, so the only place it
shows up is the error — which means it needs ground truth and therefore arrives
last. `hinkley_adwin.py` introduces the two detectors; this runs them over the
real late labels in the metrics store.

**Have the questions changed?** `embedding_drift.py` introduces Maximum Mean
Discrepancy for exactly this. Its detectors cannot be imported — it builds a
SentenceTransformer at module scope and calls an undefined `load_queries()` at
the bottom, being a slide snippet rather than a module — so the MMD formula is
restated below, over the retrieval scores this stack already produces.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable, Sequence

from jobs import metrics_store

#: Same parameters as hinkley_adwin.py, so the two files agree.
PH_PARAMS = {"min_instances": 30, "delta": 0.005, "threshold": 50, "alpha": 0.9999}
ADWIN_DELTA = 0.002


def detect_error_drift(errors: Sequence[float]) -> dict[str, list[int]]:
    """Run Page-Hinkley and ADWIN over an error stream; return where each fired.

    Page-Hinkley is a cumulative-sum test: it accumulates how far the stream has
    run above its own running mean and fires when that sum passes a threshold,
    so it catches a *sustained* directional shift and ignores a single bad batch.
    ADWIN keeps an adaptive window and shrinks it when two halves differ
    statistically, so it reacts faster to an *abrupt* change. They disagree on
    purpose — running both is how you tell a step change from a slow slide.
    """
    from river.drift import ADWIN, PageHinkley

    ph, adwin = PageHinkley(**PH_PARAMS), ADWIN(delta=ADWIN_DELTA)
    fired: dict[str, list[int]] = {"page_hinkley": [], "adwin": []}
    for i, error in enumerate(errors):
        ph.update(error)
        adwin.update(error)
        if ph.drift_detected:
            fired["page_hinkley"].append(i)
        if adwin.drift_detected:
            fired["adwin"].append(i)
    return fired


def mmd_rbf(x: Sequence[float], y: Sequence[float], gamma: float = 1.0) -> float:
    """Maximum Mean Discrepancy with an RBF kernel, on 1-D samples.

    Restated from embedding_drift.py, which is not importable. MMD compares two
    samples without binning them, which matters when a distribution is small or
    oddly shaped — the case a PSI histogram handles worst. Near 0 means the two
    samples came from the same distribution.
    """

    def k(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return 0.0
        return sum(math.exp(-gamma * (i - j) ** 2) for i in a for j in b) / (len(a) * len(b))

    return k(x, x) + k(y, y) - 2 * k(x, y)


def error_stream(city: str | None = None, limit: int = 4000) -> list[float]:
    """Absolute error per late label, oldest first — the stream the detectors read."""
    sql = "SELECT ABS(y_true - y_pred) AS err FROM labels"
    params: tuple = ()
    if city:
        sql += " WHERE city = ?"
        params = (city,)
    sql += " ORDER BY ts LIMIT ?"
    with metrics_store.connect() as conn:
        return [row["err"] for row in conn.execute(sql, (*params, limit)).fetchall()]


def cities() -> list[str]:
    """Distinct segments present in the labels table."""
    with metrics_store.connect() as conn:
        return [r["city"] for r in conn.execute("SELECT DISTINCT city FROM labels").fetchall()]


def report(by_city: bool) -> None:
    """Run the detectors and print where the error stream changed."""
    streams: list[tuple[str, list[float]]] = (
        [(c, error_stream(c)) for c in cities()] if by_city else [("all", error_stream())]
    )
    for label, errors in streams:
        if len(errors) < PH_PARAMS["min_instances"]:
            print(
                f"  {label}: only {len(errors)} labels — need "
                f"{PH_PARAMS['min_instances']} before the detectors say anything"
            )
            continue
        fired = detect_error_drift(errors)
        first_half, second_half = errors[: len(errors) // 2], errors[len(errors) // 2 :]
        mae_before = sum(first_half) / len(first_half)
        mae_after = sum(second_half) / len(second_half)
        print(f"\n  {label}: {len(errors)} labels   MAE {mae_before:.2f} -> {mae_after:.2f} min")
        for detector, points in fired.items():
            where = f"at sample {points[0]}" if points else "no drift"
            print(f"    {detector:<14} {len(points):>3} alarm(s)   {where}")
        if not any(fired.values()):
            print(
                "    Neither fired. The error stream is stationary — which is the "
                "result you want, and is only meaningful because labels arrived at all."
            )


def retrieval_drift(scores_before: Iterable[float], scores_after: Iterable[float]) -> float:
    """MMD between two windows of retrieval scores — the incident-09 statistic.

    Cheaper and more honest than eyeballing a mean: it answers "are these two
    samples from the same distribution?" rather than "did the average move?".
    """
    return mmd_rbf(list(scores_before), list(scores_after))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="python -m jobs.concept_drift", description=__doc__)
    parser.add_argument("--by-city", action="store_true", help="split the stream per segment")
    args = parser.parse_args()
    report(args.by_city)
    print()


if __name__ == "__main__":
    main()
