"""The drift job: compare today's serving window to a FROZEN reference, and STORE it.

    python -m jobs.daily_drift              # one window, now
    python -m jobs.daily_drift --loop 60    # one simulated "day" per minute

This is data_drift_evidently.py made runnable and given a memory. Two things it
adds beyond that snippet, and both are the lesson:

1. The reference is FROZEN (data/reference.parquet, written once by
   `make bootstrap`). Compare today against a rolling recent window instead and
   a slow shift becomes the new normal — the baseline chases the drift and the
   number stays reassuringly near zero the whole way down.

2. The result is WRITTEN DOWN, one row per window/feature/metric. Prometheus is
   a 15-day ring buffer of time series; "which features drifted against which
   reference, in the window ending Tuesday" is a fact with provenance, and the
   retrain gate has to be able to ask about last month.

Evidently is pinned <0.7 (see pyproject) so this uses the same legacy
`evidently.report.Report` API as the three existing scripts. 0.7 moved
everything to `evidently.Report` / `evidently.presets`; migrating is a separate
piece of work, not a side effect of this one.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from jobs import metrics_store
from services.ride_model import DATA, FEATURES, TARGET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
SERVING_LOG = DATA / "serving_log.jsonl"
REFERENCE = DATA / "reference.parquet"
#: Names the frozen baseline in every stored row. Bump it when you re-freeze,
#: never silently — a drift number compared against a different reference is a
#: different measurement wearing the same name.
REFERENCE_ID = "train-v3-50k"

MODEL_API = "http://localhost:8001"

#: PSI bands, as in the README's table: <0.1 stable, 0.1-0.25 watch, >0.25 act.
PSI_ACT = 0.25


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index — the EFFECT SIZE, not a p-value.

    Same formula as label_predicion_drift.py, repeated here rather than imported
    because that file is a teaching snippet with undefined names at module
    scope; importing it would fail.

    Why the alert threshold hangs off this and not off the KS p-value: a p-value
    answers "could this difference be chance?", and with 20 rps the answer is
    always no. Feed a million rows to KS and a shift too small to matter becomes
    "significant" — the alert fires forever and everyone mutes it. PSI answers
    "how big is the difference?", which does not inflate with sample size, which
    is why it is the credit-risk industry's threshold metric.
    """
    edges = np.histogram(expected, bins=n_bins)[1]
    exp_share = np.histogram(expected, bins=edges)[0] / max(len(expected), 1)
    act_share = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)
    exp_share = np.where(exp_share == 0, 1e-4, exp_share)
    act_share = np.where(act_share == 0, 1e-4, act_share)
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))


def load_current(minutes: int = 60, limit: int = 20_000) -> pd.DataFrame:
    """Read the recent serving window from the log the API writes."""
    if not SERVING_LOG.exists():
        raise SystemExit(f"no serving log at {SERVING_LOG} — run `make api` and `make traffic`")
    with open(SERVING_LOG) as fh:
        rows = [json.loads(line) for line in fh if line.strip()][-limit:]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("serving log is empty — is `make traffic` running?")
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    recent = df[df["ts"] >= cutoff]
    return recent if len(recent) >= 100 else df.tail(2_000)


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, out: Path) -> dict:
    """Run the legacy Evidently report and save the HTML a human opens."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import ColumnDriftMetric
    from evidently.report import Report

    report = Report(
        metrics=[
            DataDriftPreset(),
            # Same per-column test choices as data_drift_evidently.py: KS for a
            # continuous feature, chi-square for a low-cardinality categorical,
            # PSI where a magnitude is wanted rather than a p-value.
            ColumnDriftMetric(column_name="distance_km", stattest="ks"),
            ColumnDriftMetric(column_name="passengers", stattest="chisquare"),
            ColumnDriftMetric(column_name="hour_of_day", stattest="psi"),
        ]
    )
    report.run(reference_data=reference[FEATURES], current_data=current[FEATURES])
    out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out))
    return report.as_dict()


def compute_rows(reference: pd.DataFrame, current: pd.DataFrame, result: dict) -> list[tuple]:
    """Build the drift_metrics rows: PSI and p-value, globally and per city."""
    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    end = now.isoformat(timespec="seconds")
    rows: list[tuple] = []

    for metric in result.get("metrics", []):
        if metric.get("metric") != "ColumnDriftMetric":
            continue
        r = metric["result"]
        rows.append(
            (
                start,
                end,
                r["column_name"],
                "p_value",
                float(r.get("drift_score") or 0.0),
                "all",
                "v3",
                REFERENCE_ID,
            )
        )

    for feature in FEATURES:
        rows.append(
            (
                start,
                end,
                feature,
                "psi",
                psi(reference[feature].to_numpy(), current[feature].to_numpy()),
                "all",
                "v3",
                REFERENCE_ID,
            )
        )
        # Per-city PSI. The global number above is a volume-weighted average and
        # a new region at 8% of traffic cannot move it. This is incident 05.
        for city, chunk in current.groupby("city"):
            if len(chunk) < 50:
                continue
            rows.append(
                (
                    start,
                    end,
                    feature,
                    "psi",
                    psi(reference[feature].to_numpy(), chunk[feature].to_numpy()),
                    str(city),
                    "v3",
                    REFERENCE_ID,
                )
            )

    # Prediction drift: model output now vs model output on the frozen reference.
    baseline = reference["pred"] if "pred" in reference else reference[TARGET]
    rows.append(
        (
            start,
            end,
            "prediction",
            "psi",
            psi(baseline.to_numpy(), current["served_min"].to_numpy()),
            "all",
            "v3",
            REFERENCE_ID,
        )
    )
    return rows


def publish(rows: list[tuple]) -> None:
    """Push the numbers to the one process Prometheus scrapes (see model_api)."""
    feature_psi = {r[2]: r[4] for r in rows if r[1] and r[3] == "psi" and r[5] == "all"}
    segment_psi = [
        {"feature": r[2], "city": r[5], "value": r[4]}
        for r in rows
        if r[3] == "psi" and r[5] != "all"
    ]
    mae = [{"city": row["city"], "value": row["mae"]} for row in metrics_store.mae_by_city(24)]
    try:
        httpx.post(
            f"{MODEL_API}/internal/gauges",
            json={"feature_psi": feature_psi, "segment_psi": segment_psi, "mae": mae},
            timeout=5.0,
        )
    except httpx.HTTPError:
        print("  (model API unreachable — gauges not published, rows still stored)")


def run_once(window_minutes: int = 60) -> list[tuple]:
    """One drift window: report, store, publish, print."""
    reference = pd.read_parquet(REFERENCE)
    current = load_current(window_minutes)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result = evidently_report(reference, current, REPORTS / f"drift_{stamp}.html")
    rows = compute_rows(reference, current, result)
    metrics_store.record_drift(rows)
    publish(rows)

    print(f"\n  drift window: {len(current)} serving rows vs frozen {REFERENCE_ID}")
    for feature, metric, value, segment in [(r[2], r[3], r[4], r[5]) for r in rows]:
        if metric != "psi" or segment != "all":
            continue
        band = "ACT " if value > PSI_ACT else "watch" if value > 0.10 else "ok   "
        print(f"    {feature:<12} PSI={value:6.3f}  {band}")
    print(f"  report: reports/drift_{stamp}.html   rows stored: {len(rows)}")
    return rows


def main() -> None:
    """CLI entry point. --loop compresses a day into a minute so nothing waits."""
    parser = argparse.ArgumentParser(prog="python -m jobs.daily_drift", description=__doc__)
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        metavar="SECONDS",
        help="repeat every N seconds; one simulated day per iteration",
    )
    parser.add_argument("--window", type=int, default=60, help="serving window, minutes")
    args = parser.parse_args()
    while True:
        run_once(args.window)
        if not args.loop:
            return
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
