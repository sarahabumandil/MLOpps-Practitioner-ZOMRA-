"""Backfill ~14 simulated days so nothing in the lab starts empty.

    python -m tools.seed                 # metrics history only (no services needed)
    python -m tools.seed --days 21
    python -m tools.seed --llm           # also drive langfuse_workload.py for traces

This is load-bearing, not decoration. Half of what session 4 teaches is only
visible over a POPULATION and a TIMELINE:

  * trend vs threshold — the seeded distance_km PSI climbs from 0.02 to past
    0.25 over twelve days. A threshold alert fires on day 12; the trend was
    obvious on day 6. That contrast is the lesson, and it needs twelve days of
    history to exist at all.
  * the frozen reference — a drift number is meaningless without the baseline
    it was measured against, so every seeded row carries its reference_id.
  * per-segment — alexandria's rows are seeded separately from cairo's, so the
    "healthy average" lesson has data behind it.
  * the gate's memory — retrain_gate reads drift_history(days=14). With an empty
    table every gate trivially passes and the refusals never happen.

Without this the session degrades into a tool tour of empty dashboards.

The Langfuse half deliberately shells out to langfuse_workload.py rather than
reimplementing it: that file already creates traces, sessions, users, prompt
versions, datasets, experiments and all four score types.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from jobs import metrics_store
from services.ride_model import FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ID = "train-v3-50k"


#: The slow shift. Day 0 is healthy; by the last days distance_km has crossed the
#: 0.25 action band — but it has been climbing the whole time, which is the point.
def _psi_for(day: int, days: int, feature: str, rng: random.Random) -> float:
    """PSI for `feature` on simulated `day`, with a deliberate story per feature."""
    progress = day / max(days - 1, 1)
    if feature == "distance_km":
        return round(0.02 + 0.32 * progress**1.6 + rng.uniform(-0.01, 0.01), 4)
    if feature == "hour_of_day":
        # Weekly seasonality, never near the action band: the control feature
        # that shows students what ordinary movement looks like.
        return round(0.03 + 0.02 * abs((day % 7) - 3.5) / 3.5 + rng.uniform(0, 0.01), 4)
    return round(0.01 + rng.uniform(0, 0.015), 4)


def seed_metrics(days: int, seed: int = 42) -> dict[str, int]:
    """Write drift rows, late labels, deploys and retrain decisions for `days` days."""
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    drift_rows: list[tuple] = []
    label_rows: list[tuple] = []
    counts = {"drift": 0, "labels": 0, "deploys": 0, "retrain": 0}

    for day in range(days):
        start = now - timedelta(days=days - day)
        end = start + timedelta(hours=1)
        ws, we = start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")

        for feature in FEATURES:
            value = _psi_for(day, days, feature, rng)
            drift_rows.append((ws, we, feature, "psi", value, "all", "v3", REFERENCE_ID))
            # A p-value alongside every effect size, so the "significant but tiny"
            # contrast in the README's PSI table has real numbers behind it.
            drift_rows.append(
                (
                    ws,
                    we,
                    feature,
                    "p_value",
                    round(rng.uniform(0.0, 0.04), 4),
                    "all",
                    "v3",
                    REFERENCE_ID,
                )
            )
            for city in ("cairo", "alexandria"):
                jitter = 1.0 if city == "cairo" else 1.4
                drift_rows.append(
                    (ws, we, feature, "psi", round(value * jitter, 4), city, "v3", REFERENCE_ID)
                )
        drift_rows.append(
            (
                ws,
                we,
                "prediction",
                "psi",
                round(0.01 + 0.18 * (day / max(days - 1, 1)) ** 2, 4),
                "all",
                "v3",
                REFERENCE_ID,
            )
        )

        # Late labels: MAE degrades gently as the inputs shift underneath the model.
        mae_today = 2.8 + 2.4 * (day / max(days - 1, 1)) ** 2
        for i in range(120):
            ts = (start + timedelta(minutes=rng.randint(0, 1400))).isoformat(timespec="seconds")
            y_pred = rng.uniform(8.0, 75.0)
            label_rows.append(
                (f"seed-{day:02d}-{i:04d}", ts, "cairo", y_pred + rng.gauss(0, mae_today), y_pred)
            )

    counts["drift"] = metrics_store.record_drift(drift_rows)
    counts["labels"] = metrics_store.record_labels(label_rows)

    # A deploy timeline to correlate against — RUNBOOK step 3. Back-dated: a
    # history where every release happened at once correlates with nothing.
    for day, component, version in ((3, "model-api", "v3"), (9, "upstream-feed", "feed-2026.3")):
        ts = (now - timedelta(days=days - day)).isoformat(timespec="seconds")
        metrics_store.record_deploy(component, version, note="scheduled release", ts=ts)
        counts["deploys"] += 1

    # Retrain decisions, mostly holds. A gate that only logs its yeses is not auditable.
    for days_ago, trigger, decision, reasons, promoted in (
        (
            13,
            "drift:distance_km",
            "hold",
            "drift not sustained for 2 days — one window is noise",
            False,
        ),
        (11, "drift:distance_km", "hold", "only 412 new labels, need 1000", False),
        (
            9,
            "scheduled",
            "retrain",
            "sustained drift | checks passed | 5k labels | outside cooldown",
            True,
        ),
        (8, "drift:distance_km", "hold", "inside the 7-day cooldown", False),
    ):
        ts = (now - timedelta(days=days_ago)).isoformat(timespec="seconds")
        metrics_store.record_retrain_event(
            trigger, decision, reasons, "distance_km", promoted, ts=ts
        )
        counts["retrain"] += 1
    return counts


def seed_langfuse(traces: int = 500) -> bool:
    """Drive langfuse_workload.py for traces, scores, prompts, datasets, experiments.

    Not reimplemented here on purpose — that file already does all of it, and a
    second copy would be a second thing to keep correct. Needs the langfuse
    profile up and Ollama reachable; failure is reported, never fatal.
    """
    script = PROJECT_ROOT / "langfuse_workload.py"
    if not script.exists():
        print("  langfuse_workload.py not found — skipping trace seeding")
        return False
    print(f"  driving langfuse_workload.py for ~{traces} traces (this takes a while)")
    result = subprocess.run(
        [sys.executable, str(script), "--traces", str(traces)], cwd=PROJECT_ROOT
    )
    if result.returncode != 0:
        print(
            "  trace seeding failed. Check that `make up-langfuse` is running and Ollama\n"
            "  is reachable (ollama serve && ollama pull llama3.1:8b). The metrics half\n"
            "  above is unaffected — labs 1 and 2 will work regardless."
        )
        return False
    return True


def main() -> None:
    """CLI entry point — see the module docstring."""
    parser = argparse.ArgumentParser(prog="python -m tools.seed", description=__doc__)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--llm", action="store_true", help="also seed Langfuse traces")
    parser.add_argument("--traces", type=int, default=500)
    args = parser.parse_args()

    counts = seed_metrics(args.days)
    print(f"\n  seeded {args.days} simulated days:")
    print(f"    drift_metrics  {counts['drift']:>5} rows")
    print(f"    labels         {counts['labels']:>5} rows")
    print(f"    deploys        {counts['deploys']:>5} rows")
    print(f"    retrain_events {counts['retrain']:>5} rows")
    if args.llm:
        # Flush first: the subprocess writes straight to the terminal, so
        # anything still sitting in our buffer would print after its output.
        sys.stdout.flush()
        seed_langfuse(args.traces)
    print("\n  check it:  python -m jobs.retrain_gate\n")


if __name__ == "__main__":
    main()
