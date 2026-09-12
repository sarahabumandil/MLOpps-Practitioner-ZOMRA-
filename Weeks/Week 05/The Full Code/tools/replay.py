"""Re-score logged serving vectors offline and compare. The incident-02 detector.

Every other detector in this repo watches a DISTRIBUTION: the inputs (Evidently),
the output (PSI), the error (Page-Hinkley). Train/serve skew defeats all three.
The inputs are unchanged, the model file is unchanged, and the output shift is
small enough to look like ordinary drift. What changed is the transform between
them — and no distribution monitor can see a transform.

The only thing that catches it: take the exact vectors the API scored, run them
through the offline pipeline, and compare answers on identical inputs. If the
two disagree, the difference IS the serving path.

    python -m tools.replay --limit 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from services.ride_model import ARTIFACTS, DATA, FEATURES

SERVING_LOG = DATA / "serving_log.jsonl"


def load_log(path: Path, limit: int) -> list[dict]:
    """Read the most recent `limit` rows of the serving log."""
    if not path.exists():
        raise SystemExit(f"no serving log at {path} — run `make api` and `make traffic` first")
    with open(path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return rows[-limit:]


def offline_scores(rows: list[dict]) -> np.ndarray:
    """Score the logged vectors the way the offline pipeline would — current scaler."""
    model = joblib.load(ARTIFACTS / "rf_model.pkl")
    scaler = joblib.load(ARTIFACTS / "scaler.pkl")
    x = np.array(
        [[r["distance_km"], float(r["passengers"]), float(r["hour_of_day"])] for r in rows]
    )
    return model.predict(scaler.transform(x))


def replay(limit: int, tolerance: float) -> int:
    """Compare served vs offline scores. Returns a POSIX exit code."""
    rows = load_log(SERVING_LOG, limit)
    served = np.array([r["served_min"] for r in rows])
    offline = offline_scores(rows)
    diff = served - offline
    mismatched = int((np.abs(diff) > tolerance).sum())
    share = mismatched / len(rows)

    print(f"\n  replayed {len(rows)} logged serving vectors against the offline pipeline")
    print("  features are identical by construction — only the code path differs\n")
    print(f"    mean signed difference : {diff.mean():+7.2f} min")
    print(f"    mean absolute diff     : {np.abs(diff).mean():7.2f} min")
    print(f"    disagreeing (>{tolerance} min) : {mismatched}/{len(rows)} ({share:.1%})")

    if share > 0.05:
        print(
            "\n  VERDICT: the serving path and the offline path do not agree on the same\n"
            "  inputs. This is not drift — drift moves the inputs, and the inputs here\n"
            "  are byte-identical. Check what the serving process loaded at startup.\n"
        )
        return 1
    print("\n  VERDICT: serving and offline agree. Train/serve skew is ruled out.\n")
    return 0


def main() -> None:
    """CLI entry point — see RUNBOOK.md step 2."""
    parser = argparse.ArgumentParser(prog="python -m tools.replay", description=__doc__)
    parser.add_argument("--limit", type=int, default=2000, help="most recent N logged vectors")
    parser.add_argument("--tolerance", type=float, default=0.5, help="minutes of allowed drift")
    args = parser.parse_args()
    assert FEATURES == ["distance_km", "passengers", "hour_of_day"], "feature order is the contract"
    raise SystemExit(replay(args.limit, args.tolerance))


if __name__ == "__main__":
    main()
