"""What did this cost, and what would it cost on a frontier model? Incident 08.

    python -m tools.cost_report --weeks 4
    python -m tools.cost_report --weeks 4 --frontier 60      # 60x the local rate
    python -m tools.cost_report --csv cost.csv

Not a break — a measurement, and the one finance actually asks for.

Two decisions this file makes on purpose:

**It reports the MIN and MAX week, never the average.** A budget is not set by
a typical week; it is set by the week that would have overrun it. An average
hides exactly the number you needed, and it hides it more effectively the
spikier your traffic is.

**It prices a self-hosted model with an amortised GPU rate**, imported from
langfuse_workload.py rather than re-derived here. Langfuse infers cost by
matching a generation's model name against a price list, and it has no entry for
`llama3.1:8b` — so a cost dashboard over a local model reads $0.00 and everyone
concludes inference is free. It is not free; the honest unit is GPU-hours times
instance price divided by tokens produced.

The `--frontier` multiplier is the actual Incident 08 question: "we self-host,
we measured real token volume, what would migrating cost?" Keep the multiplier
visible and arguable rather than buried.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Imported, not reimplemented: langfuse_workload.py derives these from a real
# deployment shape (an A10G at ~$0.75/hour emitting ~40 output tok/s) and
# explains why input is priced ~10x below output.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from langfuse_workload import USD_PER_1K_INPUT, USD_PER_1K_OUTPUT
except ImportError:  # workload not importable (no ollama installed) — same numbers
    USD_PER_1K_INPUT, USD_PER_1K_OUTPUT = 0.0005, 0.0052


def _api(path: str, params: dict[str, Any]) -> dict[str, Any]:
    """Call the Langfuse v1 public API directly, with basic auth.

    Not `client.api.observations.get_many()`. That is the SDK's v2 API and it
    answers 404 with "only available in a Langfuse v4 write mode" — this stack
    pins the langfuse/langfuse:3.x server, and the SDK major and the SERVER
    major move independently (see the note in pyproject.toml). The v1 endpoint
    below is what a v3 server actually serves, and it carries everything the
    report needs.
    """
    import httpx

    base = os.getenv("LANGFUSE_BASE_URL", os.getenv("LANGFUSE_HOST", "http://localhost:3001"))
    auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
    resp = httpx.get(f"{base}{path}", params=params, auth=auth, timeout=60.0)
    resp.raise_for_status()
    return resp.json()


def _trace_languages(since: str, max_pages: int = 10) -> dict[str, str]:
    """Map trace id -> language, read from the `lang:` tag on the trace.

    Language lives on the TRACE (rag_pipeline tags it via propagate_attributes),
    while tokens live on the OBSERVATION. Cost per language therefore needs a
    join, and this is the cheap direction to do it in: one paged trace listing
    rather than one lookup per generation.
    """
    langs: dict[str, str] = {}
    for page in range(1, max_pages + 1):
        payload = _api("/api/public/traces", {"fromTimestamp": since, "limit": 100, "page": page})
        for trace in payload.get("data", []):
            tag = next((t for t in (trace.get("tags") or []) if t.startswith("lang:")), None)
            langs[trace["id"]] = tag.split(":", 1)[1] if tag else "-"
        if page >= payload.get("meta", {}).get("totalPages", 1):
            break
    return langs


def fetch_generations(hours: int, max_pages: int = 20) -> list[dict[str, Any]]:
    """Pull generation observations with their token usage, joined to trace language."""
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    langs = _trace_languages(since)
    rows: list[dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        payload = _api(
            "/api/public/observations",
            {"type": "GENERATION", "fromStartTime": since, "limit": 100, "page": page},
        )
        for obs in payload.get("data", []):
            usage = obs.get("usageDetails") or obs.get("usage") or {}
            meta = obs.get("metadata") or {}
            started = obs.get("startTime")
            rows.append(
                {
                    "ts": datetime.fromisoformat(started.replace("Z", "+00:00"))
                    if started
                    else None,
                    "model": obs.get("model") or "unknown",
                    "input": int(usage.get("input") or 0),
                    "output": int(usage.get("output") or 0),
                    "prompt_version": str(
                        (obs.get("promptVersion") or meta.get("prompt_version") or "-")
                    ),
                    "lang": langs.get(obs.get("traceId") or "", "-"),
                    "environment": obs.get("environment") or "-",
                }
            )
        if page >= payload.get("meta", {}).get("totalPages", 1):
            break
    return rows


def cost_of(row: dict[str, Any], multiplier: float = 1.0) -> float:
    """Amortised cost of one generation, optionally at a frontier-model rate."""
    return multiplier * (
        row["input"] / 1000 * USD_PER_1K_INPUT + row["output"] / 1000 * USD_PER_1K_OUTPUT
    )


def group_by(rows: list[dict[str, Any]], field: str, multiplier: float) -> list[tuple]:
    """Aggregate tokens and cost by one dimension, most expensive first."""
    agg: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "in": 0, "out": 0, "usd": 0.0})
    for row in rows:
        bucket = agg[str(row.get(field) or "-")]
        bucket["n"] += 1
        bucket["in"] += row["input"]
        bucket["out"] += row["output"]
        bucket["usd"] += cost_of(row, multiplier)
    return sorted(agg.items(), key=lambda kv: -kv[1]["usd"])


def weekly(rows: list[dict[str, Any]], multiplier: float) -> dict[str, float]:
    """Cost per ISO week — the series the min/max are taken from."""
    weeks: dict[str, float] = defaultdict(float)
    for row in rows:
        ts = row["ts"]
        if not ts:
            continue
        key = f"{ts.isocalendar().year}-W{ts.isocalendar().week:02d}"
        weeks[key] += cost_of(row, multiplier)
    return dict(weeks)


def report(rows: list[dict[str, Any]], multiplier: float, csv_path: str | None) -> None:
    """Print the Incident 08 table and optionally write it as CSV."""
    total_in = sum(r["input"] for r in rows)
    total_out = sum(r["output"] for r in rows)
    total = sum(cost_of(r, multiplier) for r in rows)

    print(f"\n  {len(rows)} generations   {total_in:,} input tokens   {total_out:,} output tokens")
    print(
        f"  rate: ${USD_PER_1K_INPUT}/1k in, ${USD_PER_1K_OUTPUT}/1k out"
        f"{f' x{multiplier:g} (frontier)' if multiplier != 1 else ' (amortised GPU, self-hosted)'}"
    )
    print(f"  total: ${total:,.4f}\n")

    if total == 0:
        print(
            "  WARNING: total cost is zero. Either no token usage reached Langfuse, or the\n"
            "  price constants are zero. Tokens arrive automatically from the instrumented\n"
            "  chat() call in ollama_langfuse_rag.py; the PRICE has to be supplied, because\n"
            "  Langfuse has no price list entry for a local model tag. Run\n"
            "  `python langfuse_workload.py` (which calls apply_pricing) or set\n"
            "  OLLAMA_USD_PER_1K_OUTPUT in .env.\n"
        )

    for field in ("environment", "prompt_version", "lang", "model"):
        print(f"  by {field}:")
        print(f"    {'key':<22} {'n':>6} {'in':>10} {'out':>10} {'usd':>10}")
        for key, agg in group_by(rows, field, multiplier):
            print(
                f"    {key:<22} {agg['n']:>6} {agg['in']:>10,.0f} "
                f"{agg['out']:>10,.0f} {agg['usd']:>10.4f}"
            )
        print()

    weeks = weekly(rows, multiplier)
    if weeks:
        cheapest = min(weeks.items(), key=lambda kv: kv[1])
        dearest = max(weeks.items(), key=lambda kv: kv[1])
        print("  by week — the only two numbers a budget needs:")
        for label, (week, usd) in (("MIN", cheapest), ("MAX", dearest)):
            print(f"    {label} week {week}   ${usd:,.4f}")
        spread = dearest[1] / cheapest[1] if cheapest[1] else float("inf")
        shortfall = dearest[1] - total / max(len(weeks), 1)
        print(
            f"\n    peak is {spread:.1f}x the quietest week. Budget on the peak: an average\n"
            f"    week would have under-provisioned this by ${shortfall:,.4f}.\n"
        )

    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["dimension", "key", "generations", "input_tokens", "output_tokens", "usd"]
            )
            for field in ("environment", "prompt_version", "lang", "model"):
                for key, agg in group_by(rows, field, multiplier):
                    writer.writerow(
                        [
                            field,
                            key,
                            agg["n"],
                            int(agg["in"]),
                            int(agg["out"]),
                            round(agg["usd"], 6),
                        ]
                    )
        print(f"  wrote {csv_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="python -m tools.cost_report", description=__doc__)
    parser.add_argument("--weeks", type=int, default=4, help="lookback window, in weeks")
    parser.add_argument(
        "--frontier",
        type=float,
        default=1.0,
        help="multiplier for 'what would this cost on a hosted model'",
    )
    parser.add_argument("--csv", help="also write the table to this path")
    args = parser.parse_args()

    rows = fetch_generations(args.weeks * 7 * 24)
    if not rows:
        raise SystemExit(
            "no generations found. Is the langfuse profile up, and has "
            "`python langfuse_workload.py` or `make seed --llm` run?"
        )
    report(rows, args.frontier, args.csv)


if __name__ == "__main__":
    main()
