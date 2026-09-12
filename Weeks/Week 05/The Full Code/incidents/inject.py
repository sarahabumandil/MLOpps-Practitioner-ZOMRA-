"""Inject and heal session-4 incidents. One command each, no spoilers.

    python -m incidents.inject --list
    python -m incidents.inject --incident 01
    python -m incidents.inject --heal
    python -m incidents.inject --reveal 01        # instructor: cause + signals

What "restart the affected component" means here. The perturbed services run on
the host (Prometheus scrapes ``host.docker.internal:8001``) and no seventh
container was added, so an incident flips a flag in ``incidents/state.json``
which the API, the traffic generator and the RAG path re-read within a second.
No PID juggling, no restart race, and ``make incident-01`` works regardless of
which terminal started the API.

Every injection also writes a row to ``deploys`` — that is what puts the
annotation on the Grafana panel, and it is deliberately uninformative: "something
changed on model-api at 14:07" is exactly as much as an on-call engineer gets.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from typing import Any

from incidents import catalog, state
from jobs import metrics_store


def _git_sha() -> str:
    """Short HEAD sha, or '' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _merged_flags(active: list[str]) -> dict[str, Any]:
    """Union the flags of every active incident, so two can run at once."""
    flags: dict[str, Any] = {}
    for inc_id in active:
        flags.update(catalog.get(inc_id).flags)
    return flags


def _print_card(inc: catalog.Incident) -> None:
    """Print the symptom card. Never the cause — that is the entire exercise."""
    print()
    print("─" * 72)
    if inc.kind == "measurement":
        print(f"  INCIDENT {inc.id} — this one is a measurement, not a break")
    else:
        print(f"  INCIDENT {inc.id} injected")
    print("─" * 72)
    print(f"  What was reported:  {inc.symptom}")
    if inc.ttd_minutes:
        print(f"  Time-to-detect budget: {inc.ttd_minutes} min")
    print()
    print("  You have: the Grafana dashboard, the drift report, the traces,")
    print("  the metrics store, and RUNBOOK.md. Start at step 1.")
    print("─" * 72)
    print()


def apply(incident_id: str) -> None:
    """Activate one incident: merge its flags, log the deploy, print the card."""
    inc = catalog.get(incident_id)
    if not inc.implemented:
        print(
            f"Incident {inc.id} is not implemented — that is homework #1.\n"
            "Implement the injector AND the alert that detects it, from the deck "
            "appendix description, and open a PR against this repo.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    current = state.read_state().get("active", [])
    active = sorted(set(current) | {inc.id})
    state.write_state(active, _merged_flags(active))

    if inc.kind == "break":
        metrics_store.record_deploy(
            component=inc.component,
            version=f"{inc.component}-{uuid.uuid4().hex[:6]}",
            git_sha=_git_sha(),
            note="config change",  # deliberately uninformative
        )
    _print_card(inc)


def heal() -> None:
    """Clear every active incident and log the rollback."""
    active = state.read_state().get("active", [])
    state.write_state([], {})
    if active:
        metrics_store.record_deploy(
            component="all",
            version=f"healed-{uuid.uuid4().hex[:6]}",
            git_sha=_git_sha(),
            note="rollback",
        )
        print(f"Healed: {', '.join(active)}. Flags cleared; services pick it up within ~1s.")
    else:
        print("Nothing was active. System is clean.")


def show_list() -> None:
    """List every incident id without giving away what any of them does."""
    active = set(state.read_state().get("active", []))
    print("\n  Injectable incidents\n")
    for inc in sorted(catalog.CATALOG.values(), key=lambda i: i.id):
        if not inc.implemented:
            continue
        mark = "ACTIVE" if inc.id in active else ""
        kind = "measurement" if inc.kind == "measurement" else f"{inc.ttd_minutes:>4} min budget"
        print(f"    {inc.id}   {inc.component:<20} {kind:<18} {mark}")
    homework = sorted(i.id for i in catalog.CATALOG.values() if not i.implemented)
    print(f"\n  Homework (implement injector + alert): {', '.join(homework)}")
    for key, why in catalog.NOT_INJECTABLE.items():
        print(f"  Not injectable: {key} — {why}")
    print()


def reveal(incident_id: str) -> None:
    """Instructor view: cause, signals and the lesson. Do not run this in the lab."""
    inc = catalog.get(incident_id)
    print(f"\n  INCIDENT {inc.id} — {inc.component}\n")
    print(f"  Symptom: {inc.symptom}\n")
    print(f"  Cause:   {inc.cause}\n")
    print("  Signals that catch it:")
    for sig in inc.signals:
        print(f"    - {sig}")
    print(f"\n  Teaches: {inc.teaches}\n")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — see the module docstring for the four modes."""
    parser = argparse.ArgumentParser(prog="python -m incidents.inject", description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--incident", metavar="ID", help="inject one incident, e.g. 01")
    group.add_argument("--heal", action="store_true", help="clear every active incident")
    group.add_argument("--list", action="store_true", help="list incident ids")
    group.add_argument("--reveal", metavar="ID", help="instructor: show cause and signals")
    args = parser.parse_args(argv)

    try:
        if args.list:
            show_list()
        elif args.heal:
            heal()
        elif args.reveal:
            reveal(args.reveal)
        else:
            apply(args.incident)
    except KeyError as exc:
        # A traceback is the wrong way to tell someone they typed 03. Print the
        # reason, which for 03 explains why it is a story and not a lab.
        print(f"\n  {exc.args[0]}\n", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
