"""Incident state shared by the injector and the services it perturbs.

Session 4's incidents are injected into a service that runs on the **host**, not
in a container: ``monitoring/prometheus/prometheus.yml`` scrapes
``host.docker.internal:8001``, and ``docker-compose.yaml`` deliberately adds no
seventh container. So "flip a flag and restart the container" becomes "flip a
flag in a state file the service re-reads within a second".

That is also the more robust choice for a live session: ``make incident-01``
works whether or not the API was started from this terminal, and healing is
instant. The deploy row written alongside it (``jobs/metrics_store.py``) is what
puts the annotation on the Grafana panel.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Repo-relative so the API, the injector and the traffic generator agree
#: regardless of which directory they were started from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = Path(os.getenv("INCIDENT_STATE_PATH", PROJECT_ROOT / "incidents" / "state.json"))

#: Shape of a healed system. `active` is a list so overlapping incidents are
#: possible — students diagnosing two at once is a legitimate hard mode.
EMPTY_STATE: dict[str, Any] = {"active": [], "flags": {}, "updated": None}

_cache: tuple[float, dict[str, Any]] | None = None


def read_state() -> dict[str, Any]:
    """Return the current incident state, or :data:`EMPTY_STATE` if none is set."""
    try:
        with open(STATE_PATH) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(EMPTY_STATE)


def write_state(active: list[str], flags: dict[str, Any]) -> dict[str, Any]:
    """Persist `active` incidents and their merged `flags`, and return the state."""
    state = {
        "active": sorted(set(active)),
        "flags": flags,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(STATE_PATH)  # atomic — the API may be mid-read
    return state


def flags() -> dict[str, Any]:
    """Return the active flag map, re-reading the file only when it changes.

    Called on every prediction, so it is cached on the state file's mtime rather
    than parsing JSON per request.
    """
    global _cache
    try:
        mtime = STATE_PATH.stat().st_mtime
    except FileNotFoundError:
        _cache = None
        return {}
    if _cache is None or _cache[0] != mtime:
        _cache = (mtime, read_state().get("flags", {}))
    return _cache[1]


def flag(name: str, default: Any = False) -> Any:
    """Return one flag's value, falling back to the process environment.

    The env var is the escape hatch: it lets an instructor start the API already
    broken (``USE_STALE_SCALER=1 make api``) without touching the state file.
    """
    value = flags().get(name)
    if value is not None:
        return value
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw.lower() in {"1", "true", "yes", "on"}:
        return True
    if raw.lower() in {"0", "false", "no", "off"}:
        return False
    return raw
