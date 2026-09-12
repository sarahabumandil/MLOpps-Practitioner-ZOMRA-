"""CI guard: no metric may carry an unbounded label. Incident 04, as a test.

A Prometheus metric costs one time series per LABEL VALUE combination. A ride id
is unbounded, so `ride_id` as a label is not a debugging convenience — it is an
outage with a 40-minute fuse. Incident 04 injects exactly that; this test is the
thing that should have stopped it reaching production.

The guard reads the collectors' DECLARED label names, not just the labels on
emitted samples: a metric that declares `ride_id` but has not been observed yet
emits no samples at all, and that is precisely the state a code review sees.
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

import services.model_api as api

#: Labels whose value space is per-request, per-user or per-file — i.e. unbounded.
DENYLIST = {"ride_id", "request_id", "user_id", "session_id", "filename", "email"}


def declared_labels() -> dict[str, set[str]]:
    """Map every registered metric name to the label names it declares."""
    found: dict[str, set[str]] = {}
    for name, collector in list(REGISTRY._names_to_collectors.items()):
        labels = set(getattr(collector, "_labelnames", ()) or ())
        for metric in getattr(collector, "collect", lambda: [])():
            for sample in metric.samples:
                labels |= set(sample.labels)
        if labels:
            found.setdefault(name, set()).update(labels)
    return found


def offenders() -> dict[str, set[str]]:
    """Registered metrics carrying a denylisted label, with the offending labels."""
    return {name: bad for name, labels in declared_labels().items() if (bad := labels & DENYLIST)}


def test_no_unbounded_labels_on_a_healthy_system() -> None:
    """With no incident active, nothing in the registry is unbounded."""
    assert offenders() == {}, (
        f"unbounded Prometheus labels declared: {offenders()}. "
        "Put high-cardinality identifiers in traces or logs, never in a metric label."
    )


def test_the_guard_actually_catches_incident_04() -> None:
    """A guard that cannot fail is not a guard — prove it fires on the real bug."""
    api._record_ride_id("ride-abc-123")  # what ENABLE_RIDE_ID_LABEL does per request
    try:
        assert "model_predictions_by_ride_total" in offenders()
        assert offenders()["model_predictions_by_ride_total"] == {"ride_id"}
    finally:
        REGISTRY.unregister(api._ride_id_counter)
        api._ride_id_counter = None
    assert offenders() == {}, "registry must be clean again for the other tests"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
