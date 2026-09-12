"""Unit tests for the retrain gate — one per gate, plus the incident-06 scenario.

Each gate is tested alone because each one is individually necessary: remove any
of them and there is a plausible production story where the pipeline retrains on
something it should not have. The last test is that story.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from jobs.retrain_gate import (
    cooldown_ok,
    decide,
    enough_new_labels,
    is_sustained,
    pipeline_is_healthy,
    seasonal_block,
)

NOW = datetime(2026, 6, 15, 9, 0)  # a plain Monday, no seasonal window


def drift_rows(days: int, psi: float = 0.41) -> list[dict]:
    """`days` consecutive days of drifted windows, newest last."""
    return [
        {
            "window_end": (NOW - timedelta(days=d)).isoformat(),
            "value": psi,
            "feature": "distance_km",
        }
        for d in range(days, 0, -1)
    ]


# ── is_sustained ──────────────────────────────────────────────────────
def test_one_drifted_window_is_not_sustained() -> None:
    """A single bad batch must not trigger a retrain."""
    assert is_sustained(drift_rows(1)) is False


def test_two_distinct_days_is_sustained() -> None:
    assert is_sustained(drift_rows(2)) is True


def test_many_windows_on_one_day_is_still_one_day() -> None:
    """24 hourly windows on the same day are one day of evidence, not 24."""
    same_day = [{"window_end": f"2026-06-14T{h:02d}:00:00", "value": 0.6} for h in range(24)]
    assert is_sustained(same_day) is False


def test_psi_below_threshold_does_not_count_as_drift() -> None:
    assert is_sustained(drift_rows(5, psi=0.08)) is False


# ── pipeline_is_healthy ───────────────────────────────────────────────
def test_pipeline_healthy_only_when_every_check_passes() -> None:
    assert pipeline_is_healthy({"schema": True, "nulls": True}) is True
    assert pipeline_is_healthy({"schema": True, "nulls": False}) is False


def test_no_checks_reported_is_not_healthy() -> None:
    """Absence of evidence is not evidence of health."""
    assert pipeline_is_healthy({}) is False


# ── enough_new_labels ─────────────────────────────────────────────────
def test_enough_new_labels_boundary() -> None:
    assert enough_new_labels(999, 1000) is False
    assert enough_new_labels(1000, 1000) is True


# ── cooldown_ok ───────────────────────────────────────────────────────
def test_cooldown_blocks_a_recent_retrain() -> None:
    assert cooldown_ok(NOW - timedelta(days=3), NOW, days=7) is False
    assert cooldown_ok(NOW - timedelta(days=8), NOW, days=7) is True


def test_never_retrained_passes_cooldown() -> None:
    assert cooldown_ok(None, NOW) is True


# ── seasonal_block ────────────────────────────────────────────────────
def test_seasonal_block_names_the_window() -> None:
    assert seasonal_block(datetime(2026, 3, 1)) == "ramadan-2026"
    assert seasonal_block(NOW) is None


def test_seasonal_window_is_half_open() -> None:
    """The end date is exclusive, so adjacent windows cannot both match."""
    cal = {"w": (date(2026, 6, 1), date(2026, 6, 10))}
    assert seasonal_block(datetime(2026, 6, 9), cal) == "w"
    assert seasonal_block(datetime(2026, 6, 10), cal) is None


# ── decide ────────────────────────────────────────────────────────────
def healthy_kwargs(**overrides):
    """Every gate passing — the one case that should actually retrain."""
    base = dict(
        drift_history=drift_rows(3),
        checks={"schema": True, "freshness": True, "nulls": True},
        new_labels=5_000,
        last_retrain=NOW - timedelta(days=30),
        now=NOW,
    )
    base.update(overrides)
    return base


def test_decide_retrains_when_every_gate_passes() -> None:
    decision = decide(**healthy_kwargs())
    assert decision.should_retrain is True
    assert decision.blocked_by == []


def test_decide_reports_every_failing_gate_not_just_the_first() -> None:
    """Short-circuiting throws away the audit trail. Evaluate them all."""
    decision = decide(**healthy_kwargs(new_labels=10, checks={"nulls": False}))
    assert decision.should_retrain is False
    assert len(decision.blocked_by) == 2


def test_decide_holds_when_drift_is_a_single_spike() -> None:
    decision = decide(**healthy_kwargs(drift_history=drift_rows(1)))
    assert decision.should_retrain is False
    assert any("not sustained" in b for b in decision.blocked_by)


# ── Incident 06: the scenario every other gate waves through ──────────
def test_incident_06_seasonal_drift_must_not_auto_promote() -> None:
    """Ramadan: real drift, sustained, healthy pipeline, plenty of labels, no
    cooldown — and a challenger that beats the champion on a random split of the
    anomalous window. Four gates say yes. The gate must still refuse.

    The random split is the trap: both halves are drawn from the same three
    anomalous weeks, so the challenger is validated against the very regime it
    overfitted. It will look excellent and be wrong for the other eleven months.
    """
    ramadan = datetime(2026, 3, 1, 12, 0)
    decision = decide(
        drift_history=drift_rows(6),  # sustained: yes
        checks={"schema": True, "freshness": True, "nulls": True},  # healthy: yes
        new_labels=12_000,  # labelled: yes
        last_retrain=ramadan - timedelta(days=45),  # cooldown: clear
        now=ramadan,
    )

    assert decision.should_retrain is False, "seasonal drift must not auto-promote"
    assert len(decision.reasons) == 4, "the other four gates should all have passed"
    assert len(decision.blocked_by) == 1, "seasonal is the only thing standing in the way"

    why = decision.blocked_by[0]
    assert "ramadan-2026" in why, "the refusal must name the window"
    assert "RANDOM split" in why, "and explain why the challenger's score is not evidence"


def test_incident_06_the_same_drift_does_promote_once_the_season_ends() -> None:
    """The gate blocks the WINDOW, not the feature. After it, the answer flips."""
    after = datetime(2026, 4, 20, 12, 0)
    decision = decide(
        **healthy_kwargs(
            now=after,
            last_retrain=after - timedelta(days=45),
            drift_history=[
                {"window_end": (after - timedelta(days=d)).isoformat(), "value": 0.41}
                for d in (1, 2, 3)
            ],
        )
    )
    assert decision.should_retrain is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ── Incident 06 must be teachable in any month ────────────────────────
def test_simulated_seasonal_window_opens_today() -> None:
    """Ramadan moves ~11 days earlier each year, so a lab in September sits
    nowhere near it. Incident 06 declares a window covering today, or the
    refusal that is the entire lesson never fires in a live class."""
    from incidents import state
    from jobs.retrain_gate import _calendar_for

    before = state.read_state()
    try:
        state.write_state([], {})
        assert seasonal_block(datetime.now(), _calendar_for(datetime.now())) is None

        state.write_state(["06"], {"SIMULATE_SEASONAL_WINDOW": "ramadan-2026"})
        now = datetime.now()
        assert seasonal_block(now, _calendar_for(now)) == "ramadan-2026"
    finally:
        state.write_state(before.get("active", []), before.get("flags", {}))


def test_the_real_calendar_is_still_consulted_when_nothing_is_injected() -> None:
    """The override must ADD a window, never replace the real ones."""
    from incidents import state
    from jobs.retrain_gate import _calendar_for

    before = state.read_state()
    try:
        state.write_state(["06"], {"SIMULATE_SEASONAL_WINDOW": "simulated"})
        calendar = _calendar_for(datetime.now())
        assert "ramadan-2027" in calendar, "the real windows must survive the override"
        assert seasonal_block(datetime(2027, 2, 20), calendar) == "ramadan-2027"
    finally:
        state.write_state(before.get("active", []), before.get("flags", {}))
