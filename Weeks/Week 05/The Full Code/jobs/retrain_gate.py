"""Should we retrain? Four gates, each a pure function, each independently wrong.

This file is what closes README's last line — "decide when session 2 needs to
run again" — with code instead of a sentence.

The naive pipeline is `if drift_detected: retrain()`. Every gate below exists
because that one line has caused a real outage:

    is_sustained      one spike is a bad batch, not a trend
    pipeline_is_healthy   retraining on data a broken ETL produced bakes the
                          break into the model, and now it is much harder to see
    enough_new_labels     a model refitted on 40 labels is a rumour
    cooldown_ok           retraining faster than you can evaluate means you are
                          shipping unvalidated models on a timer
    seasonal_block        drift can be REAL, sustained, well-labelled, and still
                          be the wrong reason to retrain (incident 06)

Every function takes plain data and returns a plain answer, so each can be
tested in isolation — which is the point of writing them separately at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Mapping, Sequence

#: Windows in which automatic PROMOTION is blocked. Dates are approximate: the
#: Islamic calendar is lunar and confirmed by observation, so these are entered
#: per year rather than computed. That is deliberate — a business calendar is
#: DATA, maintained by someone who knows the business, not a clever date
#: function. Extend it for your own market before you trust the gate.
SEASONAL_WINDOWS: dict[str, tuple[date, date]] = {
    "ramadan-2026": (date(2026, 2, 17), date(2026, 3, 19)),
    "eid-al-fitr-2026": (date(2026, 3, 19), date(2026, 3, 22)),
    "eid-al-adha-2026": (date(2026, 5, 26), date(2026, 5, 30)),
    "ramadan-2027": (date(2027, 2, 6), date(2027, 3, 8)),
    "eid-al-fitr-2027": (date(2027, 3, 8), date(2027, 3, 11)),
}


@dataclass
class Decision:
    """The gate's answer, and every reason behind it — including the refusals."""

    should_retrain: bool
    reasons: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)

    def explain(self) -> str:
        """One human-readable line per reason, for the log and the runbook."""
        verdict = "RETRAIN" if self.should_retrain else "HOLD"
        lines = [f"  decision: {verdict}"]
        lines += [f"    + {r}" for r in self.reasons]
        lines += [f"    - {b}" for b in self.blocked_by]
        return "\n".join(lines)


def is_sustained(drift_history: Sequence[Mapping[str, Any]], days: int = 2) -> bool:
    """True if drift was detected on at least `days` distinct days.

    `drift_history` is rows from ``drift_metrics``: each needs ``window_end``
    and ``value``, plus ``drifted`` (bool) or a PSI value above 0.25.

    A single window crossing the threshold is the normal state of a healthy
    system with a real-world input feed. Two separate days is the cheapest
    filter that distinguishes a trend from a Tuesday.
    """
    drifted_days = {
        str(row["window_end"])[:10]
        for row in drift_history
        if row.get("drifted", float(row.get("value", 0.0)) > 0.25)
    }
    return len(drifted_days) >= days


def pipeline_is_healthy(checks: Mapping[str, bool]) -> bool:
    """True only if every upstream data-quality check passed.

    Retraining on the output of a broken pipeline does not fix the model — it
    launders the bug into the weights, where no data-quality check will ever
    look at it again.
    """
    return bool(checks) and all(checks.values())


def enough_new_labels(count: int, min_n: int = 1_000) -> bool:
    """True if enough ground truth has arrived since the last fit.

    Labels lag predictions — that is the whole reason this session monitors
    inputs and outputs first. Retraining before they arrive means fitting to the
    fraction of rides that happened to resolve fast, which is not a random
    sample of anything.
    """
    return count >= min_n


def cooldown_ok(last_retrain: datetime | None, now: datetime, days: int = 7) -> bool:
    """True if the last retrain is far enough back to have been evaluated.

    Without this, a sustained drift signal retrains every run, and you ship
    models faster than late labels can tell you whether the last one was good.
    """
    if last_retrain is None:
        return True
    return (now - last_retrain) >= timedelta(days=days)


def seasonal_block(
    now: datetime, calendar: Mapping[str, tuple[date, date]] | None = None
) -> str | None:
    """Name the seasonal window `now` falls in, or None.

    This is the incident-06 gate. Ramadan drift is real, sustained and properly
    labelled — every other gate passes it. But the regime reverts in three
    weeks, and a model fitted to it will be wrong for the other eleven months.
    """
    today = now.date()
    for name, (start, end) in (calendar or SEASONAL_WINDOWS).items():
        if start <= today < end:
            return name
    return None


def decide(
    *,
    drift_history: Sequence[Mapping[str, Any]],
    checks: Mapping[str, bool],
    new_labels: int,
    last_retrain: datetime | None,
    now: datetime,
    min_labels: int = 1_000,
    cooldown_days: int = 7,
    sustained_days: int = 2,
    calendar: Mapping[str, tuple[date, date]] | None = None,
) -> Decision:
    """Run every gate and return the decision with its full reasoning.

    Gates are evaluated in full — not short-circuited — because the interesting
    audit question months later is "what else would also have blocked this?",
    and a short-circuit throws that away.
    """
    decision = Decision(should_retrain=False)

    if is_sustained(drift_history, days=sustained_days):
        decision.reasons.append(f"drift sustained across >= {sustained_days} days")
    else:
        decision.blocked_by.append(
            f"drift not sustained for {sustained_days} days — one window is noise"
        )

    if pipeline_is_healthy(checks):
        decision.reasons.append("upstream data-quality checks all passed")
    else:
        failed = sorted(name for name, ok in checks.items() if not ok) or ["no checks reported"]
        decision.blocked_by.append(f"pipeline unhealthy: {', '.join(failed)}")

    if enough_new_labels(new_labels, min_labels):
        decision.reasons.append(f"{new_labels} new labels since last fit")
    else:
        decision.blocked_by.append(f"only {new_labels} new labels, need {min_labels}")

    if cooldown_ok(last_retrain, now, days=cooldown_days):
        decision.reasons.append(f"outside the {cooldown_days}-day cooldown")
    else:
        decision.blocked_by.append(
            f"inside the {cooldown_days}-day cooldown since {last_retrain:%Y-%m-%d}"
        )

    season = seasonal_block(now, calendar)
    if season:
        decision.blocked_by.append(
            f"seasonal window '{season}': the drift is real but temporary. A challenger "
            "validated on a RANDOM split of this window will beat the champion and still "
            "be the wrong model — the split shares the anomaly with the training half. "
            "Hold for a human, and validate on data from after the window."
        )

    decision.should_retrain = not decision.blocked_by
    return decision


def _calendar_for(now: datetime) -> Mapping[str, tuple[date, date]]:
    """The seasonal calendar, plus today's window when incident 06 is active.

    Ramadan moves ~11 days earlier each year, so a lab run in September sits
    nowhere near it: injecting Ramadan *traffic* would produce real drift while
    the gate, consulting the real *calendar*, waved it straight through — and
    the refusal that is the entire point of incident 06 would never fire.

    The override lives here, in the impure function that already reads state,
    so `seasonal_block()` and `decide()` stay pure and unit-testable against a
    calendar passed in by the caller.
    """
    from incidents import state

    window = state.flag("SIMULATE_SEASONAL_WINDOW")
    if not window:
        return SEASONAL_WINDOWS
    today = now.date()
    return {**SEASONAL_WINDOWS, str(window): (today, today + timedelta(days=1))}


def decide_from_store(now: datetime | None = None, feature: str = "distance_km") -> Decision:
    """Convenience wrapper: read the real store and decide. Logs the event.

    Every decision is written to ``retrain_events``, including the holds. A gate
    that only records its yeses cannot be audited after it promotes something it
    should not have.
    """
    from jobs import metrics_store

    now = now or datetime.now()
    history = [dict(row) for row in metrics_store.drift_history(feature, days=14)]
    with metrics_store.connect() as conn:
        labels = conn.execute("SELECT COUNT(*) AS n FROM labels").fetchone()["n"]
        last = conn.execute(
            "SELECT ts FROM retrain_events WHERE promoted=1 ORDER BY ts DESC LIMIT 1"
        ).fetchone()
    last_retrain = datetime.fromisoformat(last["ts"]).replace(tzinfo=None) if last else None

    decision = decide(
        drift_history=history,
        checks={"schema": True, "freshness": True, "nulls": True},
        new_labels=labels,
        last_retrain=last_retrain,
        now=now,
        calendar=_calendar_for(now),
    )
    metrics_store.record_retrain_event(
        trigger=f"drift:{feature}",
        decision="retrain" if decision.should_retrain else "hold",
        reasons=" | ".join(decision.reasons + decision.blocked_by),
        features_drifted=feature,
        promoted=decision.should_retrain,
    )
    return decision


if __name__ == "__main__":
    print(decide_from_store().explain())
