# Postmortem — incident 01: the upstream feed switched to miles

**Date:** 2026-09-08 · **Detected by:** a human, from an ops complaint ·
**Time to detect:** 34 min (budget: 30) · **Time to resolve:** 9 min

---

## The five sentences

1. **Context.** The ride-duration model had been serving cairo traffic at a
   steady mean predicted duration of ~38.7 minutes and an MAE of ~2.9 minutes
   against late labels, with every input feature's PSI below 0.02 against the
   frozen `train-v3-50k` reference.
2. **Signal.** Ops reported ETAs "looking way too optimistic"; the Grafana
   **Predicted duration distribution** heatmap showed the whole distribution
   shifted down as a block at 09:12, and `jobs/daily_drift` put `distance_km`
   at **PSI 2.364** against a frozen reference where it had been 0.011 the day
   before — an *input* feature, not the output.
3. **Action.** I first checked `up{job="model-api"}` and the deploy timeline
   (`SELECT * FROM deploys ORDER BY ts DESC`) and found nothing — no deploy, no
   restart, no error rate — which was the wrong place to look and cost about
   fifteen minutes; the drift report then pointed at `distance_km`, and its max
   value had dropped from ~30 to ~18.6, which is 30 × 0.621.
4. **Outcome.** The upstream trip feed had begun sending **miles in a field
   named `distance_km`**; every prediction was ~38% short (mean predicted
   duration 38.7 → **23.5 min**, prediction PSI 1.742). Reverting the producer
   restored both within one scrape interval.
5. **Lesson.** No schema validator can catch this, because miles are perfectly
   valid floats in a float field — the only thing that catches a unit change is
   a **distribution or a range assertion on the values themselves**, so we added
   a range check on `distance_km` to the pipeline health checks that feed
   `retrain_gate.pipeline_is_healthy()`.

---

## Evidence

```
# jobs/daily_drift.py — the window before and the window after

HEALTHY
  distance_km  PSI= 0.011  ok
  passengers   PSI= 0.002  ok
  hour_of_day  PSI= 0.004  ok
  prediction   PSI= 0.008  ok

AFTER
  distance_km  PSI= 2.364  ACT      <-- an INPUT moved, on its own
  passengers   PSI= 0.000  ok
  hour_of_day  PSI= 0.007  ok
  prediction   PSI= 1.742  ACT      <-- the output followed it
```

```
mean predicted duration: 38.7 min -> 23.5 min
max(distance_km):        30.0     -> 18.6      (30.0 x 0.621371 = 18.64)
error rate:              unchanged at 0
up{job="model-api"}:     1 throughout
```

## Timeline

| Time | Event |
|---|---|
| 09:12 | Upstream feed switches units. No deploy on our side, no alert. |
| 09:31 | Ops reports "ETAs look too optimistic". |
| 09:38 | Checked `up`, error rate, deploy timeline — all clean. Wrong branch. |
| 09:46 | Ran `jobs/daily_drift`; `distance_km` PSI 2.364. |
| 09:49 | Compared feature ranges; 30 → 18.6 identified as the km→miles factor. |
| 09:55 | Upstream reverted; distribution recovered on the next scrape. |

## What made this hard to find

**Nothing failed.** No error, no latency change, no deploy, `up == 1` the whole
time. Every instrument that watches for *breakage* was green, because nothing
broke — the values were all valid, they just meant something different. The
signal only existed in the input distribution, which is the one place this repo
looks precisely because of failures like this one.

The fifteen minutes lost to the deploy timeline were not wasted, though: ruling
out a change is step 3 of the runbook for a reason, and it is right far more
often than it was here.

This is the same bug as session 1's non-idempotent miles→km notebook cell,
promoted to production and handed to a different team.

## Follow-up

- [x] Range assertion on `distance_km` in the pipeline health checks
- [x] Runbook step 2 now says explicitly that a unit change passes schema validation
- [ ] Ask the feed's owner for a schema-change notification, including units
