"""The incident catalogue: what breaks, what a user notices, what catches it.

Every entry separates two audiences.

``symptom`` and ``ttd_minutes`` are what ``inject.py`` prints. They are written
in the voice of the person who files the ticket, and they never name a cause —
the whole exercise is worthless if the terminal spoils the answer.

``cause``, ``signals`` and ``teaches`` are instructor-only, reachable with
``--reveal``. Use them to close the lab, and to grade the postmortems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Incident:
    """One injectable failure, mapped onto the ride-duration model."""

    id: str
    component: str
    symptom: str
    ttd_minutes: int
    flags: dict[str, Any] = field(default_factory=dict)
    cause: str = ""
    signals: tuple[str, ...] = ()
    teaches: str = ""
    kind: str = "break"  # break | measurement
    implemented: bool = True


CATALOG: dict[str, Incident] = {
    "01": Incident(
        id="01",
        component="upstream-feed",
        symptom=(
            "Ops says every ETA has looked 'way too optimistic' since this morning. "
            "No errors, no alerts, no failed requests — the app just quotes short."
        ),
        ttd_minutes=30,
        flags={"FEED_DISTANCE_IN_MILES": True},
        cause=(
            "The upstream trip feed switched from kilometres to miles (x0.621). The API "
            "still reads the field as km, so every distance arrives 38% small and every "
            "predicted duration shrinks with it. Nothing raises: miles are valid floats."
        ),
        signals=(
            "Grafana 'Predicted duration distribution' heatmap shifts down as a block",
            "histogram_quantile(0.50, sum by (le)"
            " (rate(model_prediction_duration_min_bucket[10m])))",
            "PSI on distance_km in the drift report — large, and on the input, not the output",
            "feature range assertion: max(distance_km) drops from ~30 to ~19",
        ),
        teaches=(
            "The unit bug from session 1's non-idempotent miles->km notebook cell, now in "
            "production. A schema validator passes it. Only a distribution catches it."
        ),
    ),
    "02": Incident(
        id="02",
        component="model-api",
        symptom=(
            "A data scientist re-scored yesterday's requests offline and got different "
            "numbers than the API returned for the same rides. Nobody deployed a model."
        ),
        ttd_minutes=120,
        flags={"USE_STALE_SCALER": True},
        cause=(
            "The serving container loads artifacts/scaler_stale.pkl — the preprocessing "
            "fitted before the last retrain — while the offline pipeline uses the current "
            "one. Same model, different inputs. Classic train/serve skew."
        ),
        signals=(
            "Replay: python -m tools.replay --hours 24 — serving vs offline scores diverge",
            "MAE on late labels degrades while every input distribution stays stationary",
            "No drift signal at all: the inputs are fine, the transform is not",
        ),
        teaches="Drift monitoring cannot see this. Only replaying logged features can.",
    ),
    "04": Incident(
        id="04",
        component="model-api",
        symptom=(
            "Grafana panels went blank around 14:10 and the Prometheus UI is sluggish. "
            "The API itself is healthy — /predict still returns 200 in single digits ms."
        ),
        ttd_minutes=10,
        flags={"ENABLE_RIDE_ID_LABEL": True},
        cause=(
            "Someone added ride_id as a Prometheus label 'for debuggability'. Every "
            "request now mints a new time series. The scrape trips sample_limit and "
            "Prometheus drops the whole target — so the dashboards go blank, not noisy."
        ),
        signals=(
            "prometheus_tsdb_head_series — a near-vertical line",
            'up{job="model-api"} flips to 0 while /metrics is reachable by hand',
            "Prometheus > Targets shows: sample limit exceeded",
            "topk(5, count by (__name__)({__name__=~'model.*'}))",
        ),
        teaches=(
            "Cardinality is a per-label-VALUE cost. sample_limit turns an OOM into a "
            "loud, local, recoverable failure — which is why the scrape job sets it."
        ),
    ),
    "05": Incident(
        id="05",
        component="upstream-feed",
        symptom=(
            "Complaints about bad ETAs are up — but only a trickle, and only from one "
            "region. Overall accuracy on the dashboard is unchanged."
        ),
        ttd_minutes=180,
        flags={"NEW_CITY_ROLLOUT": True},
        cause=(
            "Launch into Alexandria. Its distance/duration relationship differs from "
            "Cairo's (coastal corniche traffic, different average speed), and it is only "
            "~8% of volume — so the global aggregate absorbs it completely."
        ),
        signals=(
            "segment_psi_score{feature='distance_km'} by city — moves for alexandria only",
            "feature_psi_score (global) — flat. This is the point.",
            "model_mae_minutes by city, on late labels",
            "SQL: select city, avg(abs(y_true-y_pred)) from labels group by city",
        ),
        teaches=(
            "A healthy average is the most common way a real segment failure stays "
            "invisible. Aggregate first, and you will not see it. Ever."
        ),
    ),
    "06": Incident(
        id="06",
        component="upstream-feed",
        symptom=(
            "Drift alerts have been firing for two days and the automatic retrain "
            "promoted a new model overnight. Accuracy looks fine. Finance is asking why "
            "quoted durations jumped."
        ),
        ttd_minutes=1440,
        flags={"TRAFFIC_PROFILE": "ramadan", "SIMULATE_SEASONAL_WINDOW": "ramadan-2026"},
        cause=(
            "Ramadan shifts demand into the hours before iftar; durations inflate "
            "city-wide for a few weeks and then revert. The drift is real, so the naive "
            "gate fires, and the challenger fits the temporary regime beautifully."
        ),
        signals=(
            "retrain_events: decision='retrain' inside a seasonal window",
            "Drift is sustained AND seasonal — jobs/retrain_gate.py must refuse",
            "Challenger beats champion on a random split of the anomalous window only",
        ),
        teaches=(
            "Real drift is not sufficient grounds to retrain. A calendar is a gate, and "
            "a random split inside an anomalous window will lie to you."
        ),
    ),
    "07": Incident(
        id="07",
        component="rag-prompt",
        symptom=(
            "The assistant's answers 'read nicer' since this afternoon, but two people "
            "reported it stating things the docs do not say. No deploy went out."
        ),
        ttd_minutes=60,
        flags={"PROMPT_PRODUCTION_LABEL": "degraded"},
        cause=(
            "The production label was moved to the prompt version with the grounding "
            "sentence removed. No code changed, no container restarted, so the deploy "
            "timeline is empty — which is exactly why it takes an hour to find."
        ),
        signals=(
            "Langfuse: group faithfulness / judge_groundedness by prompt version",
            "The two versions are both live and both traced — filter, do not average",
            "Rollback is moving one label back, no deploy",
        ),
        teaches="A prompt is a deployable artifact. If it is not versioned, it is not observable.",
    ),
    "08": Incident(
        id="08",
        component="cost",
        symptom=(
            "Finance wants next quarter's inference budget, and wants to know what "
            "moving off self-hosted would cost."
        ),
        ttd_minutes=0,
        kind="measurement",
        cause=(
            "Not a break. Run tools/cost_report.py: it groups real traced token volume "
            "by prompt version, language and tag using the amortised GPU cost model in "
            "langfuse_workload.py, and reports the MIN and MAX week, never the average."
        ),
        signals=("python -m tools.cost_report --weeks 4",),
        teaches=(
            "Budgets are set by the peak, so an average week is the wrong number. And "
            "for a self-hosted model the honest unit is GPU-hours/token, not $/token."
        ),
    ),
    "09": Incident(
        id="09",
        component="rag-api",
        symptom=(
            "Arabic-speaking users say the assistant answers 'I don't know' to questions "
            "it handled last week. English users report nothing wrong."
        ),
        ttd_minutes=240,
        flags={"KB_NORMALISE_INDEX_ONLY": True},
        cause=(
            "The index build path folds Arabic text — strips diacritics and tatweel, "
            "unifies alef and teh marbuta — and the query path stopped doing it. The two "
            "sides no longer agree, so Arabic queries miss their own documents. English "
            "is untouched because every folding step is a no-op on Latin script."
        ),
        signals=(
            "Zero-hit rate and mean top-1 similarity, SPLIT BY LANGUAGE",
            "embedding_drift.py: cosine + MMD between last week's and this week's hits",
            "judge_language_match stays high — the bug is retrieval, not generation",
        ),
        teaches=(
            "A global retrieval metric hides a single-language collapse, and text "
            "normalisation must be identical on both sides of the index or it is a bug."
        ),
    ),
    "10": Incident(
        id="10",
        component="model-api",
        symptom=(
            "Two users sent the same trip and got ETAs four minutes apart, repeatedly. "
            "Both responses look plausible."
        ),
        ttd_minutes=45,
        flags={"SERVE_SECOND_VERSION": True},
        cause=(
            "A canary from session 3 was never completed: two model versions are live "
            "behind the same endpoint and traffic splits between them."
        ),
        signals=(
            "count(count by (version) (model_version_info)) > 1",
            "Latency and score distributions split by version, not in aggregate",
            "deploys table shows a promote with no matching retire",
        ),
        teaches="Session 3's blue/green and canary work, seen from the monitoring side.",
    ),
    "13": Incident(
        id="13",
        component="host",
        symptom="Nothing yet. Ask again in four hours.",
        ttd_minutes=240,
        flags={"SIMULATE_DISK_FILL": True},
        cause=(
            "A log rotation setting stopped rotating. Free space is falling linearly. "
            "SIMULATED: the API subtracts a growing figure from the real free-space "
            "reading rather than actually filling a student's disk."
        ),
        signals=(
            "predict_linear(disk_free_bytes[6h], 4*3600) < 0",
            "disk_free_bytes — a straight line, which is what makes it predictable",
        ),
        teaches=(
            "The only alert in the set that fires BEFORE the outage. Threshold alerts "
            "on a linear decline are strictly worse than extrapolation."
        ),
    ),
    "15": Incident(
        id="15",
        component="feedback-collector",
        symptom=(
            "None. Every dashboard is green. Model quality metrics have not moved in three days."
        ),
        ttd_minutes=360,
        flags={"FEEDBACK_DEAD": True},
        cause=(
            "The feedback endpoint silently drops submissions. No labels arrive, so MAE, "
            "the drift-vs-accuracy comparison and the retrain gate's enough_new_labels() "
            "all freeze at their last good value — looking healthy, measuring nothing."
        ),
        signals=(
            "rate(feedback_total[6h]) == 0",
            "labels table stops growing: select max(ts) from labels",
            "A flat quality metric is indistinguishable from a dead one without this",
        ),
        teaches=(
            "Monitor the monitoring. The most dangerous failure is the one that makes "
            "your instruments read 'fine' forever."
        ),
    ),
    # ── Homework (deck appendix). The exercise is to implement the injector AND
    #    the alert that detects it — see README, Homework #1.
    "11": Incident(id="11", component="?", symptom="", ttd_minutes=0, implemented=False),
    "12": Incident(id="12", component="?", symptom="", ttd_minutes=0, implemented=False),
    "14": Incident(id="14", component="?", symptom="", ttd_minutes=0, implemented=False),
}

#: Incident 03 (GPU thermal throttling) is deliberately absent. The ride-duration
#: model is a RandomForest on CPU, so a GPU panel here would be a lie. The Ollama
#: container IS GPU-resident, so it survives as a live instructor aside during the
#: RAGAS chapter — see README, "Incident 03".
NOT_INJECTABLE = {"03": "GPU thermal throttling — instructor aside, see README"}


def get(incident_id: str) -> Incident:
    """Look up an incident by its two-digit id, raising a readable KeyError."""
    key = incident_id.zfill(2)
    if key in NOT_INJECTABLE:
        raise KeyError(f"incident {key} is not injectable: {NOT_INJECTABLE[key]}")
    if key not in CATALOG:
        raise KeyError(f"unknown incident {key}. Try --list.")
    return CATALOG[key]
