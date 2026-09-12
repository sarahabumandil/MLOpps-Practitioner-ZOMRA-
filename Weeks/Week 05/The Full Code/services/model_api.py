"""The instrumented ride-duration service — P_G_monitoring.py, made runnable.

P_G_monitoring.py is a teaching snippet: it references an ``app`` that does not
exist. This is that snippet as a service the incidents can actually break, using
THE SAME metric names, because monitoring/grafana/dashboards/model-monitoring.json
and monitoring/prometheus/alert_rules.yml already query them.

    pip install -e ".[metrics]"
    python -m services.ride_model          # once — fit and freeze artifacts/
    make api                              # uvicorn on :8001, scraped by Prometheus

Metrics are regression metrics. There is no ml_prediction_score, no positive
rate and no outcome label: this model predicts minutes, so its output
distribution is a histogram of minutes and its error is MAE. A single uvicorn
worker keeps the Gauges and the prediction cache coherent — the multiprocess
collector is a separate lesson, noted in the README.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from incidents import state
from jobs import metrics_store
from services.ride_model import ARTIFACTS, DATA, FEATURES, true_duration

MODEL_NAME = os.getenv("MODEL_NAME", "rf_model")
CHAMPION_VERSION = "v3"
CANARY_VERSION = "v4"

# ══════════════════════════════════════════════════════════════════════
#  Metrics — names fixed by the existing dashboard and alert rules
# ══════════════════════════════════════════════════════════════════════
PREDICTION_HISTOGRAM = Histogram(
    "model_prediction_duration_min",
    "Distribution of predicted ride durations",
    buckets=[0, 5, 10, 15, 20, 30, 45, 60, 90, 120],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "End-to-end API latency", ["endpoint", "status"]
)
DRIFT_GAUGE = Gauge("feature_psi_score", "PSI drift score per feature", ["feature"])
MODEL_VERSION = Gauge("model_version_info", "Active model version", ["version", "stage"])

# ── Added for session 4's incidents, all regression-shaped ────────────
FEEDBACK_TOTAL = Counter("feedback_total", "Late labels received")  # incident 15
SEGMENT_PSI = Gauge(
    "segment_psi_score", "PSI per feature, split by city", ["feature", "city"]
)  # incident 05 — the split that the global gauge above cannot show
SEGMENT_MAE = Gauge("model_mae_minutes", "MAE on late labels, by city", ["city"])
DISK_FREE = Gauge("disk_free_bytes", "Free space on the log volume", ["path"])
DEPLOY_INFO = Gauge("deploy_info", "Unix ts of the last change", ["component", "version"])

#: Registered lazily, only while incident 04 is active — see _record_ride_id().
_ride_id_counter: Counter | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the frozen artifacts and publish the champion version."""
    global _model, _scaler, _scaler_stale
    _model = joblib.load(ARTIFACTS / "rf_model.pkl")
    _scaler = joblib.load(ARTIFACTS / "scaler.pkl")
    _scaler_stale = joblib.load(ARTIFACTS / "scaler_stale.pkl")
    MODEL_VERSION.labels(version=CHAMPION_VERSION, stage="production").set(1)
    yield


app = FastAPI(title="Ride Duration API (session 4)", lifespan=lifespan)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus scrape target.

    An explicit route, not ``app.mount("/metrics", make_asgi_app())``. The mount
    is the idiomatic one-liner, but Starlette then answers ``GET /metrics`` with
    a 307 to ``/metrics/``. Prometheus follows redirects so scraping still works
    — but ``curl localhost:8001/metrics`` prints nothing, and that curl is the
    first thing anyone does when a target goes down. Costing a lab twenty minutes
    to save one line is a bad trade.
    """
    _refresh_disk_gauge()
    _refresh_deploy_gauge()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


_model: Any = None
_scaler: Any = None
_scaler_stale: Any = None
#: request_id -> (predicted_min, city). Bounded: late labels arrive within hours,
#: not weeks, and an unbounded dict here would be its own incident.
_recent: OrderedDict[str, tuple[float, str]] = OrderedDict()


class PredictRequest(BaseModel):
    """One trip. `city` is a segment label, not a model feature (incident 05)."""

    distance_km: float = Field(gt=0, le=500)
    passengers: int = Field(ge=1, le=8)
    hour_of_day: int = Field(ge=0, le=23)
    city: str = "cairo"
    ride_id: str | None = None


class FeedbackRequest(BaseModel):
    """The actual duration, arriving late — this is what makes MAE possible."""

    request_id: str
    actual_duration_min: float


SERVING_LOG = DATA / "serving_log.jsonl"
_log_buffer: list[str] = []


def _log_serving(row: dict[str, Any]) -> None:
    """Append the exact feature vector the model scored, and what it returned.

    Incident 02 is invisible to every distribution monitor in this repo: the
    inputs are fine, the model is fine, and only the transform between them is
    stale. Nothing about the request looks wrong. The one thing that catches it
    is re-scoring these logged vectors through the offline pipeline and
    comparing (tools/replay.py) — which is what a serving log is FOR.
    """
    _log_buffer.append(json.dumps(row))
    if len(_log_buffer) >= 50:
        DATA.mkdir(parents=True, exist_ok=True)
        with open(SERVING_LOG, "a") as fh:
            fh.write("\n".join(_log_buffer) + "\n")
        _log_buffer.clear()


def _record_ride_id(ride_id: str) -> None:
    """Incident 04: mint one time series per ride. Registered on first use only,
    so the cardinality guard in tests/ passes on a healthy system and fails here."""
    global _ride_id_counter
    if _ride_id_counter is None:
        _ride_id_counter = Counter(
            "model_predictions_by_ride_total", "Predictions per ride id", ["ride_id"]
        )
    _ride_id_counter.labels(ride_id=ride_id).inc()


def _drop_ride_id_metric() -> None:
    """Unregister the exploded metric so `make heal` alone recovers the scrape.

    Removing the label from the code is not enough in a live process: the series
    already registered keep being exported, so the scrape stays over
    sample_limit and the target stays down. Real recovery means dropping the
    collector — which in production is the restart everyone forgets they need.
    """
    global _ride_id_counter
    if _ride_id_counter is not None:
        REGISTRY.unregister(_ride_id_counter)
        _ride_id_counter = None


def _active_version() -> str:
    """Incident 10: a canary that was never completed still splits traffic."""
    if not state.flag("SERVE_SECOND_VERSION"):
        return CHAMPION_VERSION
    MODEL_VERSION.labels(version=CANARY_VERSION, stage="canary").set(1)
    return CANARY_VERSION if uuid.uuid4().int % 2 else CHAMPION_VERSION


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    """Predict ride duration in minutes and record the four teaching metrics."""
    t0 = time.perf_counter()
    version = _active_version()

    # Incident 02: the preprocessing from before the last retrain. Same model,
    # same inputs, different transform — and nothing in the request looks wrong.
    scaler = _scaler_stale if state.flag("USE_STALE_SCALER") else _scaler
    x = np.array([[req.distance_km, float(req.passengers), float(req.hour_of_day)]])
    minutes = float(_model.predict(scaler.transform(x))[0])
    if version == CANARY_VERSION:
        minutes *= 1.09  # the canary is a different fit, not a different model

    request_id = req.ride_id or uuid.uuid4().hex
    _recent[request_id] = (minutes, req.city)
    while len(_recent) > 20_000:
        _recent.popitem(last=False)

    _log_serving(
        {
            "request_id": request_id,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "distance_km": req.distance_km,
            "passengers": req.passengers,
            "hour_of_day": req.hour_of_day,
            "city": req.city,
            "served_min": minutes,
            "version": version,
        }
    )
    PREDICTION_HISTOGRAM.observe(minutes)
    REQUEST_LATENCY.labels(endpoint="/predict", status="200").observe(time.perf_counter() - t0)
    if state.flag("ENABLE_RIDE_ID_LABEL"):
        _record_ride_id(request_id)
    elif _ride_id_counter is not None:
        _drop_ride_id_metric()

    return {"duration_min": round(minutes, 2), "model_version": version, "request_id": request_id}


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict[str, str]:
    """Accept a late label. Incident 15 makes this silently do nothing."""
    if state.flag("FEEDBACK_DEAD"):
        return {"status": "accepted"}  # the lie that keeps every dashboard green
    predicted, city = _recent.get(req.request_id, (None, "cairo"))
    if predicted is None:
        return {"status": "unknown request_id"}
    FEEDBACK_TOTAL.inc()
    metrics_store.record_labels(
        [
            (
                req.request_id,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                city,
                req.actual_duration_min,
                predicted,
            )
        ]
    )
    return {"status": "recorded"}


@app.post("/internal/gauges")
def publish_gauges(payload: dict[str, Any]) -> dict[str, int]:
    """Publish drift/error numbers computed by jobs/daily_drift.py.

    P_G_monitoring.py's record_drift_scores() assumes the drift check runs
    in-process. It does not — it is a separate job — and adding a Pushgateway
    would mean a seventh container. So the job posts its numbers to the one
    process Prometheus already scrapes.
    """
    n = 0
    for feature, value in (payload.get("feature_psi") or {}).items():
        DRIFT_GAUGE.labels(feature=feature).set(float(value))
        n += 1
    for row in payload.get("segment_psi") or []:
        SEGMENT_PSI.labels(feature=row["feature"], city=row["city"]).set(float(row["value"]))
        n += 1
    for row in payload.get("mae") or []:
        SEGMENT_MAE.labels(city=row["city"]).set(float(row["value"]))
        n += 1
    return {"published": n}


def _refresh_deploy_gauge() -> None:
    """Publish the most recent deploy so Grafana can draw the annotation.

    RUNBOOK step 3 is "what changed, and when" — and the answer is almost always
    a deploy. The deploys table holds it, but Grafana's annotation layer reads
    time series, so the latest row is republished here as an info metric whose
    VALUE is the deploy's unix timestamp.

    .clear() first: a new deploy means new label values, and without it every
    past version would linger as its own series forever.
    """
    rows = metrics_store.latest_deploys(limit=1)
    if not rows:
        return
    row = rows[0]
    ts = datetime.fromisoformat(row["ts"]).timestamp()
    DEPLOY_INFO.clear()
    DEPLOY_INFO.labels(component=row["component"], version=row["version"]).set(ts)


def _refresh_disk_gauge() -> None:
    """Recompute free space. Called on every SCRAPE, not just on /health.

    A gauge that is only refreshed by an endpoint nobody scrapes is a flat line
    that looks like a healthy disk forever. Prometheus reads /metrics, so the
    reading has to happen there.
    """
    usage = shutil.disk_usage(Path(__file__).resolve().parents[1])
    free = float(usage.free)
    if state.flag("SIMULATE_DISK_FILL"):
        # Simulated, not real: filling a student's laptop mid-session is not a
        # lesson, it is a support ticket. The gauge falls linearly, which is all
        # predict_linear needs to extrapolate a time-to-full.
        started = state.read_state().get("updated")
        if started:
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds()
            free = max(0.0, free - elapsed * (usage.free / 3600.0) * 0.22)
    DISK_FREE.labels(path="/var/log").set(free)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness, and the version actually serving right now."""
    _refresh_disk_gauge()
    return {"status": "ok", "model": MODEL_NAME, "version": _active_version()}


def offline_score(distance_km: float, passengers: int, hour_of_day: int) -> float:
    """Score with the CURRENT preprocessing — the offline pipeline's answer.

    tools/replay.py compares this against what the API returned. When they differ
    on identical inputs, the difference is the serving path (incident 02).
    """
    x = np.array([[distance_km, float(passengers), float(hour_of_day)]])
    return float(_model.predict(_scaler.transform(x))[0])


__all__ = ["app", "FEATURES", "true_duration", "offline_score"]
