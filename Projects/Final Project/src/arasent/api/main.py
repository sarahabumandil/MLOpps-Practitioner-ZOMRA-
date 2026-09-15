from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from arasent.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from arasent.logging_conf import configure_logging, correlation_id_var, get_logger
from arasent.predict import get_predictor

logger = get_logger(__name__)
_state: dict = {"predictor": None}

# --- Prometheus metrics (Module 5 contract: counters end in _total, histograms in base
# units, no unbounded labels like correlation_id) -----------------------------------
PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Total prediction requests", ["status", "label"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5],
)
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence", "Predicted class confidence — cheapest continuous drift signal",
    buckets=[0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    _state["predictor"] = get_predictor()
    logger.info("startup_model_loaded")
    yield
    _state.clear()


app = FastAPI(title="arasent — Arabic review sentiment API", lifespan=lifespan)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = correlation_id_var.set(correlation_id)
    try:
        response = await call_next(request)
    finally:
        correlation_id_var.reset(token)
    response.headers["X-Request-ID"] = correlation_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("validation_rejected", extra={"errors": exc.errors()})
    return JSONResponse(status_code=422, content={"detail": "Invalid request", "errors": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", extra={"error": str(exc)}, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse)
async def health():
    loaded = _state.get("predictor") is not None
    return HealthResponse(status="healthy" if loaded else "unhealthy", model_loaded=loaded)


@app.get("/metadata", response_model=MetadataResponse)
async def metadata():
    predictor = _state["predictor"]
    from arasent.config import LABELS

    return MetadataResponse(
        model_version=predictor.model_version,
        framework=predictor.framework,
        labels=LABELS,
        backend="transformer" if "arabert" in predictor.model_version else "baseline",
    )


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    predictor = _state["predictor"]
    start = time.perf_counter()
    result = predictor.predict_one(req.text)
    latency_s = time.perf_counter() - start

    PREDICTIONS_TOTAL.labels(status="ok", label=result["label"]).inc()
    PREDICTION_LATENCY.observe(latency_s)
    PREDICTION_CONFIDENCE.observe(result["confidence"])

    logger.info("prediction_served", extra={**result, "latency_ms": latency_s * 1000})
    return PredictionResponse(
        label=result["label"],
        confidence=result["confidence"],
        model_version=predictor.model_version,
        correlation_id=correlation_id_var.get(),
        latency_ms=latency_s * 1000,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(req: BatchPredictionRequest):
    predictor = _state["predictor"]
    start = time.perf_counter()
    texts = [r.text for r in req.reviews]
    results = predictor.predict_batch(texts)
    latency_s = time.perf_counter() - start

    for r in results:
        PREDICTIONS_TOTAL.labels(status="ok", label=r["label"]).inc()
        PREDICTION_CONFIDENCE.observe(r["confidence"])
    PREDICTION_LATENCY.observe(latency_s)

    logger.info("batch_prediction_served", extra={"n": len(results), "latency_ms": latency_s * 1000})
    return BatchPredictionResponse(
        predictions=results,
        model_version=predictor.model_version,
        correlation_id=correlation_id_var.get(),
        latency_ms=latency_s * 1000,
    )
