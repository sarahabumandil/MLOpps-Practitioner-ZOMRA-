from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from prodml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml.config import settings
from prodml.logging_conf import configure_logging, correlation_id_var, get_logger
from prodml.predict import DurationPredictor

logger = get_logger(__name__)
predictor = DurationPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    # Model is loaded exactly once at startup — never per-request. Loading
    # per-request is a 100x latency mistake the handbook calls out explicitly.
    try:
        predictor.load()
    except FileNotFoundError as exc:
        logger.error("startup_model_load_failed", extra={"error": str(exc)})
    yield


app = FastAPI(title="Dentiligence · prodml", version=settings.model_version, lifespan=lifespan)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    correlation_id = str(uuid.uuid4())
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
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", extra={"error": str(exc)}, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    # 200 only if the model object is actually loaded in memory.
    status = "healthy" if predictor.is_loaded else "unhealthy"
    return HealthResponse(status=status, model_loaded=predictor.is_loaded)


@app.get("/metadata", response_model=MetadataResponse)
async def metadata() -> MetadataResponse:
    return MetadataResponse(
        model_version=predictor.version,
        framework="scikit-learn (RandomForestRegressor + DictVectorizer)",
        feature_names=["PU_DO", "trip_distance"],
        trained_on="NYC TLC-style ride duration data (see data.py)",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest) -> PredictionResponse:
    start = time.perf_counter()
    pred = predictor.predict_one(req.pu_location_id, req.do_location_id, req.trip_distance)
    latency_ms = (time.perf_counter() - start) * 1000
    return PredictionResponse(
        prediction=round(pred, 2),
        model_version=predictor.version,
        correlation_id=correlation_id_var.get(),
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(req: BatchPredictionRequest) -> BatchPredictionResponse:
    start = time.perf_counter()
    rows = [r.model_dump() for r in req.rides]
    preds = predictor.predict_batch(rows)
    latency_ms = (time.perf_counter() - start) * 1000
    return BatchPredictionResponse(
        predictions=[round(p, 2) for p in preds],
        model_version=predictor.version,
        correlation_id=correlation_id_var.get(),
        latency_ms=round(latency_ms, 3),
    )
