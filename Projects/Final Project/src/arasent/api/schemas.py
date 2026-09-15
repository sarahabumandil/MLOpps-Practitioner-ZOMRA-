
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Arabic review text")

    model_config = {
        "json_schema_extra": {"example": {"text": "المنتج رائع جدا وسريع في التوصيل"}}
    }


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    model_version: str
    correlation_id: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    reviews: list[PredictionRequest]


class BatchPredictionfrom __future__ import annotations
Response(BaseModel):
    predictions: list[dict]
    model_version: str
    correlation_id: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetadataResponse(BaseModel):
    model_version: str
    framework: str
    labels: list[str]
    backend: str
