from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    pu_location_id: int = Field(..., ge=1, le=265, description="Pickup TLC zone ID")
    do_location_id: int = Field(..., ge=1, le=265, description="Dropoff TLC zone ID")
    trip_distance: float = Field(..., gt=0, lt=200, description="Trip distance in miles")

    model_config = {
        "json_schema_extra": {
            "example": {"pu_location_id": 43, "do_location_id": 151, "trip_distance": 3.2}
        }
    }


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted duration in minutes")
    model_version: str
    correlation_id: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    rides: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    model_version: str
    correlation_id: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetadataResponse(BaseModel):
    model_version: str
    framework: str
    feature_names: list[str]
    trained_on: str
