"""Feature engineering: pickup-dropoff pair + trip distance -> feature dict.

Kept as plain Python dicts (not a fitted transformer) so the same function
serves both the training pipeline (via DictVectorizer) and single-request
inference in the API, with no train/serve skew.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_FIELDS = ("PU_DO", "trip_distance")


def build_pu_do(pu: int | str, do: int | str) -> str:
    return f"{pu}_{do}"


def row_to_features(pu: int | str, do: int | str, trip_distance: float) -> dict[str, float | str]:
    """Build the feature dict for a single ride — the exact contract the API validates against."""
    if trip_distance is None or trip_distance <= 0:
        raise ValueError(f"trip_distance must be > 0, got {trip_distance!r}")
    return {"PU_DO": build_pu_do(pu, do), "trip_distance": float(trip_distance)}


def dataframe_to_features(df: pd.DataFrame) -> list[dict[str, float | str]]:
    """Vectorized version of row_to_features for a whole training dataframe."""
    pu_do = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)
    out = pd.DataFrame({"PU_DO": pu_do, "trip_distance": df["trip_distance"].astype(float)})
    return out.to_dict(orient="records")
