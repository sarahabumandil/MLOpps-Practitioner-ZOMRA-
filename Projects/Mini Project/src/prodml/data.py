"""Data loading and train/validation split.

The course reference dataset is NYC TLC green taxi trip data (Parquet).
This module tries to load a real Parquet file from ``settings.data_dir /
"raw"`` first. If none is found (e.g. on a fresh clone, or in a sandboxed
CI runner with no network access), it deterministically generates a
synthetic-but-realistic ride dataset with the same schema, so the rest of
the pipeline (features → train → predict → API) always runs end to end.

To use real data: download a month of NYC TLC green taxi Parquet data
(https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) into
``data/raw/green_tripdata_YYYY-MM.parquet`` and re-run ``prodml-train`` —
no code changes required.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from prodml.config import settings

logger = logging.getLogger(__name__)

RAW_COLUMNS = [
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
]


def _synthesize(n_rows: int = 20_000, seed: int = settings.random_seed) -> pd.DataFrame:
    """Generate a synthetic ride dataset with the NYC TLC green-taxi schema.

    Duration is a noisy function of distance and the pickup/dropoff zone
    pair, which is exactly the signal ``features.py`` is built to extract —
    swapping this for real Parquet data changes none of the downstream code.
    """
    rng = np.random.default_rng(seed)
    n_zones = 30

    pu = rng.integers(1, n_zones, size=n_rows)
    do = rng.integers(1, n_zones, size=n_rows)
    distance = np.round(rng.gamma(shape=2.0, scale=1.8, size=n_rows), 2)

    zone_effect = (np.abs(pu - do) / n_zones) * 25
    base_minutes = 4.0 + distance * 2.6 + zone_effect
    noise = rng.normal(0, 3.0, size=n_rows)
    duration_min = np.clip(base_minutes + noise, 1, 180)

    pickup = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 31 * 24 * 60, size=n_rows), unit="m"
    )
    dropoff = pickup + pd.to_timedelta(duration_min, unit="m")

    df = pd.DataFrame(
        {
            "PULocationID": pu,
            "DOLocationID": do,
            "trip_distance": distance,
            "lpep_pickup_datetime": pickup,
            "lpep_dropoff_datetime": dropoff,
        }
    )
    logger.info("synthesized_dataset", extra={"rows": n_rows})
    return df


def load_raw() -> pd.DataFrame:
    """Load raw trip data: real Parquet if present under data/raw/, else synthetic."""
    raw_dir = settings.data_dir / "raw"
    parquet_files = sorted(raw_dir.glob("*.parquet")) if raw_dir.exists() else []

    if parquet_files:
        logger.info("loading_real_parquet", extra={"file": str(parquet_files[0])})
        df = pd.read_parquet(parquet_files[0], columns=RAW_COLUMNS)
        return df

    logger.warning("no_parquet_found_using_synthetic_data")
    return _synthesize()


def compute_duration_minutes(df: pd.DataFrame) -> pd.Series:
    duration = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
    return duration


def train_val_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=settings.val_size, random_state=settings.random_seed)
