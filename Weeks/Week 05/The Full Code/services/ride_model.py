"""The ride-duration model, as session 4 needs it: trainable, freezable, replayable.

The relationship is session 3's, unchanged — ``distance / AVG_SPEED_KM_PER_MIN +
passengers * PASSENGER_OVERHEAD_MIN``, the same heuristic session 1's
``RideDurationModel`` used, plus Gaussian noise (see session_3/src/train.py).

Two deliberate extensions, both required by the incidents and neither changing
the model's identity:

* ``hour_of_day`` — session_4's own scripts already monitor it
  (data_drift_evidently.py stat-tests it with PSI). It carries the rush-hour
  congestion term, which is what makes incident 06's seasonal shift a real
  regime change rather than added noise.
* ``city`` — NOT a model feature. It is a segment label carried alongside the
  request so drift and error can be split by it (incident 05). The model never
  sees it, which is precisely why the aggregate hides the failure.

    python -m services.ride_model        # fit + freeze into artifacts/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
DATA = PROJECT_ROOT / "data"

#: Identical to session_3/src/train.py — do not drift from these.
AVG_SPEED_KM_PER_MIN = 0.5
PASSENGER_OVERHEAD_MIN = 0.5
DEFAULT_PARAMS: dict[str, Any] = {"n_estimators": 100, "max_depth": 6, "random_state": 42}

FEATURES = ["distance_km", "passengers", "hour_of_day"]
TARGET = "duration_min"

#: Demand by hour of day. This lives HERE, not in the traffic generator, because
#: the training data and the production feed must come from the SAME process. Fit
#: on uniformly-sampled hours and serve rush-hour-weighted traffic and you have
#: manufactured a permanent PSI of 0.44 on day one — an alert that fires forever
#: and teaches everyone to ignore it. "ramadan" is incident 06's regime.
HOUR_WEIGHTS: dict[str, list[int]] = {
    "normal": [1, 1, 1, 1, 1, 2, 4, 7, 9, 6, 4, 4, 5, 5, 4, 5, 7, 9, 8, 6, 4, 3, 2, 1],
    "ramadan": [3, 2, 1, 1, 2, 3, 2, 2, 3, 3, 3, 3, 3, 4, 6, 9, 14, 18, 9, 5, 6, 7, 6, 4],
}

#: Duration multiplier per city. The model is trained on cairo only; alexandria
#: is the region that arrives in incident 05 and is quietly 26% slower per km.
CITY_FACTOR = {"cairo": 1.0, "alexandria": 1.26}

#: Trip-length profile per city, in km. A new market is not the old market with
#: a different name: Alexandria's corniche generates shorter, slower rides. Both
#: halves matter to incident 05 — the shifted INPUT range is what moves per-city
#: PSI, and the slower duration per km is what moves per-city MAE. Neither one
#: moves the global aggregate while the city is a tenth of volume.
CITY_DISTANCE_RANGE = {"cairo": (0.5, 30.0), "alexandria": (0.5, 11.0)}


def congestion(hour: np.ndarray | float) -> np.ndarray | float:
    """Rush-hour multiplier on travel time: peaks at 08:00 and 17:30."""
    hour = np.asarray(hour, dtype=float)
    return (
        1.0
        + 0.45 * np.exp(-(((hour - 8.0) / 1.6) ** 2))
        + 0.55 * np.exp(-(((hour - 17.5) / 1.9) ** 2))
    )


def true_duration(
    distance_km: np.ndarray, passengers: np.ndarray, hour_of_day: np.ndarray, city: str = "cairo"
) -> np.ndarray:
    """The generating process — used for synthetic data AND for late labels."""
    return (
        distance_km / AVG_SPEED_KM_PER_MIN * congestion(hour_of_day) * CITY_FACTOR.get(city, 1.0)
        + passengers * PASSENGER_OVERHEAD_MIN
    )


def generate_data(
    n_samples: int = 20_000,
    noise: float = 1.0,
    seed: int | None = 42,
    city: str = "cairo",
    profile: str = "normal",
) -> pd.DataFrame:
    """Generate a synthetic ride dataset with the session-1/2/3 relationship."""
    rng = np.random.default_rng(seed)
    distance = rng.uniform(0.5, 30.0, size=n_samples)
    passengers = rng.integers(1, 5, size=n_samples)
    weights = np.array(HOUR_WEIGHTS[profile], dtype=float)
    hour = rng.choice(24, size=n_samples, p=weights / weights.sum())
    duration = true_duration(distance, passengers, hour, city) + rng.normal(0.0, noise, n_samples)
    return pd.DataFrame(
        {
            "distance_km": distance,
            "passengers": passengers.astype(float),
            "hour_of_day": hour.astype(float),
            TARGET: duration,
            "city": city,
        }
    )


def fit(df: pd.DataFrame, params: dict[str, Any] | None = None):
    """Fit the RandomForest on `df` and return (model, scaler)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(df[FEATURES].to_numpy())
    model = RandomForestRegressor(**(params or DEFAULT_PARAMS))
    model.fit(scaler.transform(df[FEATURES].to_numpy()), df[TARGET].to_numpy())
    return model, scaler


def bootstrap(seed: int = 42) -> None:
    """Fit and freeze everything the lab needs. Idempotent; safe to re-run.

    Writes four artifacts:
      artifacts/rf_model.pkl       the champion, version v3
      artifacts/scaler.pkl         the CURRENT preprocessing
      artifacts/scaler_stale.pkl   the preprocessing from BEFORE the last retrain
                                   — incident 02 loads this one
      data/reference.parquet       the FROZEN drift baseline. Frozen is the whole
                                   point: compare today to a moving average and
                                   a slow shift becomes the new normal.
    """
    import joblib

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    train = generate_data(n_samples=50_000, seed=seed)
    model, scaler = fit(train)

    # The stale scaler is fitted on last quarter's shorter-trip mix: same code
    # path, different statistics. That is what makes incident 02 silent.
    stale_df = generate_data(n_samples=20_000, seed=seed + 1)
    stale_df["distance_km"] *= 0.72
    from sklearn.preprocessing import StandardScaler

    stale_scaler = StandardScaler().fit(stale_df[FEATURES].to_numpy())

    # The reference carries the model's OWN predictions on it, not just the
    # target. Prediction drift means "the model's output distribution moved",
    # so the baseline has to be model output too: a RandomForest averages the
    # noise out, so its predictions are tighter than the target it was fit on,
    # and comparing the two manufactures a permanent PSI of ~0.12.
    # data_drift_evidently.py assumes this `pred` column exists.
    train["pred"] = model.predict(scaler.transform(train[FEATURES].to_numpy()))

    joblib.dump(model, ARTIFACTS / "rf_model.pkl")
    joblib.dump(scaler, ARTIFACTS / "scaler.pkl")
    joblib.dump(stale_scaler, ARTIFACTS / "scaler_stale.pkl")
    train.to_parquet(DATA / "reference.parquet", index=False)

    print(f"bootstrap: rf_model.pkl + scaler.pkl + scaler_stale.pkl -> {ARTIFACTS}")
    print(f"bootstrap: frozen reference ({len(train)} rows) -> {DATA / 'reference.parquet'}")


if __name__ == "__main__":
    bootstrap()
