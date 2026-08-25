from __future__ import annotations

import pickle

import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

from prodml.config import settings


@pytest.fixture
def sample_features() -> dict:
    return {"PU_DO": "43_151", "trip_distance": 3.2}


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory):
    """A tiny, fast-fitting pipeline written straight to models/model.pkl so
    API tests don't depend on a real full training run.
    """
    X = [
        {"PU_DO": "1_2", "trip_distance": 1.0},
        {"PU_DO": "2_3", "trip_distance": 5.0},
        {"PU_DO": "3_1", "trip_distance": 10.0},
        {"PU_DO": "1_2", "trip_distance": 2.5},
    ]
    y = [6.0, 18.0, 32.0, 9.0]

    pipeline = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer()),
            ("model", RandomForestRegressor(n_estimators=10, random_state=0)),
        ]
    )
    pipeline.fit(X, y)

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.model_path, "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline


@pytest.fixture
def client(trained_model, monkeypatch) -> TestClient:
    from prodml.api.main import app

    with TestClient(app) as c:
        yield c
