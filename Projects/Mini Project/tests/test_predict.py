from __future__ import annotations

import pytest

from prodml.predict import DurationPredictor


def test_predict_one_returns_float_in_sane_range(trained_model):
    predictor = DurationPredictor().load()
    pred = predictor.predict_one(1, 2, 1.0)
    assert isinstance(pred, float)
    assert 0 <= pred <= 180


def test_predict_one_is_deterministic_across_two_calls(trained_model):
    predictor = DurationPredictor().load()
    first = predictor.predict_one(2, 3, 5.0)
    second = predictor.predict_one(2, 3, 5.0)
    assert first == second


def test_predict_one_raises_if_not_loaded():
    predictor = DurationPredictor()
    with pytest.raises(RuntimeError):
        predictor.predict_one(1, 2, 1.0)


def test_predict_batch_returns_one_prediction_per_row(trained_model):
    predictor = DurationPredictor().load()
    rows = [
        {"pu_location_id": 1, "do_location_id": 2, "trip_distance": 1.0},
        {"pu_location_id": 3, "do_location_id": 1, "trip_distance": 10.0},
    ]
    preds = predictor.predict_batch(rows)
    assert len(preds) == 2
    assert all(isinstance(p, float) for p in preds)
