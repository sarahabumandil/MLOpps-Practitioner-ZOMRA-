from __future__ import annotations

import pandas as pd

from prodml import data, train
from prodml.config import settings


def test_load_raw_falls_back_to_synthetic_when_no_parquet(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)  # empty dir, no data/raw/*.parquet
    df = data.load_raw()
    assert set(data.RAW_COLUMNS).issubset(df.columns)
    assert len(df) > 0


def test_compute_duration_minutes():
    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": [pd.Timestamp("2023-01-01 00:00:00")],
            "lpep_dropoff_datetime": [pd.Timestamp("2023-01-01 00:12:00")],
        }
    )
    assert data.compute_duration_minutes(df).iloc[0] == 12.0


def test_train_end_to_end_on_a_small_mocked_dataset(monkeypatch, tmp_path):
    # Mocked: skip the real 20k-row synthesis and a 200-tree forest so the
    # unit suite stays fast; the full-scale path is exercised by
    # `python -m prodml.train` directly (see reports/module-1.md).
    small_df = data._synthesize(n_rows=300, seed=1)
    monkeypatch.setattr(train, "load_raw", lambda: small_df)
    monkeypatch.setattr(settings, "n_estimators", 5)
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr(settings, "model_path", tmp_path / "model.pkl")

    metrics = train.train()

    assert "mae" in metrics and metrics["mae"] >= 0
    assert (tmp_path / "model.pkl").exists()
