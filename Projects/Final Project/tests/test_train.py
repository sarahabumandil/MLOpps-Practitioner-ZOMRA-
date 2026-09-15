import mlflow

from arasent.config import settings


def test_train_and_evaluate_registers_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "baseline_model_path", tmp_path / "model.joblib")
    monkeypatch.setattr(settings, "mlflow_tracking_uri", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(settings, "mlflow_experiment_name", "test-experiment")
    monkeypatch.setattr(settings, "mlflow_registry_model_name", "TestArabicSentiment")

    import arasent.train as train_module

    # Shrink the sweep so the test runs in well under a second.
    monkeypatch.setattr(
        train_module,
        "SWEEP",
        [{"C": 1.0, "max_features": 500}, {"C": 2.0, "max_features": 500}],
    )

    result = train_module.train_and_evaluate()

    assert settings.baseline_model_path.exists()
    assert result["f1_macro"] > 0
    assert result["registered_version"] is not None

    client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    versions = client.search_model_versions(f"name='{settings.mlflow_registry_model_name}'")
    assert len(versions) >= 1
