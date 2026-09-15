from arasent.config import settings


def test_get_predictor_loads_baseline_backend(trained_model, tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    trained_model.save(path)
    monkeypatch.setattr(settings, "baseline_model_path", path)
    monkeypatch.setenv("ARASENT_MODEL_BACKEND", "baseline")

    import arasent.predict as predict_module

    predict_module.get_predictor.cache_clear()
    predictor = predict_module.get_predictor()

    assert predictor.model_version == trained_model.model_version
    predict_module.get_predictor.cache_clear()
