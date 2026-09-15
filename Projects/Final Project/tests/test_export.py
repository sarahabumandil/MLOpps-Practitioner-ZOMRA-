from arasent.config import settings
from arasent.export import export_to_onnx, parity_and_latency_check


def test_export_and_parity_check(trained_model, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "onnx_model_path", tmp_path / "model.onnx")

    export_to_onnx(trained_model)
    assert settings.onnx_model_path.exists()

    texts = ["المنتج رائع جدا", "المنتج سيء جدا", "المنتج عادي"] * 5
    result = parity_and_latency_check(trained_model, texts)
    assert result["parity_ok"] is True
