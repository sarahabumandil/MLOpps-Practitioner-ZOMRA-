import pytest


def test_predict_one_returns_valid_label(trained_model, sample_text):
    result = trained_model.predict_one(sample_text)
    assert result["label"] in {"negative", "neutral", "positive"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_deterministic_across_two_calls(trained_model, sample_text):
    first = trained_model.predict_one(sample_text)
    second = trained_model.predict_one(sample_text)
    assert first == second


def test_predict_batch_matches_predict_one(trained_model, sample_text):
    batch_result = trained_model.predict_batch([sample_text, sample_text])
    one_result = trained_model.predict_one(sample_text)
    assert batch_result[0] == one_result


def test_positive_review_predicted_positive(trained_model):
    result = trained_model.predict_one("المنتج رائع جدا وأنصح بالشراء")
    assert result["label"] == "positive"


def test_negative_review_predicted_negative(trained_model):
    result = trained_model.predict_one("المنتج سيء جدا ولا يستحق السعر")
    assert result["label"] == "negative"


def test_save_and_load_roundtrip(trained_model, tmp_path):
    from arasent.model import SentimentClassifier

    path = tmp_path / "model.joblib"
    trained_model.save(path)
    reloaded = SentimentClassifier.load(path)
    assert reloaded.predict_one("المنتج رائع جدا") == trained_model.predict_one("المنتج رائع جدا")
