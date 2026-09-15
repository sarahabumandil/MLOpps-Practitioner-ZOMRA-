import pytest
from fastapi.testclient import TestClient

from arasent.model import SentimentClassifier

_TRAIN_TEXTS = [
    "المنتج رائع جدا وسريع في التوصيل",
    "جودة ممتازة وأنصح بالشراء",
    "المنتج سيء جدا ولا يستحق السعر",
    "التوصيل متأخر جدا والخدمة سيئة",
    "المنتج عادي لا بأس به",
    "لا سيء ولا ممتاز، متوسط الجودة",
] * 10
_TRAIN_LABELS = [2, 2, 0, 0, 1, 1] * 10


@pytest.fixture
def sample_text():
    return "المنتج رائع جدا وسريع في التوصيل"


@pytest.fixture(scope="session")
def trained_model():
    clf = SentimentClassifier().fit(_TRAIN_TEXTS, _TRAIN_LABELS, C=1.0, max_features=500, seed=0)
    return clf


@pytest.fixture
def client(trained_model, monkeypatch):
    from arasent.api import main as api_main

    monkeypatch.setattr(api_main, "get_predictor", lambda: trained_model)
    monkeypatch.setitem(api_main._state, "predictor", trained_model)

    with TestClient(api_main.app) as c:
        yield c
