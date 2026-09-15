def test_health_returns_200_when_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_metadata_returns_labels(client):
    resp = client.get("/metadata")
    assert resp.status_code == 200
    assert set(resp.json()["labels"]) == {"negative", "neutral", "positive"}


def test_predict_happy_path(client):
    resp = client.post("/predict", json={"text": "المنتج رائع جدا وسريع في التوصيل"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in {"negative", "neutral", "positive"}
    assert resp.headers["X-Request-ID"] == body["correlation_id"]


def test_predict_empty_text_returns_422(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(
        "/predict/batch",
        json={"reviews": [{"text": "رائع جدا"}, {"text": "سيء جدا"}]},
    )
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2


def test_metrics_endpoint_exposes_prometheus_format(client):
    client.post("/predict", json={"text": "المنتج رائع"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"predictions_total" in resp.content
    assert b"prediction_latency_seconds" in resp.content
