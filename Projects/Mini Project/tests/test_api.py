from __future__ import annotations

from prodml.api import main as api_main


def test_health_returns_unhealthy_when_model_not_loaded(client, monkeypatch):
    # Mocked, not a real training run: force the module-level predictor into
    # an unloaded state and confirm /health reflects it faithfully.
    monkeypatch.setattr(api_main.predictor, "pipeline", None)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "unhealthy", "model_loaded": False}


def test_health_returns_200_when_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
    assert resp.json()["model_loaded"] is True


def test_metadata_returns_feature_names(client):
    resp = client.get("/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert body["feature_names"] == ["PU_DO", "trip_distance"]


def test_predict_happy_path(client):
    resp = client.post(
        "/predict",
        json={"pu_location_id": 43, "do_location_id": 151, "trip_distance": 3.2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert body["correlation_id"]
    assert "X-Request-ID" in resp.headers


def test_predict_invalid_payload_returns_422_with_readable_message(client):
    resp = client.post(
        "/predict",
        json={"pu_location_id": 43, "do_location_id": 151, "trip_distance": -5},
    )
    assert resp.status_code == 422
    assert "detail" in resp.json()


def test_predict_batch_returns_list(client):
    resp = client.post(
        "/predict/batch",
        json={
            "rides": [
                {"pu_location_id": 1, "do_location_id": 2, "trip_distance": 1.0},
                {"pu_location_id": 3, "do_location_id": 1, "trip_distance": 10.0},
            ]
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2


def test_response_schema_matches_contract(client):
    resp = client.post(
        "/predict",
        json={"pu_location_id": 5, "do_location_id": 9, "trip_distance": 2.0},
    )
    body = resp.json()
    assert set(body.keys()) == {"prediction", "model_version", "correlation_id", "latency_ms"}
