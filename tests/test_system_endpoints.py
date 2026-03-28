from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_returns_expected_shape() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "pipeline_running" in payload
    assert "last_error" in payload
    assert "stt_mock" in payload
    assert "translator_mock" in payload
    assert "translation_provider" in payload


def test_devices_endpoint_returns_list_field() -> None:
    with TestClient(app) as client:
        response = client.get("/devices")

    assert response.status_code == 200
    payload = response.json()
    assert "devices" in payload
    assert isinstance(payload["devices"], list)
