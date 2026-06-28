from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from server import create_app


def make_client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(data_root=tmp_path / "local_data", force_mock=True, load_models=True))


def test_profile_creation_returns_uuid_and_display_name(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post("/api/profile", json={"name": "Dr Local Person"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["profile_id"]) == 32
    assert data["display_name"] == "Dr"


def test_health_route_reports_mock_mode(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "mock"
    assert data["model_loaded"]["qwen_asr"] is False
