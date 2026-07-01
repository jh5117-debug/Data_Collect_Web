from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from server import create_app


def test_assistant_routes_trigger_in_mock_mode(tmp_path: Path) -> None:
    client = TestClient(create_app(data_root=tmp_path / "local_data", force_mock=True, load_models=True))
    profile_id = client.post("/api/profile", json={"name": "Demo User"}).json()["profile_id"]
    for _ in range(3):
        client.post(
            "/api/onboarding/clip",
            data={"profile_id": profile_id, "is_positive": "true", "accepted": "true"},
            files={"file": ("clip.webm", b"VIGIL", "audio/webm")},
        )
    calibration = client.post("/api/onboarding/calibrate", json={"profile_id": profile_id}).json()
    assert calibration["calibration_active"] is True
    session = client.post("/api/assistant/start", json={"profile_id": profile_id}).json()
    chunk = client.post(
        "/api/assistant/chunk",
        data={"profile_id": profile_id, "assistant_session_id": session["assistant_session_id"]},
        files={"file": ("chunk.webm", b"VIGIL", "audio/webm")},
    )
    assert chunk.status_code == 200
    data = chunk.json()
    assert data["trigger_detected"] is True
    assert data["assistant_state"] == "ASSISTANT_STATE"
    assert "VIGIL" in data["rolling_transcript"]


def test_assistant_chunk_route_does_not_500_on_runtime_error(tmp_path: Path) -> None:
    app = create_app(data_root=tmp_path / "local_data", force_mock=True, load_models=True)
    client = TestClient(app)
    profile_id = client.post("/api/profile", json={"name": "Demo User"}).json()["profile_id"]
    for _ in range(3):
        client.post(
            "/api/onboarding/clip",
            data={"profile_id": profile_id, "is_positive": "true", "accepted": "true"},
            files={"file": ("clip.webm", b"VIGIL", "audio/webm")},
        )
    client.post("/api/onboarding/calibrate", json={"profile_id": profile_id})
    session = client.post("/api/assistant/start", json={"profile_id": profile_id}).json()

    def fail_analyze(*args, **kwargs):
        raise RuntimeError("unexpected runtime failure")

    app.state.runtime.analyze_audio = fail_analyze
    chunk = client.post(
        "/api/assistant/chunk",
        data={"profile_id": profile_id, "assistant_session_id": session["assistant_session_id"]},
        files={"file": ("chunk.webm", b"bad", "audio/webm")},
    )

    assert chunk.status_code == 200
    data = chunk.json()
    assert data["trigger_detected"] is False
    assert "unexpected runtime failure" in data["debug"]["route_error"]


def test_assistant_start_requires_calibration(tmp_path: Path) -> None:
    client = TestClient(create_app(data_root=tmp_path / "local_data", force_mock=True, load_models=True))
    profile_id = client.post("/api/profile", json={"name": "Demo User"}).json()["profile_id"]
    response = client.post("/api/assistant/start", json={"profile_id": profile_id})
    assert response.status_code == 400
    assert "calibration" in response.json()["detail"]
