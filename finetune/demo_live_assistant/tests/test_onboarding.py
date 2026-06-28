from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from server import create_app


def client_and_profile(tmp_path: Path) -> tuple[TestClient, str]:
    client = TestClient(create_app(data_root=tmp_path / "local_data", force_mock=True, load_models=True))
    profile = client.post("/api/profile", json={"name": "Demo User"}).json()
    return client, profile["profile_id"]


def upload(client: TestClient, profile_id: str, *, positive: bool = True, text: bytes = b"VIGIL") -> dict:
    response = client.post(
        "/api/onboarding/clip",
        data={
            "profile_id": profile_id,
            "prompt_group": "P1_vigil_only" if positive else "P4_negative",
            "transcript": "VIGIL" if positive else "visual",
            "is_positive": str(positive).lower(),
            "accepted": "true",
        },
        files={"file": ("clip.webm", text, "audio/webm")},
    )
    assert response.status_code == 200
    return response.json()


def test_onboarding_clip_saved_played_and_deleted(tmp_path: Path) -> None:
    client, profile_id = client_and_profile(tmp_path)
    clip = upload(client, profile_id)
    audio = client.get(f"/api/onboarding/clip/{clip['clip_id']}/audio", params={"profile_id": profile_id})
    assert audio.status_code == 200
    deleted = client.delete(f"/api/onboarding/clip/{clip['clip_id']}", params={"profile_id": profile_id})
    assert deleted.status_code == 200
    missing = client.get(f"/api/onboarding/clip/{clip['clip_id']}/audio", params={"profile_id": profile_id})
    assert missing.status_code == 404


def test_path_traversal_clip_id_rejected(tmp_path: Path) -> None:
    client, profile_id = client_and_profile(tmp_path)
    response = client.get("/api/onboarding/clip/..%2Fsecret/audio", params={"profile_id": profile_id})
    assert response.status_code in {400, 404}


def test_calibration_requires_three_positive_clips_and_ignores_negative(tmp_path: Path) -> None:
    client, profile_id = client_and_profile(tmp_path)
    upload(client, profile_id, positive=True)
    upload(client, profile_id, positive=False, text=b"visual")
    early = client.post("/api/onboarding/calibrate", json={"profile_id": profile_id}).json()
    assert early["calibration_status"] == "need_more_positive_clips"
    assert early["support_count"] == 1
    upload(client, profile_id, positive=True)
    upload(client, profile_id, positive=True)
    ok = client.post("/api/onboarding/calibrate", json={"profile_id": profile_id}).json()
    assert ok["calibration_status"] == "ok"
    assert ok["support_count"] == 3
    assert ok["method"] == "bounded_positive_bias_demo"
