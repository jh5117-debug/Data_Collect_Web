import io
import math
import shutil
import wave
from pathlib import Path
from zipfile import ZipFile

import pytest

from app.services.storage import get_storage_backend


def _wav_bytes(duration_sec: float = 0.8, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    frame_count = int(duration_sec * sample_rate)
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            value = int(0.25 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate))
            frames.extend(value.to_bytes(2, byteorder="little", signed=True))
        handle.writeframes(bytes(frames))
    return buffer.getvalue()


def _login(client, email: str = "speaker@example.com") -> str:
    code_response = client.post("/api/auth/request-code", json={"email": email})
    code = code_response.json()["dev_code"]
    verify_response = client.post("/api/auth/verify-code", json={"email": email, "code": code})
    return verify_response.json()["auth_token"]


def _participant_and_session(client, email: str = "speaker@example.com") -> tuple[str, str, str]:
    token = _login(client, email)
    participant = client.post(
        "/api/participants",
        json={
            "user_email": email,
            "english_native_speaker": "prefer_not_to_say",
            "recording_device_type": "webcam_microphone",
        },
    ).json()
    session = client.post(
        "/api/sessions",
        json={"participant_id": participant["participant_id"], "batch_id": "vigil_batch_v0_1"},
    ).json()
    return participant["participant_id"], session["session_id"], token


def _upload(client, participant_id: str, session_id: str, prompt_group: str, transcript: str | None = None):
    data = {
        "participant_id": participant_id,
        "session_id": session_id,
        "prompt_group": prompt_group,
        "clip_type": "normal",
    }
    if transcript is not None:
        data["transcript"] = transcript
    return client.post(
        "/api/clips",
        data=data,
        files={"audio": ("clip.wav", _wav_bytes(), "audio/wav")},
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required for conversion success")
def test_prompt_group_upload_validation_and_labels(client):
    participant_id, session_id, _token = _participant_and_session(client)

    p1 = _upload(client, participant_id, session_id, "P1_vigil_only", "ignored")
    assert p1.status_code == 200
    assert p1.json()["transcript"] == "Vigil"
    assert p1.json()["prompt_group"] == "P1_vigil_only"
    assert p1.json()["contains_vigil"] is True
    assert p1.json()["wake_intent"] is True
    assert p1.json()["is_negative"] is False

    p2_missing = _upload(client, participant_id, session_id, "P2_phrase_plus_vigil", "Hi there")
    assert p2_missing.status_code == 400
    assert p2_missing.json()["detail"] == "This prompt should include the word 'Vigil'."

    p3_missing = _upload(client, participant_id, session_id, "P3_vigil_plus_phrase", "go back")
    assert p3_missing.status_code == 400
    assert p3_missing.json()["detail"] == "This prompt should include the word 'Vigil'."

    p2 = _upload(client, participant_id, session_id, "P2_phrase_plus_vigil", "Hi VIGIL.")
    assert p2.status_code == 200
    assert p2.json()["normalized_transcript"] == "Hi Vigil."

    p3 = _upload(client, participant_id, session_id, "P3_vigil_plus_phrase", "VIGIL, go back.")
    assert p3.status_code == 200
    assert p3.json()["normalized_transcript"] == "Vigil, go back."

    p4_reject = _upload(client, participant_id, session_id, "P4_negative", "Vigil next")
    assert p4_reject.status_code == 400
    assert p4_reject.json()["detail"].startswith("Negative examples should not contain")

    for word in ["visual", "digital", "individual", "visible", "vigilant"]:
        response = _upload(client, participant_id, session_id, "P4_negative", word)
        assert response.status_code == 200
        assert response.json()["prompt_group"] == "P4_negative"
        assert response.json()["is_negative"] is True
        assert response.json()["contains_vigil"] is False


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required for conversion success")
def test_summary_detail_export_and_deletion_for_prompt_groups(client):
    email = "owner@example.com"
    participant_id, session_id, token = _participant_and_session(client, email)
    _upload(client, participant_id, session_id, "P1_vigil_only")
    _upload(client, participant_id, session_id, "P2_phrase_plus_vigil", "Hi VIGIL.")
    _upload(client, participant_id, session_id, "P4_negative", "visual")

    summary = client.get("/api/admin/summary").json()
    assert summary["positive_clips"] == 2
    assert summary["negative_clips"] == 1
    assert summary["prompt_group_counts"]["P4_negative"] == 1

    account_clips = client.get(
        f"/api/auth/accounts/{email}/sessions/{session_id}/clips",
        headers={"X-Auth-Token": token},
    ).json()
    assert account_clips[0]["prompt_group"] == "P1_vigil_only"
    assert account_clips[0]["transcript"] == "Vigil"

    admin_clips = client.get(f"/api/admin/sessions/{session_id}/clips").json()
    assert {clip["prompt_group"] for clip in admin_clips} == {
        "P1_vigil_only",
        "P2_phrase_plus_vigil",
        "P4_negative",
    }
    assert any(clip["transcript"] == "visual" for clip in admin_clips)

    export_response = client.post("/api/admin/export").json()
    export_path = get_storage_backend().absolute_path(f"exports/{export_response['file_name']}")
    with ZipFile(export_path) as archive:
        names = archive.namelist()
        assert any("/by_prompt_group/P4_negative/processed_wav/" in name for name in names)
        assert any(name.endswith("/metadata/clips.csv") for name in names)
        qwen_train = "\n".join(
            archive.read(name).decode("utf-8")
            for name in names
            if name.endswith("/qwen_asr/train.jsonl") or name.endswith("/qwen_asr/eval.jsonl")
        )
        kws_train = "\n".join(
            archive.read(name).decode("utf-8")
            for name in names
            if name.endswith("/keyword_spotting/kws_train.jsonl") or name.endswith("/keyword_spotting/kws_eval.jsonl")
        )
        assert "language English<asr_text>visual" in qwen_train
        assert '"is_negative": true' in kws_train
        assert '"prompt_group": "P4_negative"' in kws_train

    first_clip = admin_clips[0]
    raw_path = get_storage_backend().absolute_path(first_clip["raw_audio_path"])
    wav_path = get_storage_backend().absolute_path(first_clip["processed_wav_path"])
    assert raw_path.exists()
    assert wav_path.exists()
    delete_clip = client.delete(f"/api/admin/clips/{first_clip['clip_id']}")
    assert delete_clip.status_code == 200
    assert not raw_path.exists()
    assert not wav_path.exists()

    remaining_before_session_delete = client.get(f"/api/admin/sessions/{session_id}/clips").json()
    assert remaining_before_session_delete
    delete_session = client.delete(f"/api/admin/sessions/{session_id}")
    assert delete_session.status_code == 200
    assert client.get(f"/api/admin/sessions/{session_id}/clips").status_code == 404

    participant_id, session_id, _token = _participant_and_session(client, "delete-me@example.com")
    _upload(client, participant_id, session_id, "P4_negative", "digital")
    export_response = client.post("/api/admin/export").json()
    old_export_path = get_storage_backend().absolute_path(f"exports/{export_response['file_name']}")
    assert old_export_path.exists()
    delete_account = client.delete("/api/admin/clients/delete-me@example.com")
    assert delete_account.status_code == 200
    assert not old_export_path.exists()
