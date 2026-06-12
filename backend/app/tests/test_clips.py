import io
import math
import shutil
import wave

import pytest


def _create_participant_and_session(client):
    participant = client.post(
        "/api/participants",
        json={
            "english_native_speaker": "prefer_not_to_say",
            "recording_device_type": "webcam_microphone",
        },
    ).json()
    session = client.post(
        "/api/sessions",
        json={"participant_id": participant["participant_id"], "batch_id": "vigil_batch_v0_1"},
    ).json()
    return participant["participant_id"], session["session_id"]


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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required for conversion success")
def test_clip_upload_with_valid_audio(client):
    participant_id, session_id = _create_participant_and_session(client)

    response = client.post(
        "/api/clips",
        data={
            "participant_id": participant_id,
            "session_id": session_id,
            "prompt_id": "POS_SINGLE_001",
            "clip_type": "normal",
        },
        files={"audio": ("clip.wav", _wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "uploaded"
    assert payload["clip_id"] == "C000001"
    assert payload["auto_qc_status"] == "auto_accepted"
    assert payload["segmentation_status"] == "not_required"


def test_ffmpeg_conversion_failure_is_recorded(client):
    participant_id, session_id = _create_participant_and_session(client)

    response = client.post(
        "/api/clips",
        data={
            "participant_id": participant_id,
            "session_id": session_id,
            "prompt_id": "POS_SINGLE_001",
            "clip_type": "normal",
        },
        files={"audio": ("not-audio.webm", b"not actually audio", "audio/webm")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["auto_qc_status"] == "auto_rejected"
    assert "ffmpeg_conversion_failed" in payload["auto_qc_flags"]
