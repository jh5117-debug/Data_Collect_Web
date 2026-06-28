from __future__ import annotations

import pytest

from model_runtime import AssistantModelRuntime
from transcript import safe_extract_transcript


class FakeASRResult:
    def __init__(self, text: str) -> None:
        self.text = text


def test_transcript_extraction_rejects_object_repr() -> None:
    assert safe_extract_transcript([FakeASRResult("VIGIL")]) == "VIGIL"
    with pytest.raises(ValueError):
        safe_extract_transcript("ASRTranscription(language='English', text='VIGIL')")


class FakeQwenModel:
    def transcribe(self, path: str, language: object = None) -> list[FakeASRResult]:
        assert language is None
        assert path
        return [FakeASRResult("ordinary background speech")]


class FakeQwenRuntime:
    wrapper = FakeQwenModel()
    model = None


class FakeInference:
    runtime = type("Runtime", (), {"qwen": FakeQwenRuntime()})()

    def analyze(self, path: object, *, run_transcript_after_trigger: bool) -> dict[str, object]:
        assert path
        assert run_transcript_after_trigger is False
        return {
            "variant": "stage2_bce_supcon",
            "theta_1": 0.9,
            "theta_2": 0.8,
            "stage1_score": 0.1,
            "stage2_score": None,
            "window_table": [{"Candidate": False, "Stage 2": None}],
            "winning_window": None,
        }


class FailingTriggerInference(FakeInference):
    def analyze(self, path: object, *, run_transcript_after_trigger: bool) -> dict[str, object]:
        assert run_transcript_after_trigger is False
        raise RuntimeError("ffmpeg conversion failed")


def test_live_runtime_transcribes_each_chunk_without_trigger(tmp_path) -> None:
    audio_path = tmp_path / "chunk.webm"
    audio_path.write_bytes(b"not a wake word")
    runtime = AssistantModelRuntime(force_mock=True)
    runtime.mode = "real"
    runtime.inference = FakeInference()

    result = runtime.analyze_audio(audio_path)

    assert result["trigger_detected"] is False
    assert result["rolling_transcript"] == "ordinary background speech"
    assert result["debug"]["qwen_transcript_extraction_path"] == "$[0].text"
    assert result["debug"]["qwen_weight_instances"] == 1
    assert result["debug"]["stage2_qwen_feature_path_used"] is False


def test_live_runtime_keeps_transcript_when_trigger_path_fails(tmp_path) -> None:
    audio_path = tmp_path / "chunk.webm"
    audio_path.write_bytes(b"bad but transcribable in fake model")
    runtime = AssistantModelRuntime(force_mock=True)
    runtime.mode = "real"
    runtime.inference = FailingTriggerInference()

    result = runtime.analyze_audio(audio_path)

    assert result["rolling_transcript"] == "ordinary background speech"
    assert result["trigger_detected"] is False
    assert result["stage1_score"] == 0.0
    assert "ffmpeg conversion failed" in result["debug"]["trigger_path_error"]
