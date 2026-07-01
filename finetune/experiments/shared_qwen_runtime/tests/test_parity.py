from __future__ import annotations

from parity import compare_transcripts, cost_table_row, score_difference, score_parity_status
from shared_qwen_adapter import BLOCKER, SharedAttempt


def test_transcript_parity_utility_normalizes_punctuation() -> None:
    parity = compare_transcripts("Hello, VIGIL!", "hello vigil")
    assert parity is not None
    assert not parity.exact_match
    assert parity.normalized_match
    assert parity.word_edit_distance == 0


def test_score_parity_tolerance_logic() -> None:
    assert score_difference(0.5, 0.50005) < 1e-4
    assert score_parity_status(score_difference(0.5, None)) == "blocked"
    assert score_parity_status(score_difference(0.5, 0.50005)) == "passed"
    assert score_parity_status(score_difference(0.5, 0.51)) == "failed"


def test_cost_table_generation() -> None:
    row = cost_table_row("current", 1, "2", "yes", "yes", 13.663, "working")
    assert row["qwen_weight_copies"] == 1
    assert row["median_latency_ms"] == 13.663


def test_shared_adapter_blocked_result_is_explicit() -> None:
    result = SharedAttempt(
        status="blocked_by_runtime_interface",
        transcript="VIGIL",
        raw_transcribe_result_type="qwen_asr.inference.qwen3_asr.ASRTranscription",
        audio_features_available=False,
        feature_shape=None,
        feature_dtype=None,
        encoder_call_count=1,
        decoder_call_count=1,
        feature_path=None,
        transcript_path="$[0].text",
        latency_ms=1.0,
        blocker=BLOCKER,
    )
    data = result.as_dict()
    assert data["status"] == "blocked_by_runtime_interface"
    assert data["audio_features_available"] is False
    assert "does not expose" in data["blocker"]
