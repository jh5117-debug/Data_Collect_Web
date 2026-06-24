from __future__ import annotations

import pytest

from finetune.scripts.run_qwen_text_baseline import group_rows_for_evaluation


def base_row(clip_id: str, window_index: int, label: int = 1):
    return {
        "clip_id": clip_id,
        "window_index": window_index,
        "label": label,
        "split": "test",
        "full_wav_path": f"/tmp/{clip_id}.wav",
        "prompt_group": "P1_vigil_only",
        "phrase_id": "vigil",
    }


def test_clip_evaluation_deduplicates_repeated_windows():
    rows = [base_row("C1", 0), base_row("C1", 1), base_row("C2", 0)]
    out = group_rows_for_evaluation(rows, evaluation_unit="clip", deduplicate_by="clip_id")
    assert [row["clip_id"] for row in out] == ["C1", "C2"]
    assert out[0]["deduplicated_rows"] == 2


def test_window_evaluation_keeps_repeated_windows_for_legacy_mode():
    rows = [base_row("C1", 0), base_row("C1", 1)]
    out = group_rows_for_evaluation(rows, evaluation_unit="window", deduplicate_by="clip_id")
    assert len(out) == 2


def test_clip_evaluation_rejects_inconsistent_labels():
    rows = [base_row("C1", 0, label=1), base_row("C1", 1, label=0)]
    with pytest.raises(ValueError):
        group_rows_for_evaluation(rows, evaluation_unit="clip", deduplicate_by="clip_id")


def test_clip_evaluation_rejects_inconsistent_audio_path():
    rows = [base_row("C1", 0), base_row("C1", 1)]
    rows[1]["full_wav_path"] = "/tmp/other.wav"
    with pytest.raises(ValueError):
        group_rows_for_evaluation(rows, evaluation_unit="clip", deduplicate_by="clip_id")

