from __future__ import annotations

import pytest

from aggregate_clip_predictions import (
    PredictionConsistencyError,
    aggregate_clip_cascade_predictions,
    aggregate_stage_clip_predictions,
)


def row(clip_id: str, window_index: int, label: int, score: float, **extra):
    data = {
        "clip_id": clip_id,
        "window_index": window_index,
        "label": label,
        "split": "test",
        "speaker_id": "spk_a",
        "session_id": "S1",
        "prompt_group": "P1_vigil_only" if label else "P4_negative",
        "phrase_id": "vigil" if label else "visual",
        "transcript": "VIGIL" if label else "visual",
        "score": score,
        "window_start_sec": float(window_index),
        "window_end_sec": float(window_index + 2),
    }
    data.update(extra)
    return data


def s2(clip_id: str, window_index: int, label: int, score: float):
    data = row(clip_id, window_index, label, 0.0)
    data.pop("score")
    data["stage2_score"] = score
    return data


def test_stage1_clip_aggregation_uses_max_score():
    rows = [row("C1", 0, 1, 0.2), row("C1", 1, 1, 0.9), row("C2", 0, 0, 0.1)]
    agg = aggregate_stage_clip_predictions(rows, 0.5)
    by_clip = {item["clip_id"]: item for item in agg}
    assert by_clip["C1"]["score"] == 0.9
    assert by_clip["C1"]["candidate_window_index"] == 1
    assert by_clip["C1"]["stage1_candidate"] is True
    assert by_clip["C2"]["stage1_candidate"] is False


def test_clip_label_inconsistency_is_rejected():
    rows = [row("C1", 0, 1, 0.2), row("C1", 1, 0, 0.9)]
    with pytest.raises(PredictionConsistencyError):
        aggregate_stage_clip_predictions(rows, 0.5)


def test_cascade_requires_both_thresholds_on_same_window():
    stage1 = [row("C1", 0, 1, 0.9), row("C1", 1, 1, 0.8)]
    stage2 = [s2("C1", 0, 1, 0.1), s2("C1", 1, 1, 0.7)]
    agg = aggregate_clip_cascade_predictions(stage1, stage2, 0.5, 0.6, top_k=1)
    assert agg[0]["final_trigger"] is False
    agg = aggregate_clip_cascade_predictions(stage1, stage2, 0.5, 0.6, top_k=2)
    assert agg[0]["final_trigger"] is True
    assert agg[0]["winning_candidate"]["window_index"] == 1


def test_no_stage2_acceptance_when_stage1_has_no_candidate():
    stage1 = [row("C1", 0, 1, 0.4)]
    stage2 = [s2("C1", 0, 1, 0.99)]
    agg = aggregate_clip_cascade_predictions(stage1, stage2, 0.5, 0.6)
    assert agg[0]["stage1_candidate"] is False
    assert agg[0]["evaluated_candidate_count"] == 0
    assert agg[0]["final_trigger"] is False


def test_candidate_window_timestamps_are_preserved():
    stage1 = [row("C1", 3, 1, 0.9, window_start_sec=0.75, window_end_sec=2.75)]
    stage2 = [s2("C1", 3, 1, 0.99)]
    agg = aggregate_clip_cascade_predictions(stage1, stage2, 0.5, 0.6)
    candidate = agg[0]["candidate_windows"][0]
    assert candidate["window_start_sec"] == 0.75
    assert candidate["window_end_sec"] == 2.75

