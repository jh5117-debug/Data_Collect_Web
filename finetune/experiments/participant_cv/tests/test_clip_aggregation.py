from __future__ import annotations

from vigil_participant_cv.clip_aggregation import cascade_clip_decision, stage1_clip_score


def test_clip_aggregation_uses_maximum_stage1_score():
    assert stage1_clip_score([{"score": 0.1}, {"score": 0.9}]) == 0.9


def test_cascade_requires_both_thresholds_on_same_candidate_window():
    stage1 = [{"clip_id": "C1", "window_index": 0, "score": 0.9}, {"clip_id": "C1", "window_index": 1, "score": 0.8}]
    result = cascade_clip_decision(stage1, {("C1", 0): 0.1, ("C1", 1): 0.95}, theta_1=0.85, theta_2=0.9, top_k=3)
    assert result["final_trigger"] is False


def test_top_k_candidate_logic():
    stage1 = [{"clip_id": "C1", "window_index": i, "score": 1.0 - i * 0.1} for i in range(5)]
    result = cascade_clip_decision(stage1, {("C1", 4): 1.0}, theta_1=0.0, theta_2=0.5, top_k=3)
    assert len(result["candidate_windows"]) == 3
    assert result["final_trigger"] is False
