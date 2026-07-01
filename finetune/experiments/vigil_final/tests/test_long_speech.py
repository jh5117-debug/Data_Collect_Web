from __future__ import annotations

from vigil_final.long_speech import false_activations_per_hour, summarize_stress, window_count


def test_false_activations_per_hour():
    assert false_activations_per_hour(2, 3600.0) == 2.0


def test_window_count_and_summary():
    assert window_count(2.0, 2.0, 0.25) == 1
    assert window_count(2.5, 2.0, 0.25) == 3
    summary = summarize_stress([{"final_trigger": True, "windows": 3, "stage1_candidates": 2}], 3600.0)
    assert summary["final_false_accepts_per_hour"] == 1.0
