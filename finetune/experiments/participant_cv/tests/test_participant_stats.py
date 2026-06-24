from __future__ import annotations

from vigil_participant_cv.participant_stats import attach_aliases, dedupe_clips, participant_statistics
from vigil_participant_cv.privacy import build_alias_map


def test_deterministic_privacy_safe_participant_aliases():
    assert build_alias_map(["spk_b", "spk_a"]) == {"spk_a": "P001", "spk_b": "P002"}


def test_participant_stats_counts_clip_and_window_rows():
    rows = [
        {"clip_id": "C1", "speaker_id": "s1", "session_id": "S1", "label": 1, "prompt_group": "P1_vigil_only", "phrase_id": "vigil", "window_index": 0},
        {"clip_id": "C1", "speaker_id": "s1", "session_id": "S1", "label": 1, "prompt_group": "P1_vigil_only", "phrase_id": "vigil", "window_index": 1},
        {"clip_id": "C2", "speaker_id": "s1", "session_id": "S1", "label": 0, "prompt_group": "P4_negative", "phrase_id": "video", "window_index": 0},
    ]
    clips, _ = attach_aliases(dedupe_clips(rows))
    stats = participant_statistics(clips)
    assert stats[0]["total_unique_clips"] == 2
    assert stats[0]["total_windows"] == 3
