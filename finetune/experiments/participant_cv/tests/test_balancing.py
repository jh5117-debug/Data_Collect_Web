from __future__ import annotations

from collections import Counter

from vigil_participant_cv.balancing import balance_max_clips_per_participant
from vigil_participant_cv.utils import stable_json_dumps


def _clip(i: int, prompt: str, phrase: str = "vigil") -> dict:
    return {"participant_alias": "P001", "clip_id": f"C{i:04d}", "label": 0 if prompt == "P4_negative" else 1, "prompt_group": prompt, "phrase_id": phrase}


def test_clip_level_cap_not_window_level_cap():
    clips = [_clip(i, "P1_vigil_only") for i in range(120)]
    selected, _ = balance_max_clips_per_participant(clips, max_clips=100)
    assert len(selected) == 100


def test_no_participant_exceeds_100_selected_clips():
    clips = [_clip(i, "P2_phrase_plus_vigil") for i in range(150)]
    selected, _ = balance_max_clips_per_participant(clips, max_clips=100)
    assert Counter(c["participant_alias"] for c in selected)["P001"] <= 100


def test_cap_preserves_prompt_group_diversity():
    clips = [_clip(i, "P1_vigil_only") for i in range(50)] + [_clip(100 + i, "P4_negative", "video") for i in range(90)]
    selected, _ = balance_max_clips_per_participant(clips, max_clips=100)
    prompts = {c["prompt_group"] for c in selected}
    assert {"P1_vigil_only", "P4_negative"} <= prompts


def test_cap_preserves_hard_negative_diversity():
    clips = [_clip(i, "P4_negative", "video") for i in range(70)] + [_clip(100 + i, "P4_negative", "visual") for i in range(70)]
    selected, _ = balance_max_clips_per_participant(clips, max_clips=100)
    phrases = {c["phrase_id"] for c in selected}
    assert {"video", "visual"} <= phrases


def test_cap_is_byte_deterministic():
    clips = [_clip(i, "P2_phrase_plus_vigil") for i in range(120)]
    a, _ = balance_max_clips_per_participant(clips, max_clips=100)
    b, _ = balance_max_clips_per_participant(clips, max_clips=100)
    assert stable_json_dumps(a) == stable_json_dumps(b)
