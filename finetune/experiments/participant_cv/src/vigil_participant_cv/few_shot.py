from __future__ import annotations

from typing import Any


def eligible_for_shots(clips: list[dict[str, Any]], shots: int) -> tuple[bool, str]:
    positives = sum(int(clip.get("label", 0)) == 1 for clip in clips)
    negatives = sum(int(clip.get("label", 0)) == 0 for clip in clips)
    if shots == 3 and positives < 4:
        return False, "requires_at_least_4_positive_clips"
    if shots == 5 and positives < 6:
        return False, "requires_at_least_6_positive_clips"
    if negatives < 1:
        return False, "requires_at_least_1_negative_query_clip"
    return True, "eligible"
