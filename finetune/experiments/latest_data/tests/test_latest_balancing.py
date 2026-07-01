from __future__ import annotations

from collections import Counter

from vigil_latest.balancing import allocation_by_largest_remainder, balance_max_clips_per_participant


def test_largest_remainder_respects_cap() -> None:
    assert sum(allocation_by_largest_remainder({"a": 80, "b": 40, "c": 10}, 100).values()) == 100


def test_max_100_cap_per_participant() -> None:
    clips = [
        {
            "participant_alias": "P001",
            "clip_id": f"c{i:03d}",
            "prompt_group": "P1_vigil_only" if i < 60 else "P4_negative",
            "phrase_id": "vigil" if i < 60 else "visual",
            "label": 1 if i < 60 else 0,
        }
        for i in range(130)
    ]
    selected, summary = balance_max_clips_per_participant(clips, max_clips=100, seed=1)
    assert len(selected) == 100
    assert summary["participant_summary"][0]["clips_removed"] == 30
    assert Counter(row["prompt_group"] for row in selected)["P4_negative"] > 0
