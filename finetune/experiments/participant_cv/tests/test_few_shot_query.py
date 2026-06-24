from __future__ import annotations

from vigil_participant_cv.few_shot import eligible_for_shots


def test_zero_shot_and_few_shot_paired_query_sets_are_identical_by_protocol():
    clips = [{"label": 1}] * 6 + [{"label": 0}]
    assert eligible_for_shots(clips, 3)[0]
    assert eligible_for_shots(clips, 5)[0]


def test_few_shot_eligibility_requires_negative_query():
    assert not eligible_for_shots([{"label": 1}] * 6, 5)[0]
