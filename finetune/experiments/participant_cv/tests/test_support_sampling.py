from __future__ import annotations

from vigil_participant_cv.support_sampling import select_positive_support, support_query_split


def _clips():
    return [
        {"participant_alias": "P001", "clip_id": "P1", "label": 1, "prompt_group": "P1_vigil_only", "full_wav_sha256": "h1"},
        {"participant_alias": "P001", "clip_id": "P2", "label": 1, "prompt_group": "P2_phrase_plus_vigil", "full_wav_sha256": "h2"},
        {"participant_alias": "P001", "clip_id": "P3", "label": 1, "prompt_group": "P3_vigil_plus_phrase", "full_wav_sha256": "h3"},
        {"participant_alias": "P001", "clip_id": "P4", "label": 1, "prompt_group": "P2_phrase_plus_vigil", "full_wav_sha256": "h4"},
        {"participant_alias": "P001", "clip_id": "P5", "label": 1, "prompt_group": "P3_vigil_plus_phrase", "full_wav_sha256": "h5"},
        {"participant_alias": "P001", "clip_id": "N1", "label": 0, "prompt_group": "P4_negative", "full_wav_sha256": "h6"},
    ]


def test_3_shot_support_contains_exactly_three_positives():
    support = select_positive_support(_clips(), k=3, seed=1)
    assert len(support) == 3
    assert all(c["label"] == 1 for c in support)


def test_5_shot_support_contains_exactly_five_positives():
    assert len(select_positive_support(_clips(), k=5, seed=1)) == 5


def test_3_shot_is_nested_inside_5_shot_when_possible():
    three = {c["clip_id"] for c in select_positive_support(_clips(), k=3, seed=1)}
    five = {c["clip_id"] for c in select_positive_support(_clips(), k=5, seed=1)}
    assert three <= five


def test_support_and_query_are_disjoint():
    support = select_positive_support(_clips(), k=3, seed=1)
    support, query = support_query_split(_clips(), support)
    assert {c["clip_id"] for c in support}.isdisjoint({c["clip_id"] for c in query})
