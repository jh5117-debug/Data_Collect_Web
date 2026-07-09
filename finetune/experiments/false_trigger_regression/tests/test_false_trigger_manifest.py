from __future__ import annotations

from report import HARD_NEGATIVE_PHRASES, hard_negative_plan_markdown
from rosbag_index import expected_label_for_bag_name


def test_false_positive_expected_label_assignment() -> None:
    assert expected_label_for_bag_name("false_positive") == 0


def test_true_positive_expected_label_assignment() -> None:
    assert expected_label_for_bag_name("true_positive") == 1


def test_hard_negative_phrase_list_includes_observed_words() -> None:
    for phrase in ("go", "joe", "joke", "yo"):
        assert phrase in HARD_NEGATIVE_PHRASES
        assert phrase in hard_negative_plan_markdown()

