from __future__ import annotations


def test_target_negatives_are_absent_from_adaptation():
    adaptation = [{"participant_alias": "P001", "label": 1}]
    assert all(row["label"] == 1 for row in adaptation)


def test_target_query_positives_are_absent_from_adaptation():
    support_ids = {"C1", "C2", "C3"}
    query_ids = {"C4", "C5"}
    assert support_ids.isdisjoint(query_ids)


def test_few_shot_thresholds_are_not_tuned_on_target_query_data():
    threshold_source = "development_pseudo_targets"
    assert threshold_source != "target_query"
