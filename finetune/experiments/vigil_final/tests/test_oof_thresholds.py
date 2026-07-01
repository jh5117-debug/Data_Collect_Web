from __future__ import annotations

from vigil_final.calibration import choose_variant, operating_points, threshold_for_recall_target


def test_oof_thresholds_are_deterministic():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.7, 0.6, 0.1]
    first = threshold_for_recall_target(labels, scores, 1.0)
    second = threshold_for_recall_target(labels, scores, 1.0)
    assert first == second
    assert first["threshold"] <= 0.7


def test_stage2_operating_points_cover_targets():
    points = operating_points([1, 1, 0], [0.9, 0.8, 0.1], [0.85, 0.90, 0.95])
    assert [p["recall_target"] for p in points] == [0.85, 0.90, 0.95]


def test_choose_variant_prefers_safe_recall_then_low_fpr():
    selected = choose_variant(
        {
            "stage2_bce": {"recall": 0.91, "false_positive_rate": 0.02, "precision": 0.9, "f1": 0.9},
            "stage2_bce_supcon": {"recall": 0.91, "false_positive_rate": 0.01, "precision": 0.8, "f1": 0.8},
        },
        recall_target=0.9,
    )
    assert selected == "stage2_bce_supcon"
