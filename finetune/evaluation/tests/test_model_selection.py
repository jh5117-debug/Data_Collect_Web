from __future__ import annotations

from model_selection import select_stage2_variant


def metric(recall, p4_fpr, precision, f1):
    return {
        "recall": recall,
        "P4_false_positive_rate": p4_fpr,
        "precision": precision,
        "f1": f1,
    }


def test_model_selection_uses_validation_not_test():
    metrics = {
        "splits": {
            "val": {
                "stage2_bce": {"cascade_clip": metric(0.95, 0.0, 1.0, 0.97)},
                "stage2_bce_supcon": {"cascade_clip": metric(0.91, 0.2, 0.9, 0.92)},
            },
            "test": {
                "stage2_bce": {"cascade_clip": metric(0.1, 1.0, 0.1, 0.1)},
                "stage2_bce_supcon": {"cascade_clip": metric(1.0, 0.0, 1.0, 1.0)},
            },
        }
    }
    selection = select_stage2_variant(metrics, recall_constraint=0.90)
    assert selection["selected_variant"] == "stage2_bce"
    assert selection["test_metrics_used_for_selection"] is False


def test_model_selection_applies_recall_constraint_first():
    metrics = {
        "splits": {
            "val": {
                "stage2_bce": {"cascade_clip": metric(0.89, 0.0, 1.0, 0.94)},
                "stage2_bce_supcon": {"cascade_clip": metric(0.90, 0.5, 0.5, 0.6)},
            }
        }
    }
    selection = select_stage2_variant(metrics, recall_constraint=0.90)
    assert selection["selected_variant"] == "stage2_bce_supcon"

