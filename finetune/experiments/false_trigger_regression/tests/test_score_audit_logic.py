from __future__ import annotations

from report import inspection_markdown
from score_audit import detect_constant_stage2_scores, detect_identical_hashes, diagnose_score_rows, final_trigger


def test_final_trigger_requires_both_stages() -> None:
    assert final_trigger(True, True) is True
    assert final_trigger(True, False) is False
    assert final_trigger(False, True) is False


def test_constant_stage2_score_detector() -> None:
    result = detect_constant_stage2_scores([0.988, 0.98801, 0.98799], tolerance=0.0001)
    assert result["constant"] is True


def test_identical_embedding_hash_detector() -> None:
    result = detect_identical_hashes(["abc", "abc"])
    assert result["identical"] is True


def test_diagnose_cascade_decision_bug() -> None:
    result = diagnose_score_rows(
        [
            {
                "case_id": "go",
                "stage1_accept": True,
                "stage2_accept": False,
                "final_trigger": True,
                "stage2_score": 0.1,
                "feature_hash": "a",
                "embedding_hash": "b",
            }
        ]
    )
    assert result["diagnosis"] == "cascade_decision_bug"


def test_diagnose_integration_cache_bug() -> None:
    result = diagnose_score_rows(
        [
            {"case_id": "go", "stage1_accept": True, "stage2_accept": True, "final_trigger": True, "stage2_score": 0.988, "feature_hash": "same", "embedding_hash": "same"},
            {"case_id": "joke", "stage1_accept": True, "stage2_accept": True, "final_trigger": True, "stage2_score": 0.988, "feature_hash": "same", "embedding_hash": "same"},
        ]
    )
    assert result["diagnosis"] == "integration_cache_or_window_bug"


def test_diagnose_heldout_false_positive_without_cache_bug() -> None:
    result = diagnose_score_rows(
        [
            {
                "case_id": "joe",
                "expected_label": 0,
                "stage1_accept": True,
                "stage2_accept": True,
                "final_trigger": True,
                "stage2_score": 0.8479,
                "feature_hash": "feature_a",
                "embedding_hash": "embedding_a",
            },
            {
                "case_id": "vigil",
                "expected_label": 1,
                "stage1_accept": True,
                "stage2_accept": True,
                "final_trigger": True,
                "stage2_score": 0.8467,
                "feature_hash": "feature_b",
                "embedding_hash": "embedding_b",
            },
        ]
    )
    assert result["diagnosis"] == "heldout_false_positive_model_or_threshold_issue"
    assert result["false_accepts"] == ["joe"]


def test_reports_can_be_generated_without_audio_decode() -> None:
    md = inspection_markdown({"zip_path": "x.zip", "zip_size_bytes": 1, "bag_count": 0, "bags": []})
    assert "ROS Bag Inspection Report" in md
