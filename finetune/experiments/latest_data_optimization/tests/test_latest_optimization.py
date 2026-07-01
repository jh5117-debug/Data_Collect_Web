from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import numpy as np

from vigil_latest_opt.cascade import clip_score_rows
from vigil_latest_opt.long_speech import false_accepts_per_hour, sliding_window_count
from vigil_latest_opt.prototype import build_prototype
from vigil_latest_opt.support import choose_positive_support
from vigil_latest_opt.target_doctor import choose_supports_for_seed, improvement_counts, split_support_query
from vigil_latest_opt.thresholds import threshold_for_recall_target, threshold_for_safe_f1
from vigil_latest_opt.timing import cuda_synchronized_timer


REPORTS = Path("finetune/experiments/latest_data_optimization/reports")


def test_stage2_operating_point_uses_development_only() -> None:
    rows = (REPORTS / "latest_opt_stage2_operating_points.csv").read_text(encoding="utf-8").splitlines()
    assert rows
    assert "selection_split" in rows[0]
    assert all(",val," in row or row.startswith("dev_") for row in rows[1:])


def test_no_test_data_used_for_selected_thresholds() -> None:
    selected = json.loads((REPORTS / "latest_opt_stage2_selected_config.json").read_text(encoding="utf-8"))
    assert len(selected["thresholds"]) == 5
    assert "test" in selected
    assert selected["dev"]["n"] != selected["test"]["n"]


def test_top_k_candidate_selection() -> None:
    stage1 = [
        {"clip_id": "a", "window_index": 0, "score": 0.9, "label": 1},
        {"clip_id": "a", "window_index": 1, "score": 0.8, "label": 1},
        {"clip_id": "a", "window_index": 2, "score": 0.7, "label": 1},
    ]
    stage2 = [
        {"clip_id": "a", "window_index": 0, "stage2_score": 0.1},
        {"clip_id": "a", "window_index": 1, "stage2_score": 0.9},
        {"clip_id": "a", "window_index": 2, "stage2_score": 0.8},
    ]
    top1 = clip_score_rows(stage1, stage2, theta1=0.0, top_k=1)
    top3 = clip_score_rows(stage1, stage2, theta1=0.0, top_k=3)
    assert top1[0]["stage2_candidate_score"] == 0.1
    assert top3[0]["stage2_candidate_score"] == 0.9


def test_stage2_threshold_search_recall_and_safety() -> None:
    rows = [
        {"label": 1, "score": 0.9},
        {"label": 1, "score": 0.8},
        {"label": 0, "score": 0.7},
        {"label": 0, "score": 0.1},
    ]
    recall = threshold_for_recall_target(rows, 1.0)
    safe = threshold_for_safe_f1(rows, preferred_fpr=0.0, allowed_fpr=0.5)
    assert recall["metrics"]["recall"] == 1.0
    assert safe["metrics"]["false_positive_rate"] <= 0.5


def test_support_query_disjoint_and_no_target_negatives() -> None:
    rows = [{"clip_id": f"p{i}", "label": 1} for i in range(4)] + [{"clip_id": "n0", "label": 0}]
    support, query = choose_positive_support(rows, shots=3, seed=20260620)
    assert len(support) == 3
    assert all(row["label"] == 1 for row in support)
    assert not ({row["clip_id"] for row in support} & {row["clip_id"] for row in query})


def test_fewshot_output_changes_when_adaptation_is_real() -> None:
    from finetune.experiments.latest_data_optimization.scripts.run_real_fewshot import apply_recipe

    support = [{"base_score": 0.4, "base_logit": -0.4, "threshold": 0.8, "embedding": np.array([1.0, 0.0], dtype=np.float32)}] * 3
    query = [
        {
            "base_decision": False,
            "stage1_candidate": True,
            "base_score": 0.7,
            "base_logit": 0.0,
            "threshold": 0.8,
            "qwen_exact_decision": False,
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
        }
    ]
    adapted = apply_recipe(query, support, {"method": "support_threshold_calibration", "support_quantile": "median", "margin": 0.0, "max_threshold_drop": 0.2})
    assert adapted[0]["adapted_decision"] != query[0]["base_decision"]


def test_fallback_clearly_marked_when_no_adaptation_selected() -> None:
    selected = json.loads((REPORTS / "latest_opt_selected_fewshot_recipe.json").read_text(encoding="utf-8"))
    if not selected["support_based_selected"]:
        assert all(item["method"] == "no_adaptation_zero_shot_fallback" for item in selected["selected_by_fold"].values())


def test_paired_query_metrics_present() -> None:
    summary = json.loads((REPORTS / "latest_opt_real_few_shot_summary.json").read_text(encoding="utf-8"))
    assert "paired_delta" in summary["conditions"]["3-shot"]
    assert "paired_delta" in summary["conditions"]["5-shot"]


def test_compute_timing_uses_cuda_synchronization() -> None:
    src = inspect.getsource(cuda_synchronized_timer)
    assert "torch.cuda.synchronize" in src


def test_qwen_call_counter_accounting_present() -> None:
    cost = json.loads((REPORTS / "latest_opt_compute_cost.json").read_text(encoding="utf-8"))
    assert cost["qwen_call_accounting"]["system_c_extra_encoder_forward_per_stage1_candidate"] == 1


def test_false_accepts_per_hour_calculation() -> None:
    assert false_accepts_per_hour(2, 3600.0) == 2.0
    assert sliding_window_count(2.0, 2.0, 0.25) == 1
    assert sliding_window_count(2.5, 2.0, 0.25) == 3


def test_model_bundle_excludes_qwen_weights_and_private_data() -> None:
    manifest = json.loads((REPORTS / "latest_opt_final_model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["include_qwen_weights"] is False
    text = json.dumps(manifest).lower()
    for forbidden in ("email", "account_id", "participant_alias_map"):
        assert forbidden not in text
    assert manifest["stage1_checkpoint"]["committed"] is False
    assert manifest["stage2_checkpoint"]["committed"] is False


def test_reports_do_not_contain_raw_email_identity() -> None:
    email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")
    for path in REPORTS.glob("*"):
        if path.suffix in {".md", ".json", ".csv"}:
            assert not email_re.search(path.read_text(encoding="utf-8", errors="ignore")), path


def test_target_doctor_support_query_contains_only_target() -> None:
    rows = []
    for i in range(6):
        rows.append({"clip_id": f"tp{i}", "participant_alias": "P001", "label": 1, "prompt_group": "P1_vigil_only"})
    for i in range(2):
        rows.append({"clip_id": f"tn{i}", "participant_alias": "P001", "label": 0, "prompt_group": "P4_negative"})
    rows.append({"clip_id": "other", "participant_alias": "P002", "label": 1, "prompt_group": "P1_vigil_only"})
    supports = choose_supports_for_seed([row for row in rows if row["participant_alias"] == "P001"], 20260620)
    support, query = split_support_query(rows, supports[3], "P001")
    assert len(support) == 3
    assert all(row["participant_alias"] == "P001" for row in query)
    assert all(row["label"] == 1 for row in support)
    assert not ({row["clip_id"] for row in support} & {row["clip_id"] for row in query})


def test_target_doctor_three_shot_is_subset_of_five_shot() -> None:
    rows = [{"clip_id": f"p{i}", "participant_alias": "P001", "label": 1, "prompt_group": f"P{1 + i % 3}"} for i in range(6)]
    rows.append({"clip_id": "n0", "participant_alias": "P001", "label": 0, "prompt_group": "P4_negative"})
    supports = choose_supports_for_seed(rows, 20260620)
    assert {row["clip_id"] for row in supports[3]} <= {row["clip_id"] for row in supports[5]}


def test_target_doctor_negatives_rejected_from_support() -> None:
    rows = [{"clip_id": f"p{i}", "participant_alias": "P001", "label": 1} for i in range(4)]
    rows.append({"clip_id": "n0", "participant_alias": "P001", "label": 0})
    try:
        split_support_query(rows, [rows[-1]], "P001")
    except ValueError as exc:
        assert "negatives" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("target negatives must be rejected from support")


def test_target_doctor_improvement_counts() -> None:
    rows = [
        {"3-shot_delta_f1": 0.1},
        {"3-shot_delta_f1": -0.1},
        {"3-shot_delta_f1": 0.0},
    ]
    assert improvement_counts(rows, "3-shot") == {"improved": 1, "degraded": 1, "unchanged": 1}


def test_stage1_structure_report_has_parameter_count() -> None:
    report = REPORTS / "stage1_openwakeword_structure.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["head_parameters"]["total"] == 56321
        assert data["stage1_not_qwen"] is True


def test_librispeech_report_uses_corrected_run_only() -> None:
    report = REPORTS / "current_qwen_librispeech_benchmark.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["status"] == "verified"
        assert data["malformed_hypotheses"] == 0
        assert "fixed_text_extraction" in data["run"]
        assert abs(data["combined_normalized_wer"] - 0.02751646508258752) < 1e-12


def test_shared_qwen_cannot_claim_verified_without_one_encoder_call() -> None:
    from vigil_two_stage.shared_qwen_adapter import can_claim_verified_one_encoder_forward

    diagnostic = {
        "status": "blocked_by_runtime_interface",
        "encoder_call_count": 2,
        "blocker": "public wrapper does not expose reusable hidden-state handoff",
    }
    assert not (diagnostic["status"] == "verified_one_encoder_forward" and diagnostic["encoder_call_count"] != 1)
    assert can_claim_verified_one_encoder_forward("verified_one_encoder_forward", 1)
    assert not can_claim_verified_one_encoder_forward("verified_one_encoder_forward", 2)
    assert not can_claim_verified_one_encoder_forward("blocked_by_runtime_interface", 1)
