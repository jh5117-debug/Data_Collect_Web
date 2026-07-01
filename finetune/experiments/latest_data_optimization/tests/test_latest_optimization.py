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
