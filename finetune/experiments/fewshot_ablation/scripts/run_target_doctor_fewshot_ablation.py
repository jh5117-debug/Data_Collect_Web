#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from vigil_fewshot_ablation.core import (
    METHOD_ORDER,
    SUPPORT_SEEDS,
    FewShotRecord,
    aggregate_per_doctor,
    aggregate_per_method,
    aggregate_per_seed,
    apply_recipe,
    build_records,
    evaluate_records,
    load_base_by_alias,
    method_changed_stage,
    read_json,
    select_recipe,
    stable_hash,
    summarize_prediction_rows,
    write_csv,
    write_json,
)


ROOT = Path("finetune/experiments/fewshot_ablation")
REPORTS = ROOT / "reports"
DEFAULT_BASE_RUN = Path("finetune/experiments/latest_data_optimization/runs/target_doctor_fewshot")


METHODS = [
    "zero_shot",
    "stage2_cosine_prototype",
    "stage2_positive_bias",
    "stage2_finetune_bias_only",
    "stage2_finetune_head",
    "stage1_finetune_bias_only",
    "stage1_finetune_head",
]


def _records_by_target_shot(records: list[FewShotRecord]) -> dict[tuple[str, int], list[FewShotRecord]]:
    grouped: dict[tuple[str, int], list[FewShotRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.target, record.shot)].append(record)
    return grouped


def _apply_selected_by_target(records: list[FewShotRecord], method: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped = _records_by_target_shot(records)
    rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    selected_by_target: dict[str, Any] = {}
    for (target, shot), target_records in sorted(grouped.items()):
        doctor = target_records[0].doctor_alias
        selected_by_target.setdefault(doctor, {})
        if method == "zero_shot":
            recipe = {"method": "zero_shot", "changed_stage": "none", "support_based": False, "selection_safe": True}
            search_rows.append(
                {
                    "doctor_alias": doctor,
                    "shot": shot,
                    "method": method,
                    "selected": True,
                    "selection_source": "no_support_used",
                    "recipe": json.dumps(recipe, sort_keys=True),
                }
            )
        else:
            dev_records = [record for record in records if record.target != target and record.shot == shot]
            recipe, search = select_recipe(dev_records, method)
            for item in search:
                search_rows.append(
                    {
                        **item,
                        "doctor_alias": doctor,
                        "shot": shot,
                        "method_family": method,
                        "selected": json.dumps({k: v for k, v in item.items() if k in recipe and recipe[k] == v}, sort_keys=True)
                        == json.dumps({k: v for k, v in recipe.items() if k in item and item[k] == v}, sort_keys=True),
                        "recipe": json.dumps({k: v for k, v in item.items() if k in recipe}, sort_keys=True),
                    }
                )
        selected_by_target[doctor][str(shot)] = recipe
        rows.extend(evaluate_records(target_records, recipe))
    return rows, search_rows, selected_by_target


def _combine_stage1_stage2(
    records: list[FewShotRecord],
    stage1_selected: dict[str, Any],
    stage2_selected: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = _records_by_target_shot(records)
    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    for (target, shot), target_records in sorted(grouped.items()):
        doctor = target_records[0].doctor_alias
        s1_recipe = stage1_selected.get(doctor, {}).get(str(shot))
        s2_recipe = stage2_selected.get(doctor, {}).get(str(shot))
        if not s1_recipe or not s2_recipe:
            continue
        selected.setdefault(doctor, {})[str(shot)] = {
            "method": "stage1_stage2_combined",
            "stage1_recipe": s1_recipe,
            "stage2_recipe": s2_recipe,
            "changed_stage": "stage1+stage2",
        }
        for record in target_records:
            s1_rows = apply_recipe(record, s1_recipe)
            s2_rows = apply_recipe(record, s2_recipe)
            by_hash = {row["clip_hash"]: row for row in s2_rows}
            for s1 in s1_rows:
                s2 = by_hash[s1["clip_hash"]]
                stage1_candidate = bool(s1["adapted_decision"] or (s1["zero_decision"] and s1["stage2_reject"]))
                decision = bool(stage1_candidate and s2["adapted_decision"])
                row = {
                    **s2,
                    "method": "stage1_stage2_combined",
                    "changed_stage": "stage1+stage2",
                    "support_based": True,
                    "adapted_decision": decision,
                    "final_false_accept": bool(int(s2["label"]) == 0 and decision),
                }
                rows.append(row)
    return rows, selected


def _write_report(summary: dict[str, Any], per_method: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    lines = [
        "# Few-Shot Ablation Final Report",
        "",
        "This target-doctor-only ablation answers whether the few-shot effect is Stage 1 or Stage 2, whether Stage 2 cosine prototypes help, and whether simple few-shot fine-tuning helps. It uses the existing leave-one-target-doctor base predictions and frozen cached representations; no Qwen weights, openWakeWord feature extractor weights, audio, checkpoints, or private metadata are written.",
        "",
        "## Protocol Check",
        "",
        f"- Support seeds: `{summary['support_seeds']}`",
        f"- Eligible 3-shot doctors: `{summary['eligible_3shot']}`",
        f"- Eligible 5-shot doctors: `{summary['eligible_5shot']}`",
        "- Base training excludes the target doctor because the source rows are the existing leave-one-target-doctor base rows.",
        "- Support uses target positive clips only.",
        "- Support clips are removed from query.",
        "- Query contains only the target doctor.",
        "- Target negatives are query-only and are never used for adaptation.",
        "- Method and hyperparameter selection uses development pseudo-target doctors only, never the held-out target doctor's query labels.",
        "",
        "## Direct Answers",
        "",
        "1. Our cosine method uses Stage 2 embeddings, not Stage 1.",
        "2. Stage 1 remains the high-recall candidate detector.",
        "3. Stage 2 is the main location for doctor-specific similarity and adaptation.",
        f"4. Best selected method: `{summary['best_method']['method']}` at `{summary['best_method']['shot']}`-shot.",
        "5. Simple fine-tuning is compared against cosine and positive-bias calibration below.",
        "6. Stage 1 fine-tuning is treated as an ablation and must pass FPR safety before being useful.",
        "7. Stage 2 fine-tuning updates only small cached-representation adapters in this experiment; Qwen remains frozen.",
        "8. Train/test separation is strict at the target-doctor split and support/query pairing level.",
        "",
        "## Metrics Table",
        "",
        "| Shot | Method | F1 | Recall | FPR | Delta F1 | Delta Recall | Delta FPR | Improved | Degraded | Safety pass rate | Changed stage |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(per_method, key=lambda item: (int(item["shot"]), METHOD_ORDER.get(str(item["method"]), 99))):
        lines.append(
            f"| {row['shot']} | {row['method']} | {row.get('f1')} | {row.get('recall')} | {row.get('fpr')} | {row.get('delta_f1')} | {row.get('delta_recall')} | {row.get('delta_fpr')} | {row.get('improved_doctors')} | {row.get('degraded_doctors')} | {row.get('safety_pass_rate')} | {row.get('changed_stage')} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Stage 2 cosine result: {summary['interpretation']['stage2_cosine']}",
        f"- Stage 2 fine-tuning result: {summary['interpretation']['stage2_finetune']}",
        f"- Stage 1 fine-tuning result: {summary['interpretation']['stage1_finetune']}",
        f"- Best method: {summary['interpretation']['best_method']}",
        "",
        "## Limitations",
        "",
        "- This run uses cached leave-one-target-out base predictions and cached frozen representations. Fine-tuning methods are lightweight bias/linear/head-style adapters on those cached representations, not a full checkpoint-producing training run.",
        "- The source replay rows are non-target doctors only, but they come from sanitized cached leave-one-target-out rows so that no Qwen/openWakeWord feature extraction is repeated.",
        "- No target negative clips are used for adaptation.",
        "",
        "## Selected Recipes",
        "",
        "Selection JSON is recorded in `fewshot_ablation_summary.json` under `selected_by_method`.",
        "",
    ]
    (REPORTS / "FEWSHOT_ABLATION_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _best_method(per_method: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in per_method if row["method"] != "zero_shot" and bool(row.get("safety_pass"))]
    if not candidates:
        candidates = [row for row in per_method if row["method"] != "zero_shot"]
    return max(
        candidates,
        key=lambda row: (
            float(row.get("delta_f1") or 0.0),
            float(row.get("delta_recall") or 0.0),
            -METHOD_ORDER.get(str(row["method"]), 99),
        ),
    )


def _interpret(per_method: list[dict[str, Any]], best: dict[str, Any]) -> dict[str, str]:
    by_method = {(int(row["shot"]), str(row["method"])): row for row in per_method}
    parts = {}
    for shot in (3, 5):
        cos = by_method.get((shot, "stage2_cosine_prototype"))
        bias = by_method.get((shot, "stage2_positive_bias"))
        if cos and bias:
            parts[f"cos{shot}"] = f"{shot}-shot cosine delta F1 {cos.get('delta_f1')} vs positive-bias delta F1 {bias.get('delta_f1')}"
    s2_ft = [row for row in per_method if str(row["method"]).startswith("stage2_finetune")]
    s1_ft = [row for row in per_method if str(row["method"]).startswith("stage1_finetune")]
    best_s2 = max(s2_ft, key=lambda row: float(row.get("delta_f1") or 0.0), default=None)
    best_s1 = max(s1_ft, key=lambda row: float(row.get("delta_f1") or 0.0), default=None)
    return {
        "stage2_cosine": "; ".join(parts.values()) or "not run",
        "stage2_finetune": f"best Stage 2 fine-tune method `{best_s2['method']}` at {best_s2['shot']}-shot delta F1 {best_s2.get('delta_f1')}" if best_s2 else "not run",
        "stage1_finetune": f"best Stage 1 fine-tune method `{best_s1['method']}` at {best_s1['shot']}-shot delta F1 {best_s1.get('delta_f1')}" if best_s1 else "not run",
        "best_method": f"`{best['method']}` at {best['shot']}-shot with delta F1 {best.get('delta_f1')}, recall {best.get('recall')}, FPR {best.get('fpr')}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, default=DEFAULT_BASE_RUN)
    parser.add_argument("--max-targets", type=int, default=0)
    args = parser.parse_args()
    start = time.time()
    REPORTS.mkdir(parents=True, exist_ok=True)
    base_by_alias = load_base_by_alias(args.base_run)
    records = build_records(base_by_alias, max_targets=args.max_targets)
    eligible_3 = len({record.doctor_alias for record in records if record.shot == 3})
    eligible_5 = len({record.doctor_alias for record in records if record.shot == 5})
    all_rows: list[dict[str, Any]] = []
    all_search_rows: list[dict[str, Any]] = []
    selected_by_method: dict[str, Any] = {}
    for method in METHODS:
        print(json.dumps({"phase": "method", "method": method}, sort_keys=True), flush=True)
        rows, search, selected = _apply_selected_by_target(records, method)
        all_rows.extend(rows)
        all_search_rows.extend(search)
        selected_by_method[method] = selected
    combined_rows, combined_selected = _combine_stage1_stage2(
        records,
        selected_by_method.get("stage1_finetune_bias_only", {}),
        selected_by_method.get("stage2_positive_bias", {}),
    )
    if combined_rows:
        all_rows.extend(combined_rows)
        selected_by_method["stage1_stage2_combined"] = combined_selected
    per_seed = aggregate_per_seed(all_rows)
    per_doctor = aggregate_per_doctor(all_rows)
    per_method = aggregate_per_method(all_rows)
    best = _best_method(per_method)
    summary = {
        "status": "ok",
        "protocol": {
            "target_doctor_only": True,
            "support_positive_only": True,
            "support_removed_from_query": True,
            "target_negatives_used_for_adaptation": False,
            "qwen_weights_updated": False,
            "openwakeword_feature_extractor_updated": False,
            "main_cosine_embedding_stage": "stage2",
            "stage1_cosine_used_for_main_method": False,
            "selection_uses_target_query_labels": False,
            "source_replay_target_policy": "non_target_doctors_only",
        },
        "base_run": str(args.base_run),
        "support_seeds": SUPPORT_SEEDS,
        "eligible_3shot": eligible_3,
        "eligible_5shot": eligible_5,
        "records": len(all_rows),
        "elapsed_sec": time.time() - start,
        "best_method": best,
        "interpretation": _interpret(per_method, best),
        "selected_by_method": selected_by_method,
    }
    write_csv(REPORTS / "fewshot_ablation_support_seed_results.csv", per_seed)
    write_csv(REPORTS / "fewshot_ablation_per_doctor.csv", per_doctor)
    write_csv(REPORTS / "fewshot_ablation_per_method.csv", per_method)
    write_csv(REPORTS / "fewshot_ablation_method_search.csv", all_search_rows)
    write_json(REPORTS / "fewshot_ablation_summary.json", summary)
    _write_report(summary, per_method, selected_by_method)
    print(json.dumps({"status": "ok", "eligible_3shot": eligible_3, "eligible_5shot": eligible_5, "best_method": best["method"], "elapsed_sec": summary["elapsed_sec"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
