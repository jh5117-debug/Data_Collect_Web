#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: dict[str, Any], method: str, name: str) -> float:
    return float(summary["methods"][method][name]["mean"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="finetune/experiments/latest_data/reports")
    parser.add_argument("--shared", default="finetune/experiments/latest_data/shared")
    parser.add_argument("--dataset-report", default="finetune/data/processed/2b78e211183d47fb/dataset_report.json")
    args = parser.parse_args()
    reports = Path(args.reports)
    shared = Path(args.shared)
    reports.mkdir(parents=True, exist_ok=True)

    summary = read_json(reports / "admin_summary_before_export.json")
    inspection = read_json(reports / "latest_export_inspection.json")
    rejection = read_json(reports / "latest_audio_rejection_summary.json")
    dataset = read_json(Path(args.dataset_report))
    balanced = read_json(shared / "latest_balanced_summary.json")
    protocol = read_json(shared / "latest_shared_experiment_protocol.json")
    nested = read_json(reports / "latest_nested_zero_shot_summary.json")
    few = read_json(reports / "latest_real_few_shot_summary.json")
    recipe = read_json(reports / "latest_selected_few_shot_recipe.json")
    cost = read_json(reports / "latest_compute_accuracy_tradeoff.json")
    shared_qwen = read_json(reports / "latest_shared_qwen_diagnostic.json")
    long = read_json(reports / "latest_long_speech_summary.json")
    final = read_json(reports / "latest_final_model_status.json")
    blind = read_json(reports / "latest_blind_test_lock.json")

    lines = [
        "# Latest Professor Meeting Report",
        "",
        "## Latest Data Counts",
        "",
        f"- Production raw clips: `{summary['total_clips']}` ({summary['positive_clips']} positive, {summary['negative_clips']} negative).",
        f"- Accounts shown by Admin: `{summary['participants']}`; submitted sessions: `{summary['submitted_sessions']}` / `{summary['sessions']}`.",
        "- ZIP source: read-only Admin fallback export because production export job failed at 433/1673 with read timeout.",
        f"- ZIP SHA-256: `{inspection['zip_sha256']}`.",
        f"- Valid unique clips after audio QC: `{balanced['source_unique_clips']}`; manifest windows: `{dataset['manifest_windows']}`.",
        f"- Rejected silent clips: `{rejection['rejected_silent_count']}` (`{rejection['clip_id_min']}` to `{rejection['clip_id_max']}`).",
        f"- Prompt groups: P1 `{summary['prompt_group_counts']['P1_vigil_only']}`, P2 `{summary['prompt_group_counts']['P2_phrase_plus_vigil']}`, P3 `{summary['prompt_group_counts']['P3_vigil_plus_phrase']}`, P4 `{summary['prompt_group_counts']['P4_negative']}`.",
        "",
        "## Balanced Max-100 Dataset",
        "",
        f"- Balanced clips: `{balanced['clips_after']}`; windows: `{balanced['windows_after_cap']}`; participants: `{balanced['participants']}`.",
        f"- Balanced manifest SHA-256: `{balanced['balanced_manifest_sha256']}`.",
        "",
        "## Five-Fold Table",
        "",
        "| Fold | Participants | Clips | Pos | Neg | P1 | P2 | P3 | P4 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in protocol["folds"]:
        p = fold["prompt_counts"]
        lines.append(
            f"| {fold['fold']} | {fold['participants']} | {fold['clips']} | {fold['positive']} | {fold['negative']} | "
            f"{p.get('P1_vigil_only', 0)} | {p.get('P2_phrase_plus_vigil', 0)} | {p.get('P3_vigil_plus_phrase', 0)} | {p.get('P4_negative', 0)} |"
        )

    lines += [
        "",
        "## Nested Zero-Shot Table",
        "",
        "| Method | Recall | FPR | Precision | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ["qwen_exact", "stage1_only", "stage2_bce", "stage2_bce_supcon", "validation_selected"]:
        lines.append(
            f"| {method} | {metric(nested, method, 'recall'):.6f} | "
            f"{metric(nested, method, 'false_positive_rate'):.6f} | "
            f"{metric(nested, method, 'precision'):.6f} | {metric(nested, method, 'f1'):.6f} |"
        )

    lines += [
        "",
        "## Stage 2 Operating Point Table",
        "",
        "- Status: `requires_inner_oof_predictions`; no development-only operating point is claimed yet.",
        "",
        "## Real 0/3/5-Shot Onboarding Table",
        "",
        f"- Selected recipe: `{recipe['selected_recipe']}`.",
        f"- Safe support-using improvement found: `{recipe['safe_improvement_found']}`.",
        "- The table uses the no-adaptation fallback, so 3-shot/5-shot differences are not adaptation-improvement claims.",
        "",
        "| Condition | Recall | FPR | Precision | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for cond in ["0", "3", "5"]:
        item = few["conditions"][cond]
        lines.append(f"| {cond}-shot | {item['recall']:.6f} | {item['false_positive_rate']:.6f} | {item['precision']:.6f} | {item['f1']:.6f} |")

    lines += [
        "",
        "## Stage Error Analysis",
        "",
        "- Validation-selected errors: 33 Stage 1 misses, 79 Stage 2 rejects, 32 Stage 1 false candidates.",
        "",
        "## Accuracy-Cost Table",
        "",
        f"- Status: `{cost['status']}`; full Qwen forward latency is still a limitation.",
        "| Component | Median ms | P95 ms | Peak GB |",
        "|---|---:|---:|---:|",
    ]
    for component in cost["components"]:
        lines.append(f"| {component['component']} | {component['median_ms']:.6f} | {component['p95_ms']:.6f} | {component['peak_allocated_gb']:.6f} |")

    lines += [
        "",
        "## Shared-Qwen Status",
        "",
        f"- Status: `{shared_qwen['status']}`.",
        f"- Blocker: {shared_qwen['blocker']}",
        "",
        "## Long-Speech False Accepts/Hour",
        "",
        f"- Status: `{long['status']}`. No false-accepts/hour result is claimed.",
        "",
        "## Final Model Status",
        "",
        f"- Status: `{final['status']}`. Choices are not frozen, so no deployment bundle was trained.",
        "",
        "## Blind-Test Protocol",
        "",
        f"- Status: `{blind['status']}`. Future exports must reject known development participants and cannot tune thresholds.",
        "",
        "## Known Limitations",
        "",
        "- Production background export failed; latest ZIP was reconstructed through read-only Admin APIs.",
        "- 55 raw clips were silent and excluded from training/evaluation manifests.",
        "- Stage 2 operating-point search still needs inner OOF predictions.",
        "- Few-shot support-using adaptation was not safely proven on this latest run.",
        "- Full Qwen/cascade latency and long-speech false accepts/hour were not completed.",
        "",
        "## Speaking Script",
        "",
        "English:",
        "",
        f"- The latest collection has {summary['total_clips']} raw clips. After audio QC, {rejection['rejected_silent_count']} silent clips were excluded.",
        f"- We capped each participant at 100 clips, leaving {balanced['clips_after']} balanced clips.",
        "- No participant appears in both train and test folds.",
        "- The participant-level five-fold zero-shot result is the strict unseen-participant estimate.",
        f"- The validation-selected two-stage method reached recall {metric(nested, 'validation_selected', 'recall'):.3f} with FPR {metric(nested, 'validation_selected', 'false_positive_rate'):.3f}.",
        f"- Stage1 alone had higher recall {metric(nested, 'stage1_only', 'recall'):.3f}, but higher FPR {metric(nested, 'stage1_only', 'false_positive_rate'):.3f}.",
        "- We do not yet have a safe support-using few-shot improvement.",
        "- We still need inner operating-point search, full latency, long-speech stress testing, and future blind-test data.",
        "",
        "Chinese:",
        "",
        f"- 最新收集有 {summary['total_clips']} 条原始 clips，其中 {rejection['rejected_silent_count']} 条是静音，已经从训练/评估 manifest 里剔除。",
        f"- 每个参与者最多保留 100 条，平衡后是 {balanced['clips_after']} 条 clips。",
        "- 五折是按参与者切分的，同一个人不会同时出现在训练和测试里。",
        f"- validation-selected 两阶段方法的 recall 是 {metric(nested, 'validation_selected', 'recall'):.3f}，FPR 是 {metric(nested, 'validation_selected', 'false_positive_rate'):.3f}。",
        "- 目前还不能声称 few-shot adaptation 有安全收益。",
        "- 下一步需要补 inner operating point、完整延迟、长语音误触发和未来盲测。",
        "",
        "## Next Decision",
        "",
        "Decide whether deployment prioritizes zero FPR via Stage2 or higher recall via Stage1, then run inner OOF operating-point search before locking a final model.",
    ]
    out = reports / "LATEST_PROFESSOR_MEETING_REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
