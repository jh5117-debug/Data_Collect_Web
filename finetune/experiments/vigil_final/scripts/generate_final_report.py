#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from vigil_final.utils import read_json


def optional_json(path: str) -> dict:
    p = Path(path)
    return read_json(p) if p.exists() else {"status": "missing"}


def main() -> int:
    nested = optional_json("finetune/experiments/vigil_final/reports/nested_zero_shot_summary.json")
    fewshot = optional_json("finetune/experiments/vigil_final/reports/real_few_shot_summary.json")
    cost = optional_json("finetune/experiments/vigil_final/reports/component_cost_summary.json")
    shared = optional_json("finetune/experiments/vigil_final/reports/shared_qwen_diagnostic.json")
    ablation = optional_json("finetune/experiments/vigil_final/reports/balanced_vs_full_summary.json")
    long = optional_json("finetune/experiments/vigil_final/reports/long_speech_summary.json")
    blind = optional_json("finetune/experiments/vigil_final/reports/blind_test_lock.json")
    nested_methods = nested.get("methods", {})
    few_conditions = fewshot.get("conditions", {})
    components = cost.get("components", [])
    stage1 = components[0] if len(components) > 0 else {}
    stage2 = components[1] if len(components) > 1 else {}
    lines = [
        "# Final Professor Meeting Report",
        "",
        "## Summary",
        "",
        "We first corrected the evaluation protocol. The old few-shot table used a no-adaptation fallback, so it was not learned personalization.",
        "The strict nested participant-CV result strengthens the Stage 2 conclusion: Stage 2 keeps high recall while reducing the Stage 1 false-positive rate.",
        "The real prototype onboarding path is implemented and selected with development pseudo-targets, but the outer-test paired result is negative: 3-shot and 5-shot did not improve F1.",
        "",
        "## Strict Nested CV",
        "",
        f"Status: `{nested.get('status')}`",
        "",
        "| Method | Recall | FPR | Precision | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("qwen_exact", "stage1_only", "stage2_bce", "stage2_bce_supcon", "validation_selected"):
        item = nested_methods.get(method)
        if item:
            lines.append(
                f"| {method} | {item['recall']['mean']:.6f} | {item['false_positive_rate']['mean']:.6f} | "
                f"{item['precision']['mean']:.6f} | {item['f1']['mean']:.6f} |"
            )
    lines.extend(
        [
        "",
        "## Stage 2 Operating Point",
        "",
        "See `STAGE2_OPERATING_POINT_REPORT.md`. Threshold targets were selected from development OOF predictions only.",
        "",
        "## Real 0/3/5-Shot Onboarding",
        "",
        f"Status: `{fewshot.get('status')}`",
        f"Learned personalization claimed: `{fewshot.get('learned_personalization_claimed')}`",
        "",
        "| Condition | Recall | FPR | Precision | F1 |",
        "|---|---:|---:|---:|---:|",
    ])
    for key, label in (("0_for_3", "0-shot on 3-shot query"), ("3", "3-shot prototype"), ("0_for_5", "0-shot on 5-shot query"), ("5", "5-shot prototype")):
        pooled = few_conditions.get(key, {}).get("pooled", {})
        if pooled:
            lines.append(
                f"| {label} | {pooled.get('recall')} | {pooled.get('false_positive_rate')} | {pooled.get('precision')} | {pooled.get('f1')} |"
            )
    lines.extend([
        "",
        "3-shot and 5-shot support recordings changed model outputs, but the paired F1 deltas were negative. This is a safe negative result rather than an onboarding win.",
        "",
        "## Accuracy-Cost",
        "",
        f"Status: `{cost.get('status')}`",
        "",
        "| Component | Median latency | P95 latency | Peak allocated VRAM |",
        "|---|---:|---:|---:|",
        f"| Stage 1 head | {stage1.get('median')} | {stage1.get('p95')} | {stage1.get('peak_allocated_gb')} GB |",
        f"| Stage 2 head | {stage2.get('median')} | {stage2.get('p95')} | {stage2.get('peak_allocated_gb')} GB |",
        "",
        "Current System C accounting: one loaded Qwen instance, one continuous ASR forward, plus one additional audio-encoder forward for each Stage 2 candidate.",
        "",
        "## Shared-Qwen",
        "",
        f"Status: `{shared.get('status')}`",
        f"Blocker: {shared.get('blocker')}",
        "",
        "## Balanced Versus Full Data",
        "",
        f"Status: `{ablation.get('status')}`. Full unbalanced nested ablation was not run, so no full-data result is claimed.",
        "",
        "## Long-Speech False Activations",
        "",
        f"Status: `{long.get('status')}`. No false-activations-per-hour result is claimed.",
        "",
        "## Future Blind-Test Protocol",
        "",
        f"Status: lock fields present = `{bool(blind)}`. The lock is a protocol scaffold until final thresholds/checkpoints are frozen.",
        "",
        "## Limitations",
        "",
        "- Full unbalanced-data ablation is pending.",
        "- Complete Qwen ASR/audio-encoder/cascade latency benchmark is pending.",
        "- Gradient adaptation search is guarded in code but not fully run.",
        "- Long-speech false-activation stress test waits for a locked final model.",
        "- The final deployment candidate is not trained because the research choices are not fully frozen.",
        "",
        "## Simple Speaking Script",
        "",
        "We first corrected the evaluation protocol. The old few-shot table used a no-adaptation fallback. We now separate development selection from outer-test reporting, and we do not claim personalization unless support recordings change the model output.",
    ])
    Path("finetune/experiments/vigil_final/reports/FINAL_PROFESSOR_MEETING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
