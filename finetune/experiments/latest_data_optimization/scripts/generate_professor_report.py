#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from vigil_latest_opt.utils import read_json


def fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def main() -> int:
    reports = Path("finetune/experiments/latest_data_optimization/reports")
    audit = read_json(reports / "optimization_start_audit.json")
    selected = read_json(reports / "latest_opt_stage2_selected_config.json")
    few = read_json(reports / "latest_opt_real_few_shot_summary.json")
    cost = read_json(reports / "latest_opt_compute_cost.json")
    long_speech = read_json(reports / "latest_opt_long_speech_summary.json")
    ablation = read_json(reports / "latest_opt_balanced_vs_full_summary.json")
    shared = read_json(reports / "latest_opt_shared_qwen_diagnostic.json")
    final_model = read_json(reports / "latest_opt_final_model_manifest.json")
    errors = read_json(reports / "latest_opt_stage_error_summary.json")
    stage1 = selected["baseline"]["stage1_only_recomputed"]
    qwen = selected["baseline"]["qwen_exact"]
    stage2 = selected["test"]
    c_stage2 = next(row for row in cost["components"] if row["component"] == "stage2_head")
    c_encoder = next(row for row in cost["components"] if row["component"] == "qwen_audio_encoder_forward")
    lines = [
        "# Latest Optimized VIGIL Professor Meeting Report",
        "",
        "## English Summary",
        "",
        f"- Latest export: {audit['raw_canonical_clips']} raw clips; {audit['valid_unique_clips_after_qc']} valid clips after audio QC; {audit['audio_qc_rejected']} silent clips rejected.",
        f"- Formal balanced max-100 set: {audit['balanced_clips']} clips, {audit['balanced_windows']} windows, {audit['balanced_participants']} participants.",
        "- Split policy: participant-level five-fold CV; the same participant never appears in both train/development and outer test.",
        f"- Qwen transcript keyword baseline: F1 {fmt(qwen['f1'])}, recall {fmt(qwen['recall'])}, FPR {fmt(qwen['false_positive_rate'])}.",
        f"- Optimized two-stage detector: F1 {fmt(stage2['f1'])}, recall {fmt(stage2['recall'])}, FPR {fmt(stage2['false_positive_rate'])}, precision {fmt(stage2['precision'])}.",
        f"- Stage1-only: F1 {fmt(stage1['f1'])}, recall {fmt(stage1['recall'])}, FPR {fmt(stage1['false_positive_rate'])}.",
        f"- Selected Stage2 config: {selected['variant']}, top_k={selected['top_k']}, threshold-only, selected on development predictions only.",
        f"- Stage errors after optimization: {errors['counts']}.",
        f"- Real few-shot onboarding: {few['claim']}",
        f"- 3-shot paired F1: {fmt(few['conditions']['3-shot']['adapted']['f1'])}; 5-shot paired F1: {fmt(few['conditions']['5-shot']['adapted']['f1'])}.",
        f"- Compute: Stage2 head median {fmt(c_stage2['median_ms'])} ms; extra Qwen encoder forward median {fmt(c_encoder['median_ms'])} ms on {c_encoder['n']} measured samples.",
        f"- Long normal speech: {long_speech['status']} LibriSpeech run, {fmt(long_speech['total_audio_hours'])} h, {long_speech['final_false_accepts']} false accepts, FAPH {fmt(long_speech['false_accepts_per_hour'])}. Full run is tracked separately if tmux is active.",
        f"- Balanced-vs-full ablation: {ablation['status']}; full-unbalanced selected Stage2 F1 {fmt(ablation['full_unbalanced']['selected_stage2']['pooled']['f1'])}, FPR {fmt(ablation['full_unbalanced']['selected_stage2']['pooled']['false_positive_rate'])}.",
        f"- Shared Qwen status: {shared['status']} because the public wrapper does not expose reusable ASR hidden states.",
        f"- Final deployment candidate: {final_model['status']}; Qwen weights are not included in the bundle.",
        "",
        "## Simple Chinese Summary",
        "",
        f"- 最新数据有 {audit['raw_canonical_clips']} 条原始 clips；音频质检后 {audit['valid_unique_clips_after_qc']} 条有效 clips，剔除了 {audit['audio_qc_rejected']} 条静音。",
        f"- 正式 balanced max-100 数据集有 {audit['balanced_clips']} clips、{audit['balanced_participants']} 位参与者。",
        "- 我们使用 participant-level split，所以同一个人不会同时出现在训练和测试里。",
        f"- Qwen 文字关键词 baseline 的 F1 是 {fmt(qwen['f1'])}；优化后的 two-stage detector F1 是 {fmt(stage2['f1'])}，recall 是 {fmt(stage2['recall'])}，FPR 是 {fmt(stage2['false_positive_rate'])}。",
        f"- Stage2 把 Stage1-only 的 FPR 从 {fmt(stage1['false_positive_rate'])} 降到 {fmt(stage2['false_positive_rate'])}，同时 F1 从 {fmt(stage1['f1'])} 提高到 {fmt(stage2['f1'])}。",
        f"- 3-shot/5-shot 个性化已经真实实现并评估，但结果是：{few['claim']}",
        f"- 计算成本方面，Stage2 head 中位数 {fmt(c_stage2['median_ms'])} ms，额外 Qwen encoder forward 中位数 {fmt(c_encoder['median_ms'])} ms。",
        f"- 普通长语音 false wake-up subset 结果是 {fmt(long_speech['false_accepts_per_hour'])} 次/小时；完整 LibriSpeech run 若仍在跑，需要等 tmux 完成后更新。",
        "- shared-Qwen 目前被 runtime interface 阻塞；需要 qwen_asr 暴露可复用 hidden states。",
        "",
        "## Speaking Script",
        "",
        f"On the latest dataset, we have {audit['valid_unique_clips_after_qc']} valid clips from {audit['balanced_participants']} balanced participants. "
        "We use participant-level folds, so the same person never appears in both training and testing. "
        f"The Qwen transcript baseline gets F1 {fmt(qwen['f1'])}. "
        f"Our selected two-stage detector gets F1 {fmt(stage2['f1'])}, recall {fmt(stage2['recall'])}, and FPR {fmt(stage2['false_positive_rate'])}. "
        f"Stage2 reduces false triggers from {fmt(stage1['false_positive_rate'])} to {fmt(stage2['false_positive_rate'])}, but adds about {fmt(c_encoder['median_ms'])} ms for the extra Qwen encoder forward plus {fmt(c_stage2['median_ms'])} ms for the verifier head. "
        f"For 3-shot/5-shot onboarding, we tried support-based personalization; it did not improve paired F1 safely. "
        f"Long-speech false accepts per hour is {fmt(long_speech['false_accepts_per_hour'])} on the current subset. "
        "The best next step is to let the full long-speech job finish and then run a future blind export against the trained deployment candidate.",
        "",
        "## Limitations",
        "",
        "- Long-speech report committed here is a subset, not the full LibriSpeech test-clean/test-other stress test.",
        "- Full-unbalanced ablation is heads-only because full-unbalanced Qwen transcript cache was not generated.",
        "- Final model bundle is for deployment/future blind test only, not a new held-out scientific score.",
    ]
    (reports / "LATEST_OPT_PROFESSOR_MEETING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(reports / "LATEST_OPT_PROFESSOR_MEETING_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
