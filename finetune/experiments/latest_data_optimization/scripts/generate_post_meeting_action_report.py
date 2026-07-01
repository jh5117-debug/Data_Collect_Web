#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

from vigil_latest_opt.utils import read_json


REPORTS = Path("finetune/experiments/latest_data_optimization/reports")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.4f}%"


def main() -> int:
    stage1 = read_json(REPORTS / "stage1_openwakeword_structure.json")
    stage2 = read_json(REPORTS / "latest_opt_stage2_selected_config.json")
    fewshot = read_json(REPORTS / "target_doctor_fewshot_summary.json")
    librispeech = read_json(REPORTS / "current_qwen_librispeech_benchmark.json")
    shared = read_json(REPORTS / "shared_qwen_asr_diagnostic.json")
    compute = read_json(REPORTS / "latest_opt_compute_cost.json")
    components = {row["component"]: row for row in compute.get("components", [])}
    qwen_encoder = components.get("qwen_audio_encoder_forward", {})
    oww = components.get("official_openwakeword_feature_extraction", {})
    s1_head = components.get("stage1_head", {})
    test = stage2["test"]
    qwen = stage2["baseline"]["qwen_exact"]
    clean = librispeech["per_split"]["test-clean"]
    other = librispeech["per_split"]["test-other"]
    lines = [
        "# Post-Meeting VIGIL Action Report",
        "",
        "## 1. What The Professor Asked",
        "",
        "- Frame this as a VIGIL voice trigger module inside an ASR-based clinical workflow.",
        "- Explain Stage 1 openWakeWord structure clearly.",
        "- Redo few-shot onboarding as target-doctor-only personalization.",
        "- Integrate the corrected LibriSpeech benchmark for the frozen continuous Qwen ASR branch.",
        "- Try to reduce Stage 2 cost through shared Qwen-ASR hidden-state reuse, or document the blocker.",
        "",
        "## 2. Corrected Clinical Workflow",
        "",
        "```text",
        "Microphone audio",
        "  -> Continuous Qwen3-ASR branch",
        "       -> full doctor-patient transcript for the medical report",
        "  -> Parallel VIGIL trigger branch",
        "       -> Stage 1 openWakeWord candidate detector",
        "       -> Stage 2 frozen-Qwen-feature verifier",
        "       -> enter assistant / VQA state when VIGIL is detected",
        "```",
        "",
        "## 3. Stage 1 openWakeWord Structure",
        "",
        "- Stage 1 is not Qwen and not LoRA.",
        "- It is a lightweight KWS front-end.",
        "- Input is 16 kHz audio.",
        "- Frozen front-end: official openWakeWord melspectrogram and embedding ONNX assets.",
        "- Trainable head: `LayerNorm -> 2-layer GRU -> Linear`.",
        f"- Trainable Stage 1 head parameters: `{stage1['head_parameters']['total']}`.",
        f"- openWakeWord feature extraction median/p95: `{fmt(oww.get('median_ms'))}` / `{fmt(oww.get('p95_ms'))}` ms.",
        f"- Stage 1 head median/p95: `{fmt(s1_head.get('median_ms'))}` / `{fmt(s1_head.get('p95_ms'))}` ms.",
        "- Output is `p1`, a candidate probability. Stage 1 does not produce text.",
        "",
        "## 4. Latest VIGIL Trigger Result",
        "",
        "| Method | Recall | FPR | Precision | F1 |",
        "|---|---:|---:|---:|---:|",
        f"| Base Qwen exact keyword | {fmt(qwen.get('recall'))} | {fmt(qwen.get('false_positive_rate'))} | {fmt(qwen.get('precision'))} | {fmt(qwen.get('f1'))} |",
        f"| Optimized two-stage VIGIL trigger | {fmt(test.get('recall'))} | {fmt(test.get('false_positive_rate'))} | {fmt(test.get('precision'))} | {fmt(test.get('f1'))} |",
        "",
        "The VIGIL trigger branch improves wake-word F1 over exact transcript keyword matching while keeping false positives low.",
        "",
        "## 5. Target-Doctor Few-Shot Result",
        "",
        "| Setting | Recall | FPR | Precision | F1 | Delta F1 | Improved | Degraded | Unchanged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in ("3-shot", "5-shot"):
        item = fewshot["aggregate"].get(setting, {})
        adapted = item.get("adapted", {})
        counts = item.get("paired_doctor_counts", {})
        lines.append(
            f"| {setting} | {fmt(adapted.get('recall'))} | {fmt(adapted.get('false_positive_rate'))} | {fmt(adapted.get('precision'))} | {fmt(adapted.get('f1'))} | {fmt(item.get('delta_f1'))} | {counts.get('improved')} | {counts.get('degraded')} | {counts.get('unchanged')} |"
        )
    lines += [
        "",
        f"Conclusion: target-doctor-only support-based onboarding improved safely: `{fewshot['conclusion']['improved']}`. The query is now only the target doctor's remaining clips.",
        "",
        "## 6. LibriSpeech Benchmark For Frozen Qwen ASR",
        "",
        "| Qwen module | Qwen updated? | Benchmark | test-clean WER | test-other WER | Combined WER |",
        "|---|---:|---|---:|---:|---:|",
        f"| Continuous frozen Qwen3-ASR-1.7B | No | LibriSpeech | {pct(clean.get('wer'))} | {pct(other.get('wer'))} | {pct(librispeech.get('combined_normalized_wer'))} |",
        "",
        "This benchmark measures general ASR ability of the frozen continuous Qwen branch. It is separate from VIGIL trigger recall/FPR. If Qwen is updated in the future, LibriSpeech must be rerun on that updated Qwen.",
        "",
        "## 7. Shared-Qwen Status",
        "",
        f"- Status: `{shared['status']}`",
        f"- Extra Qwen encoder cost median: `{fmt(qwen_encoder.get('median_ms'))}` ms per Stage 1 candidate.",
        f"- Exact blocker: {shared['blocker']}",
        "",
        "## 8. Next Step",
        "",
        "- Validate the target-doctor onboarding gain on future blind doctors before treating it as a locked deployment claim.",
        "- Test stronger adaptation methods only if they still avoid target negatives and keep Qwen/openWakeWord frozen.",
        "- For shared Qwen, request or implement an upstream interface that returns decoder-compatible audio hidden states and accepts those states for ASR decoding.",
        "- Keep the current extra encoder-forward prototype until one-forward reuse is proven by call counters.",
        "",
        "## Speaking Script",
        "",
        "In simple English: VIGIL is a voice trigger module for a clinical ASR workflow. Qwen keeps transcribing the full conversation. In parallel, Stage 1 cheaply proposes possible VIGIL events, and Stage 2 verifies them using frozen Qwen audio features. Qwen is not fine-tuned, so the LibriSpeech result is the frozen ASR branch. The shared-Qwen experiment shows that the current wrapper does not expose hidden states for reuse, so Stage 2 still needs an extra encoder forward.",
        "",
        "中文备注：这个系统不是商业语音助手产品，而是临床 ASR 流程里的 VIGIL 触发模块。Qwen 连续转写完整对话；VIGIL 分支并行检测唤醒词。Stage 1 很小，只做候选检测；Stage 2 用冻结的 Qwen 音频特征做验证。当前没有微调 Qwen，所以 LibriSpeech 衡量的是冻结 Qwen ASR 的通用转写能力。shared-Qwen 目前被公开接口阻塞，还不能证明一次 encoder forward 同时服务 ASR 和 Stage 2。",
        "",
    ]
    (REPORTS / "POST_MEETING_ACTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(REPORTS / "POST_MEETING_ACTION_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
