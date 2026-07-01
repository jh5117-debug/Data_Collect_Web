#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from finetune.scripts.extract_openwakeword_features import OfficialOpenWakeWordExtractor
from vigil_latest_opt.long_speech import false_accepts_per_hour, sliding_window_count
from vigil_latest_opt.metrics import metrics_from_rows
from vigil_latest_opt.utils import read_json, read_jsonl, write_json, write_jsonl
from vigil_two_stage.qwen_audio_adapter import FrozenQwenAudioAdapter
from vigil_two_stage.stage1_model import Stage1GRUClassifier
from vigil_two_stage.stage2_model import QwenVerifierHead


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def load_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return audio.astype(np.float32), sr


def write_window(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")


def load_stage1(run_dir: Path, device: str) -> Stage1GRUClassifier:
    cfg = read_json(run_dir / "stage1/model_config.json")
    model = Stage1GRUClassifier(cfg["input_dim"], cfg["gru_hidden_size"], cfg["gru_layers"], cfg["dropout"]).to(device)
    ckpt = torch.load(run_dir / "stage1/checkpoint_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


def load_stage2(run_dir: Path, variant: str, device: str) -> QwenVerifierHead:
    ckpt = torch.load(run_dir / variant / "checkpoint_best.pt", map_location=device)
    model = QwenVerifierHead(int(ckpt["input_dim"]), int(ckpt["config"]["projection_dim"]), int(ckpt["config"]["embedding_dim"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


def stage1_score(model: Stage1GRUClassifier, features: np.ndarray, device: str) -> float:
    x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
    lengths = torch.tensor([features.shape[0]], device=device)
    with torch.no_grad():
        return float(torch.sigmoid(model(x, lengths)).detach().cpu().item())


def stage2_score(model: QwenVerifierHead, features: torch.Tensor, device: str) -> float:
    hidden = features.detach().float().to(device).unsqueeze(0)
    mask = torch.ones(1, hidden.shape[1], dtype=torch.bool, device=device)
    with torch.no_grad():
        return float(torch.sigmoid(model(hidden, mask)["logit"]).detach().cpu().item())


def select_manifest_rows(manifest_paths: list[Path], max_per_split: int | None) -> list[dict[str, Any]]:
    rows = []
    for path in manifest_paths:
        split_rows = [row for row in read_jsonl(path) if "vigil" not in normalize_words(row["reference"])]
        if max_per_split is not None:
            split_rows = split_rows[:max_per_split]
        rows.extend(split_rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="finetune/experiments/latest_data/runs/nested_zero_shot")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--reports", default="finetune/experiments/latest_data_optimization/reports")
    parser.add_argument("--selected-config", default="finetune/experiments/latest_data_optimization/reports/latest_opt_stage2_selected_config.json")
    parser.add_argument("--work-dir", default="finetune/experiments/latest_data_optimization/runs/long_speech_subset")
    parser.add_argument("--max-utterances-per-split", type=int, default=20)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=0.25)
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    window_dir = work_dir / "windows"
    selected = read_json(args.selected_config)
    variant = selected["variant"]
    threshold = float(selected["thresholds"][args.fold])
    run_dir = Path(args.run_root) / f"fold_{args.fold}"
    theta1 = float(read_json(run_dir / "stage1/threshold.json")["threshold"])
    top_k = int(selected["top_k"])
    stage1 = load_stage1(run_dir, device)
    stage2 = load_stage2(run_dir, variant, device)
    openwakeword = OfficialOpenWakeWordExtractor()
    qwen = FrozenQwenAudioAdapter("Qwen/Qwen3-ASR-1.7B")
    qwen.load()
    rows = select_manifest_rows(
        [Path("finetune/benchmarks/asr/manifests/test_clean.jsonl"), Path("finetune/benchmarks/asr/manifests/test_other.jsonl")],
        args.max_utterances_per_split if args.max_utterances_per_split > 0 else None,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    per_utt = []
    false_accepts = []
    for item in rows:
        audio, sr = load_audio(item["audio_path"])
        samples_per_window = int(args.window_seconds * sr)
        stride = int(args.stride_seconds * sr)
        if len(audio) < samples_per_window:
            starts = [0]
        else:
            starts = list(range(0, max(1, len(audio) - samples_per_window + 1), stride))
        candidates = []
        for idx, start in enumerate(starts):
            chunk = audio[start : start + samples_per_window]
            if len(chunk) < samples_per_window:
                chunk = np.pad(chunk, (0, samples_per_window - len(chunk)))
            wav_path = window_dir / item["split"] / f"{item['id']}_w{idx:04d}.wav"
            write_window(wav_path, chunk, sr)
            features = openwakeword.extract(wav_path)
            s1 = stage1_score(stage1, features, device)
            if s1 >= theta1:
                candidates.append({"window_index": idx, "start_sec": start / sr, "end_sec": (start + samples_per_window) / sr, "stage1_score": s1, "wav_path": str(wav_path)})
        selected_candidates = sorted(candidates, key=lambda row: float(row["stage1_score"]), reverse=True)[:top_k]
        final_trigger = False
        evaluated = []
        for cand in selected_candidates:
            features = qwen.extract_audio_features(cand["wav_path"])
            s2 = stage2_score(stage2, features, device)
            accepted = s2 >= threshold
            evaluated.append({**{k: v for k, v in cand.items() if k != "wav_path"}, "stage2_score": s2, "accepted": accepted})
            final_trigger = final_trigger or accepted
        row = {
            "utterance_id": item["id"],
            "split": item["split"],
            "duration_sec": float(item["duration_sec"]),
            "windows": len(starts),
            "stage1_candidates": len(candidates),
            "stage2_invocations": len(selected_candidates),
            "final_trigger": final_trigger,
            "reference": item["reference"],
            "evaluated_candidates": evaluated,
        }
        per_utt.append(row)
        if final_trigger:
            false_accepts.append(row)
    total_seconds = sum(float(row["duration_sec"]) for row in per_utt)
    by_split = defaultdict(list)
    for row in per_utt:
        by_split[row["split"]].append(row)
    split_summary = {}
    for split, group in sorted(by_split.items()):
        secs = sum(float(row["duration_sec"]) for row in group)
        split_summary[split] = {
            "utterances": len(group),
            "audio_hours": secs / 3600.0,
            "windows": sum(int(row["windows"]) for row in group),
            "stage1_candidates": sum(int(row["stage1_candidates"]) for row in group),
            "stage2_invocations": sum(int(row["stage2_invocations"]) for row in group),
            "false_accepts": sum(1 for row in group if row["final_trigger"]),
            "false_accepts_per_hour": false_accepts_per_hour(sum(1 for row in group if row["final_trigger"]), secs) if secs > 0 else None,
        }
    summary = {
        "status": "subset" if args.max_utterances_per_split > 0 else "full",
        "device": device,
        "model_fold": args.fold,
        "variant": variant,
        "theta1": theta1,
        "theta2": threshold,
        "top_k": top_k,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "utterances": len(per_utt),
        "total_audio_hours": total_seconds / 3600.0,
        "total_windows": sum(int(row["windows"]) for row in per_utt),
        "stage1_candidates": sum(int(row["stage1_candidates"]) for row in per_utt),
        "stage1_candidates_per_hour": false_accepts_per_hour(sum(int(row["stage1_candidates"]) for row in per_utt), total_seconds),
        "stage2_invocations": sum(int(row["stage2_invocations"]) for row in per_utt),
        "final_false_accepts": len(false_accepts),
        "false_accepts_per_hour": false_accepts_per_hour(len(false_accepts), total_seconds),
        "split_summary": split_summary,
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None,
        "qwen_extraction_path": qwen.extraction_path,
        "metrics_if_treated_as_all_negative": metrics_from_rows([{**row, "label": 0, "decision": row["final_trigger"]} for row in per_utt]),
    }
    write_json(reports / "latest_opt_long_speech_summary.json", summary)
    write_jsonl(reports / "latest_opt_long_speech_false_accepts.jsonl", false_accepts[:50])
    lines = [
        "# Latest Optimized Long-Speech False Accepts Per Hour",
        "",
        f"- Status: `{summary['status']}`",
        f"- Model fold used: `{args.fold}`",
        f"- Variant/top_k/theta1/theta2: `{variant}` / `{top_k}` / `{theta1}` / `{threshold}`",
        f"- Utterances: `{summary['utterances']}`",
        f"- Total audio hours: `{summary['total_audio_hours']}`",
        f"- Total windows: `{summary['total_windows']}`",
        f"- Stage1 candidates: `{summary['stage1_candidates']}`",
        f"- Stage2 invocations: `{summary['stage2_invocations']}`",
        f"- Final false accepts: `{summary['final_false_accepts']}`",
        f"- False accepts per hour: `{summary['false_accepts_per_hour']}`",
        "",
        "| Split | Hours | Utterances | Windows | Stage1 candidates | False accepts | FAPH |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, item in split_summary.items():
        lines.append(
            f"| {split} | {item['audio_hours']} | {item['utterances']} | {item['windows']} | "
            f"{item['stage1_candidates']} | {item['false_accepts']} | {item['false_accepts_per_hour']} |"
        )
    if false_accepts:
        lines.extend(["", "## False Activation Examples", ""])
        for row in false_accepts[:5]:
            lines.append(f"- `{row['utterance_id']}` {row['split']} candidates `{row['evaluated_candidates']}`")
    (reports / "LATEST_OPT_LONG_SPEECH_FALSE_ACCEPTS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": summary["status"], "utterances": len(per_utt), "false_accepts_per_hour": summary["false_accepts_per_hour"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
