#!/usr/bin/env python3
from __future__ import annotations

import argparse
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch

from finetune.scripts.extract_openwakeword_features import OfficialOpenWakeWordExtractor
from vigil_latest_opt.cascade import clip_score_rows
from vigil_latest_opt.metrics import metrics_from_rows
from vigil_latest_opt.timing import cuda_synchronized_timer, summarize_seconds
from vigil_latest_opt.utils import read_json, read_jsonl, write_csv, write_json
from vigil_two_stage.qwen_audio_adapter import FrozenQwenAudioAdapter, QwenAdapterUnavailable
from vigil_two_stage.stage1_model import Stage1GRUClassifier, count_parameters
from vigil_two_stage.stage2_model import QwenVerifierHead


def wav_seconds(path: str | Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def load_npz(path: str) -> np.ndarray:
    data = np.load(path)
    return (data["features"] if "features" in data else data[data.files[0]]).astype(np.float32)


def choose_subset(manifest: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    buckets = {
        "P1_vigil_only": 20,
        "P2_phrase_plus_vigil": 20,
        "P3_vigil_plus_phrase": 20,
        "P4_negative": 40,
    }
    out = []
    seen = set()
    for group, target in buckets.items():
        rows = [row for row in manifest if row.get("prompt_group") == group]
        step = max(1, len(rows) // target) if rows else 1
        for row in rows[::step]:
            if len([x for x in out if x.get("prompt_group") == group]) >= target:
                break
            if row["clip_id"] not in seen:
                out.append(row)
                seen.add(row["clip_id"])
    if len(out) < n:
        for row in manifest:
            if row["clip_id"] not in seen:
                out.append(row)
                seen.add(row["clip_id"])
            if len(out) >= n:
                break
    return out[:n]


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


def _run_stage1(stage1: Stage1GRUClassifier, feature_path: str, device: str) -> None:
    s1_arr = load_npz(feature_path)
    x = torch.from_numpy(s1_arr).unsqueeze(0).to(device)
    lengths = torch.tensor([s1_arr.shape[0]], device=device)
    _ = torch.sigmoid(stage1(x, lengths))


def _run_stage2(stage2: QwenVerifierHead, feature_path: str, device: str) -> None:
    q_arr = load_npz(feature_path)
    hidden = torch.from_numpy(q_arr).unsqueeze(0).to(device)
    mask = torch.ones(1, q_arr.shape[0], dtype=torch.bool, device=device)
    _ = stage2(hidden, mask)


def measure_heads(run_dir: Path, variant: str, sample_rows: list[dict[str, Any]], device: str, warmups: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage1 = load_stage1(run_dir, device)
    stage2 = load_stage2(run_dir, variant, device)
    stage1_by_clip = {row["clip_id"]: row for row in read_jsonl(run_dir / "stage1_features_manifest.jsonl")}
    qwen_by_clip = {row["clip_id"]: row for row in read_jsonl(run_dir / "stage2_qwen_features/qwen_features_manifest.jsonl")}
    stage1_times = []
    stage2_times = []
    stage1_total_times = []
    stage2_total_times = []
    pairs = [(stage1_by_clip.get(row["clip_id"]), qwen_by_clip.get(row["clip_id"])) for row in sample_rows]
    pairs = [(s1, q) for s1, q in pairs if s1 and q]
    with torch.no_grad():
        for s1, q in pairs[:warmups]:
            _run_stage1(stage1, s1["feature_path"], device)
            _run_stage2(stage2, q["feature_path"], device)
        for s1, q in pairs:
            with cuda_synchronized_timer() as timer:
                with cuda_synchronized_timer() as head_timer:
                    _run_stage1(stage1, s1["feature_path"], device)
                stage1_times.append(head_timer["seconds"])
            stage1_total_times.append(timer["seconds"])
            with cuda_synchronized_timer() as timer:
                with cuda_synchronized_timer() as head_timer:
                    _run_stage2(stage2, q["feature_path"], device)
                stage2_times.append(head_timer["seconds"])
            stage2_total_times.append(timer["seconds"])
    components = [
        {"component": "stage1_head", **summarize_seconds(stage1_times)},
        {"component": "stage1_cached_feature_load_plus_head", **summarize_seconds(stage1_total_times)},
        {"component": "stage2_head", **summarize_seconds(stage2_times)},
        {"component": "stage2_cached_qwen_feature_load_plus_head", **summarize_seconds(stage2_total_times)},
    ]
    params = {
        "stage1": count_parameters(stage1),
        "stage2": {"total": sum(p.numel() for p in stage2.parameters()), "trainable": sum(p.numel() for p in stage2.parameters() if p.requires_grad)},
    }
    return components, params


def measure_openwakeword(sample_rows: list[dict[str, Any]], warmups: int) -> dict[str, Any]:
    extractor = OfficialOpenWakeWordExtractor()
    times = []
    for row in sample_rows[:warmups]:
        _ = extractor.extract(Path(row["window_wav_path"]))
    for row in sample_rows:
        with cuda_synchronized_timer() as timer:
            _ = extractor.extract(Path(row["window_wav_path"]))
        times.append(timer["seconds"])
    return {"component": "official_openwakeword_feature_extraction", **summarize_seconds(times), "openwakeword_version": extractor.version}


def measure_qwen_encoder(sample_rows: list[dict[str, Any]], model_name: str, limit: int, warmups: int) -> dict[str, Any]:
    if limit <= 0:
        return {"component": "qwen_audio_encoder_forward", "status": "skipped", "n": 0}
    adapter = FrozenQwenAudioAdapter(model_name)
    try:
        adapter.load()
        integrity = adapter.integrity()
    except QwenAdapterUnavailable as exc:
        return {"component": "qwen_audio_encoder_forward", "status": "blocked", "reason": str(exc), "n": 0}
    times = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for row in sample_rows[: min(warmups, len(sample_rows))]:
        _ = adapter.extract_audio_features(str(Path(row["window_wav_path"]).resolve()))
    for row in sample_rows[:limit]:
        with cuda_synchronized_timer() as timer:
            _ = adapter.extract_audio_features(str(Path(row["window_wav_path"]).resolve()))
        times.append(timer["seconds"])
    return {
        "component": "qwen_audio_encoder_forward",
        "status": "ok",
        **summarize_seconds(times),
        "extraction_path": adapter.extraction_path,
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None,
        "total_qwen_parameters": integrity.total_parameters,
        "trainable_qwen_parameters": integrity.trainable_parameters,
    }


def qwen_asr_from_cache(cache_rows: list[dict[str, Any]], sample_ids: set[str]) -> dict[str, Any]:
    latencies = [float(row["latency_sec"]) for row in cache_rows if row["clip_id"] in sample_ids and row.get("latency_sec") is not None]
    peaks = [float(row["peak_gpu_memory_gb"]) for row in cache_rows if row["clip_id"] in sample_ids and row.get("peak_gpu_memory_gb") is not None]
    result = {"component": "qwen_asr_transcript_from_recorded_cache", **summarize_seconds(latencies)}
    result["peak_gpu_memory_gb_max"] = max(peaks) if peaks else None
    result["source"] = "qwen_transcript_cache_balanced_max100_latest.jsonl recorded latency_sec"
    return result


def candidate_rate(run_root: Path, selected: dict[str, Any]) -> dict[str, Any]:
    all_rows = []
    total_seconds_by_clip = {}
    manifest = read_jsonl("finetune/experiments/latest_data/shared/balanced_max100_latest_manifest.jsonl")
    for row in manifest:
        total_seconds_by_clip.setdefault(row["clip_id"], wav_seconds(row["full_wav_path"]))
    for fold in range(5):
        root = run_root / f"fold_{fold}"
        rows = clip_score_rows(
            read_jsonl(root / "stage1/test_predictions.jsonl"),
            read_jsonl(root / selected["variant"] / "test_predictions.jsonl"),
            theta1=float(read_json(root / "stage1/threshold.json")["threshold"]),
            top_k=int(selected["top_k"]),
        )
        rows = [{**row, "decision": float(row["score"]) >= float(selected["thresholds"][fold])} for row in rows]
        all_rows.extend(rows)
    total_hours = sum(total_seconds_by_clip.get(row["clip_id"], 0.0) for row in all_rows) / 3600.0
    candidates = sum(1 for row in all_rows if row["stage1_candidate"])
    accepted = sum(1 for row in all_rows if row["decision"])
    return {
        "outer_test_clips": len(all_rows),
        "total_audio_hours": total_hours,
        "stage1_candidates": candidates,
        "candidate_rate_per_clip": candidates / len(all_rows) if all_rows else None,
        "stage1_candidates_per_hour": candidates / total_hours if total_hours > 0 else None,
        "final_triggers": accepted,
        "metrics": metrics_from_rows(all_rows),
    }


def make_plot(summary: dict[str, Any], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = []
        f1 = []
        median_ms = []
        for item in summary["accuracy_cost_points"]:
            labels.append(item["system"])
            f1.append(item["f1"])
            median_ms.append(item["median_ms"])
        plt.figure(figsize=(7, 4))
        plt.scatter(median_ms, f1)
        for x, y, label in zip(median_ms, f1, labels):
            plt.annotate(label, (x, y), fontsize=8)
        plt.xlabel("Median latency ms")
        plt.ylabel("F1")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=160)
        plt.close()
    except Exception:
        out.write_bytes(b"")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="finetune/experiments/latest_data/runs/nested_zero_shot")
    parser.add_argument("--reports", default="finetune/experiments/latest_data_optimization/reports")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--qwen-encoder-samples", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=10)
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    reports = Path(args.reports)
    selected = read_json(reports / "latest_opt_stage2_selected_config.json")
    manifest = read_jsonl("finetune/experiments/latest_data/shared/balanced_max100_latest_manifest.jsonl")
    sample = choose_subset(manifest, args.sample_size)
    sample_ids = {row["clip_id"] for row in sample}
    run_dir = Path(args.run_root) / "fold_0"
    components, params = measure_heads(run_dir, selected["variant"], sample, device, args.warmups)
    components.append(measure_openwakeword(sample, args.warmups))
    qwen_cache = read_jsonl("finetune/experiments/latest_data/shared/qwen_transcript_cache_balanced_max100_latest.jsonl")
    components.append(qwen_asr_from_cache(qwen_cache, sample_ids))
    components.append(measure_qwen_encoder(sample, "Qwen/Qwen3-ASR-1.7B", args.qwen_encoder_samples, args.warmups))
    rate = candidate_rate(Path(args.run_root), selected)
    qwen_f1 = selected["baseline"]["qwen_exact"]["f1"]
    stage1_f1 = selected["baseline"]["stage1_only_recomputed"]["f1"]
    selected_f1 = selected["test"]["f1"]
    qwen_median = next(row for row in components if row["component"] == "qwen_asr_transcript_from_recorded_cache").get("median_ms")
    stage1_median = next(row for row in components if row["component"] == "stage1_cached_feature_load_plus_head").get("median_ms")
    stage2_median = next(row for row in components if row["component"] == "stage2_cached_qwen_feature_load_plus_head").get("median_ms")
    encoder = next(row for row in components if row["component"] == "qwen_audio_encoder_forward")
    encoder_median = encoder.get("median_ms") or 0.0
    summary = {
        "status": "ok",
        "device": device,
        "sample_size": len(sample),
        "qwen_encoder_samples": args.qwen_encoder_samples,
        "warmups": args.warmups,
        "components": components,
        "parameters": params,
        "candidate_rate": rate,
        "accuracy": {"qwen_exact_f1": qwen_f1, "stage1_only_f1": stage1_f1, "selected_stage2_f1": selected_f1},
        "accuracy_gain": {"stage2_minus_qwen_exact_f1": selected_f1 - qwen_f1, "stage2_minus_stage1_only_f1": selected_f1 - stage1_f1},
        "qwen_call_accounting": {
            "system_a_qwen_weight_copies": 1,
            "system_c_qwen_weight_copies": 1,
            "system_c_extra_encoder_forward_per_stage1_candidate": 1,
            "shared_qwen_hidden_state_path": "blocked_by_runtime_interface",
        },
        "accuracy_cost_points": [
            {"system": "Qwen exact", "f1": qwen_f1, "median_ms": qwen_median},
            {"system": "Stage1 only head path", "f1": stage1_f1, "median_ms": stage1_median},
            {"system": "Selected Stage2 cached path", "f1": selected_f1, "median_ms": (stage1_median or 0.0) + (stage2_median or 0.0)},
            {"system": "Selected Stage2 extra encoder path", "f1": selected_f1, "median_ms": (stage1_median or 0.0) + encoder_median + (stage2_median or 0.0)},
        ],
    }
    write_json(reports / "latest_opt_compute_cost.json", summary)
    flat_rows = []
    for row in components:
        flat_rows.append(row)
    write_csv(reports / "latest_opt_compute_cost_table.csv", flat_rows)
    make_plot(summary, reports / "plots/accuracy_cost_tradeoff.png")
    lines = [
        "# Latest Optimized Compute Cost Report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Device: `{device}`",
        f"- Deterministic subset size: `{len(sample)}`",
        f"- Qwen ASR transcript latency source: recorded transcript cache (`{next(row for row in components if row['component'] == 'qwen_asr_transcript_from_recorded_cache')['n']}` samples)",
        f"- Qwen audio encoder forward measured samples: `{encoder.get('n')}`; status `{encoder.get('status', 'ok')}`",
        f"- Stage2 F1 gain over Qwen exact: `{summary['accuracy_gain']['stage2_minus_qwen_exact_f1']}`",
        f"- Stage2 F1 gain over Stage1-only: `{summary['accuracy_gain']['stage2_minus_stage1_only_f1']}`",
        f"- Stage1 candidates per hour on outer-test clips: `{rate['stage1_candidates_per_hour']}`",
        f"- Current System C uses one Qwen weight copy plus one extra audio-encoder forward per Stage1 candidate.",
        "",
        "| Component | n | median ms | p95 ms |",
        "|---|---:|---:|---:|",
    ]
    for row in components:
        lines.append(f"| {row['component']} | {row.get('n')} | {row.get('median_ms')} | {row.get('p95_ms')} |")
    (reports / "LATEST_OPT_COMPUTE_COST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": "ok", "components": len(components), "qwen_encoder_status": encoder.get("status", "ok")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
