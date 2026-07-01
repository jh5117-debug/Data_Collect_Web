#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vigil_final.latency import cuda_synchronized_timer, summarize_latencies
from vigil_final.memory import cuda_memory_summary, reset_cuda_peak_memory
from vigil_final.qwen_call_counter import QwenCallCounter
from vigil_final.utils import read_json, read_jsonl, write_csv, write_json
from vigil_two_stage.stage1_model import Stage1GRUClassifier, count_parameters
from vigil_two_stage.stage2_model import QwenVerifierHead


def load_npz(path: str) -> np.ndarray:
    data = np.load(path)
    return (data["features"] if "features" in data else data[data.files[0]]).astype(np.float32)


def measure_noop(n: int) -> list[float]:
    values = []
    for _ in range(n):
        with cuda_synchronized_timer() as timer:
            time.sleep(0)
        values.append(timer["wall_seconds"])
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="finetune/experiments/participant_cv/runs/zero_shot/fold_0")
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    stage1_rows = read_jsonl(run_dir / "stage1_features_manifest.jsonl")[: max(1, args.runs)]
    qwen_rows = read_jsonl(run_dir / "stage2_qwen_features" / "qwen_features_manifest.jsonl")[: max(1, args.runs)]

    stage1_cfg = read_json(run_dir / "stage1" / "model_config.json")
    stage1 = Stage1GRUClassifier(stage1_cfg["input_dim"], stage1_cfg["gru_hidden_size"], stage1_cfg["gru_layers"], stage1_cfg["dropout"]).to(device)
    stage1_ckpt = torch.load(run_dir / "stage1" / "checkpoint_best.pt", map_location=device)
    stage1.load_state_dict(stage1_ckpt["model_state"])
    stage1.eval()

    stage2_ckpt = torch.load(run_dir / "stage2_bce" / "checkpoint_best.pt", map_location=device)
    stage2 = QwenVerifierHead(stage2_ckpt["input_dim"], stage2_ckpt["config"]["projection_dim"], stage2_ckpt["config"]["embedding_dim"]).to(device)
    stage2.load_state_dict(stage2_ckpt["model_state"])
    stage2.eval()

    results = []
    reset_cuda_peak_memory()
    stage1_times = []
    with torch.no_grad():
        for row in stage1_rows:
            arr = load_npz(row["feature_path"])
            x = torch.from_numpy(arr).unsqueeze(0).to(device)
            lengths = torch.tensor([arr.shape[0]], device=device)
            with cuda_synchronized_timer() as timer:
                _ = torch.sigmoid(stage1(x, lengths))
            stage1_times.append(timer["wall_seconds"])
    stage1_memory = cuda_memory_summary()
    results.append({"component": "stage1_head", **summarize_latencies(stage1_times), **stage1_memory})

    reset_cuda_peak_memory()
    stage2_times = []
    with torch.no_grad():
        for row in qwen_rows:
            arr = load_npz(row["feature_path"])
            hidden = torch.from_numpy(arr).unsqueeze(0).to(device)
            mask = torch.ones(1, arr.shape[0], dtype=torch.bool, device=device)
            with cuda_synchronized_timer() as timer:
                _ = stage2(hidden, mask)
            stage2_times.append(timer["wall_seconds"])
    stage2_memory = cuda_memory_summary()
    results.append({"component": "stage2_head", **summarize_latencies(stage2_times), **stage2_memory})
    results.append({"component": "timer_overhead", **summarize_latencies(measure_noop(args.runs))})

    counter = QwenCallCounter(loaded_weight_instances=1, transcribe_calls=1, audio_encoder_forward_calls=1, lm_generation_calls=1)
    summary = {
        "status": "partial_head_benchmark",
        "device": device,
        "stage1_parameters": count_parameters(stage1),
        "stage2_parameters": {"total": sum(p.numel() for p in stage2.parameters()), "trainable": sum(p.numel() for p in stage2.parameters() if p.requires_grad)},
        "qwen_call_accounting_current_system_c": {
            "loaded_weight_instances": 1,
            "continuous_asr_forward": 1,
            "additional_stage2_audio_encoder_forward_per_candidate": 1,
            "counter_example": counter.as_dict(),
        },
        "components": results,
    }
    out = Path("finetune/experiments/vigil_final/reports")
    write_json(out / "component_cost_summary.json", summary)
    write_csv(out / "component_cost_table.csv", results)
    (out / "COMPONENT_COST_REPORT.md").write_text(
        "# Component Cost Report\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Device: `{device}`\n"
        f"- Stage 1 parameters: `{summary['stage1_parameters']}`\n"
        f"- Stage 2 parameters: `{summary['stage2_parameters']}`\n"
        "- Current System C uses one loaded Qwen instance, a continuous ASR forward, and an additional Stage 2 audio-encoder forward per candidate.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
