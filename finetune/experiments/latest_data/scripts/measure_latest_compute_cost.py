#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from vigil_latest.utils import read_json, read_jsonl, write_json
from vigil_two_stage.stage1_model import Stage1GRUClassifier, count_parameters
from vigil_two_stage.stage2_model import QwenVerifierHead


def load_npz(path: str) -> np.ndarray:
    data = np.load(path)
    return (data["features"] if "features" in data else data[data.files[0]]).astype(np.float32)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean() * 1000.0),
        "median_ms": float(np.median(arr) * 1000.0),
        "p95_ms": float(np.percentile(arr, 95) * 1000.0),
    }


def timed(callable_obj, runs: int) -> list[float]:
    values = []
    for _ in range(runs):
        sync()
        start = time.perf_counter()
        callable_obj()
        sync()
        values.append(time.perf_counter() - start)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reports", default="finetune/experiments/latest_data/reports")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    stage1_rows = read_jsonl(run_dir / "stage1_features_manifest.jsonl")[: max(1, args.runs)]
    qwen_rows = read_jsonl(run_dir / "stage2_qwen_features" / "qwen_features_manifest.jsonl")[: max(1, args.runs)]
    stage1_cfg = read_json(run_dir / "stage1" / "model_config.json")
    stage1 = Stage1GRUClassifier(stage1_cfg["input_dim"], stage1_cfg["gru_hidden_size"], stage1_cfg["gru_layers"], stage1_cfg["dropout"]).to(device)
    stage1.load_state_dict(torch.load(run_dir / "stage1" / "checkpoint_best.pt", map_location=device)["model_state"])
    stage1.eval()

    stage2_ckpt = torch.load(run_dir / "stage2_bce" / "checkpoint_best.pt", map_location=device)
    stage2 = QwenVerifierHead(stage2_ckpt["input_dim"], stage2_ckpt["config"]["projection_dim"], stage2_ckpt["config"]["embedding_dim"]).to(device)
    stage2.load_state_dict(stage2_ckpt["model_state"])
    stage2.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        s1_times = []
        for row in stage1_rows:
            arr = load_npz(row["feature_path"])
            x = torch.from_numpy(arr).unsqueeze(0).to(device)
            lengths = torch.tensor([arr.shape[0]], device=device)
            s1_times.extend(timed(lambda: torch.sigmoid(stage1(x, lengths)), 1))
        s1_peak = float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        s2_times = []
        for row in qwen_rows:
            arr = load_npz(row["feature_path"])
            hidden = torch.from_numpy(arr).unsqueeze(0).to(device)
            mask = torch.ones(1, arr.shape[0], dtype=torch.bool, device=device)
            s2_times.extend(timed(lambda: stage2(hidden, mask), 1))
        s2_peak = float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None

    rows = [
        {"component": "stage1_head", **summarize(s1_times), "peak_allocated_gb": s1_peak},
        {"component": "stage2_head", **summarize(s2_times), "peak_allocated_gb": s2_peak},
    ]
    summary = {
        "status": "partial_head_benchmark",
        "device": device,
        "stage1_parameters": count_parameters(stage1),
        "stage2_parameters": {"total": sum(p.numel() for p in stage2.parameters()), "trainable": sum(p.numel() for p in stage2.parameters() if p.requires_grad)},
        "components": rows,
        "limitations": "Full Qwen ASR/audio-encoder forward latency and full cascade cost were not measured in this run.",
    }
    write_json(reports / "latest_compute_accuracy_tradeoff.json", summary)
    with (reports / "latest_compute_accuracy_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Latest Compute/Accuracy Tradeoff",
        "",
        f"- Status: `{summary['status']}`",
        f"- Device: `{device}`",
        f"- Stage 1 parameters: `{summary['stage1_parameters']}`",
        f"- Stage 2 parameters: `{summary['stage2_parameters']}`",
        "",
        "| Component | Median ms | P95 ms | Peak allocated GB |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['component']} | {row['median_ms']} | {row['p95_ms']} | {row['peak_allocated_gb']} |")
    lines.append("\nFull Qwen ASR/audio-encoder forward latency remains a limitation.")
    (reports / "LATEST_COMPUTE_ACCURACY_TRADEOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
