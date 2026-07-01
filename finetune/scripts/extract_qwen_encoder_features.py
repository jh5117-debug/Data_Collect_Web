#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from vigil_two_stage.qwen_audio_adapter import FrozenQwenAudioAdapter, QwenAdapterUnavailable
from vigil_two_stage.utils import ensure_dir, read_jsonl, sha256_file, stable_json, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--allow-skip", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = ensure_dir(Path(args.run_dir) / "stage2_qwen_features")
    feature_dir = ensure_dir(out_dir / "features")
    status = {
        "status": "blocked",
        "model_name": config["stage2"]["model_name"],
        "cuda_available": bool(torch.cuda.is_available()),
        "reason": "",
    }
    if not torch.cuda.is_available():
        status["status"] = "skipped"
        status["reason"] = "CUDA is not available; Qwen3-ASR-1.7B encoder features were not extracted."
        write_json(out_dir / "feature_status.json", status)
        (out_dir / "report.md").write_text(
            "# Qwen Encoder Feature Extraction\n\n"
            "Status: skipped.\n\n"
            "Reason: CUDA is not available on this host. Qwen parameters were not loaded or modified.\n",
            encoding="utf-8",
        )
        return 0 if args.allow_skip else 2
    adapter = FrozenQwenAudioAdapter(config["stage2"]["model_name"])
    try:
        adapter.load()
        integrity_before = adapter.integrity()
    except QwenAdapterUnavailable as exc:
        status["reason"] = str(exc)
        write_json(out_dir / "feature_status.json", status)
        return 0 if args.allow_skip else 2
    rows = read_jsonl(Path(args.dataset_dir) / "manifest_all.jsonl")
    feature_rows = []
    diagnostics = []
    latencies = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for row in rows:
        wav_path = Path(row["window_wav_path"])
        key = sha256_file(wav_path) + "_" + stable_json(
            {
                "model_name": config["stage2"]["model_name"],
                "sample_rate": config["audio"]["sample_rate"],
                "window_seconds": config["audio"]["window_seconds"],
                "dtype": "bf16_or_fp16",
            }
        )
        out_path = feature_dir / f"{row['clip_id']}_w{row['window_index']:02d}_{key[:12]}.npz"
        if not out_path.exists():
            started = time.perf_counter()
            try:
                features = adapter.extract_audio_features(str(wav_path.resolve()))
            except QwenAdapterUnavailable as exc:
                status["reason"] = str(exc)
                write_json(out_dir / "feature_status.json", status)
                return 0 if args.allow_skip else 2
            latency = time.perf_counter() - started
            latencies.append(latency)
            arr = features.detach().float().cpu().numpy().astype(np.float16)
            np.savez_compressed(out_path, features=arr)
        else:
            arr = np.load(out_path)["features"]
            latency = None
        if len(diagnostics) < 2 and row["label"] in (0, 1):
            diagnostics.append(
                {
                    "clip_id": row["clip_id"],
                    "label": row["label"],
                    "shape": list(arr.shape),
                    "finite": bool(np.isfinite(arr).all()),
                    "latency_sec": latency,
                }
            )
        feature_row = dict(row)
        feature_row.update(
            {
                "feature_path": str(out_path.resolve()),
                "feature_backend": "frozen_qwen_audio_encoder",
                "feature_dim": int(arr.shape[-1]),
                "qwen_model_name": config["stage2"]["model_name"],
                "qwen_extraction_path": adapter.extraction_path,
            }
        )
        feature_rows.append(feature_row)
    integrity_after = adapter.integrity()
    checksums_unchanged = integrity_before.representative_checksums == integrity_after.representative_checksums
    frozen_ok = integrity_after.trainable_parameters == 0 and checksums_unchanged
    integrity_report = {
        "status": "ok" if frozen_ok else "failed",
        "total_qwen_parameters": integrity_after.total_parameters,
        "trainable_qwen_parameters": integrity_after.trainable_parameters,
        "checksums_unchanged": checksums_unchanged,
        "representative_checksums_before": integrity_before.representative_checksums,
        "representative_checksums_after": integrity_after.representative_checksums,
        "qwen_parameters_modified": not checksums_unchanged,
    }
    write_json(out_dir / "frozen_qwen_integrity.json", integrity_report)
    if not frozen_ok:
        write_json(out_dir / "feature_status.json", {"status": "failed", "reason": "Qwen frozen integrity check failed"})
        return 2
    write_jsonl(out_dir / "qwen_features_manifest.jsonl", feature_rows)
    status.update(
        {
            "status": "ok",
            "reason": "",
            "feature_rows": len(feature_rows),
            "qwen_extraction_path": adapter.extraction_path,
            "diagnostics": diagnostics,
            "mean_extraction_latency_sec": float(sum(latencies) / len(latencies)) if latencies else None,
            "peak_gpu_memory_gb": float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None,
            "total_qwen_parameters": integrity_after.total_parameters,
            "trainable_qwen_parameters": integrity_after.trainable_parameters,
        }
    )
    write_json(out_dir / "feature_status.json", status)
    (out_dir / "report.md").write_text(
        "# Qwen Audio Encoder Feature Extraction\n\n"
        f"- Status: ok\n"
        f"- Model: `{config['stage2']['model_name']}`\n"
        f"- Rows: {len(feature_rows)}\n"
        f"- Extraction path: `{adapter.extraction_path}`\n"
        f"- Mean extraction latency seconds: {status['mean_extraction_latency_sec']}\n"
        f"- Peak GPU memory GB: {status['peak_gpu_memory_gb']}\n"
        f"- Trainable Qwen parameters: {integrity_after.trainable_parameters}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
