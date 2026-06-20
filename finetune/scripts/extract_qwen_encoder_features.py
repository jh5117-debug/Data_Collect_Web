#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from vigil_two_stage.qwen_audio_adapter import FrozenQwenAudioAdapter, QwenAdapterUnavailable
from vigil_two_stage.utils import ensure_dir, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--allow-skip", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = ensure_dir(Path(args.run_dir) / "stage2_qwen_features")
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
        integrity = adapter.integrity()
    except QwenAdapterUnavailable as exc:
        status["reason"] = str(exc)
        write_json(out_dir / "feature_status.json", status)
        return 0 if args.allow_skip else 2
    status.update(
        {
            "status": "loaded_no_bulk_extraction",
            "total_qwen_parameters": integrity.total_parameters,
            "trainable_qwen_parameters": integrity.trainable_parameters,
            "reason": "Adapter loaded but version-specific audio hidden-state path still needs source inspection.",
        }
    )
    write_json(out_dir / "feature_status.json", status)
    return 0 if args.allow_skip else 2


if __name__ == "__main__":
    raise SystemExit(main())
