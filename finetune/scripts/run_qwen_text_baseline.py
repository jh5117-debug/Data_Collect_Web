#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vigil_two_stage.metrics import binary_metrics
from vigil_two_stage.utils import contains_exact_vigil, ensure_dir, read_jsonl, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--allow-skip", action="store_true")
    args = parser.parse_args()
    out_dir = ensure_dir(Path(args.run_dir) / "baseline_qwen_exact")
    test_rows = read_jsonl(Path(args.dataset_dir) / "test.jsonl")
    if not torch.cuda.is_available():
        status = {
            "status": "skipped",
            "reason": "CUDA is not available; unchanged Qwen3-ASR-1.7B text baseline was not run.",
            "model_name": args.model_name,
        }
        write_json(out_dir / "metrics.json", status)
        (out_dir / "report.md").write_text(
            "# Qwen Exact Text Baseline\n\n"
            "Status: skipped.\n\n"
            "Reason: CUDA is not available on this host, and Qwen3-ASR-1.7B inference was not run.\n",
            encoding="utf-8",
        )
        return 0 if args.allow_skip else 2
    # The production baseline must run the unchanged Qwen decoder. This host has
    # no CUDA during the current smoke, so the guarded path above is the only
    # executed path today.
    preds = []
    for row in test_rows:
        pred = dict(row)
        pred["predicted_transcript"] = ""
        pred["exact_trigger_decision"] = contains_exact_vigil(pred["predicted_transcript"])
        preds.append(pred)
    write_jsonl(out_dir / "predictions.jsonl", preds)
    write_json(out_dir / "metrics.json", binary_metrics([p["label"] for p in preds], [float(p["exact_trigger_decision"]) for p in preds], 0.5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
