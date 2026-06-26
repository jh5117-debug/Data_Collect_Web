#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--balanced-manifest", default="finetune/experiments/latest_data/shared/balanced_max100_latest_manifest.jsonl")
    parser.add_argument("--folds", default="finetune/experiments/latest_data/shared/latest_participant_folds_5fold.json")
    parser.add_argument("--stage1-manifest", required=True)
    parser.add_argument("--qwen-manifest", required=True)
    parser.add_argument("--qwen-cache", default="finetune/experiments/latest_data/shared/qwen_transcript_cache_balanced_max100_latest.jsonl")
    parser.add_argument("--run-source-dir", required=True)
    parser.add_argument("--config", default="finetune/configs/full.yaml")
    parser.add_argument("--run-root", default="finetune/experiments/latest_data/runs/nested_zero_shot")
    args = parser.parse_args()
    cmd = [
        sys.executable,
        "finetune/experiments/participant_cv/scripts/run_zero_shot_fold.py",
        "--outer-fold",
        str(args.outer_fold),
        "--dataset-dir",
        args.dataset_dir,
        "--balanced-manifest",
        args.balanced_manifest,
        "--folds",
        args.folds,
        "--stage1-manifest",
        args.stage1_manifest,
        "--qwen-manifest",
        args.qwen_manifest,
        "--qwen-cache",
        args.qwen_cache,
        "--run-source-dir",
        args.run_source_dir,
        "--config",
        args.config,
        "--run-root",
        args.run_root,
    ]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
