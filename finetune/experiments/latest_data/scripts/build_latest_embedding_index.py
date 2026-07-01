#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-manifest", default="finetune/experiments/latest_data/shared/balanced_max100_latest_manifest.jsonl")
    parser.add_argument("--stage1-manifest", required=True)
    parser.add_argument("--qwen-manifest", required=True)
    parser.add_argument("--out", default="finetune/experiments/latest_data/reports/latest_feature_coverage_report.json")
    args = parser.parse_args()
    cmd = [
        sys.executable,
        "finetune/experiments/participant_cv/scripts/build_feature_index.py",
        "--balanced-manifest",
        args.balanced_manifest,
        "--stage1-manifest",
        args.stage1_manifest,
        "--qwen-manifest",
        args.qwen_manifest,
        "--out",
        args.out,
    ]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
