#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="finetune/experiments/latest_data/runs/nested_zero_shot")
    parser.add_argument("--out-dir", default="finetune/experiments/latest_data/reports")
    args = parser.parse_args()
    code = subprocess.run(
        [
            sys.executable,
            "finetune/experiments/participant_cv/scripts/aggregate_zero_shot.py",
            "--run-root",
            args.run_root,
            "--out-dir",
            args.out_dir,
        ]
    ).returncode
    out = Path(args.out_dir)
    if (out / "ZERO_SHOT_5FOLD_REPORT.md").exists():
        shutil.copyfile(out / "ZERO_SHOT_5FOLD_REPORT.md", out / "LATEST_NESTED_ZERO_SHOT_5FOLD_REPORT.md")
    if (out / "zero_shot_summary.json").exists():
        shutil.copyfile(out / "zero_shot_summary.json", out / "latest_nested_zero_shot_summary.json")
    if (out / "zero_shot_fold_results.csv").exists():
        shutil.copyfile(out / "zero_shot_fold_results.csv", out / "latest_nested_fold_results.csv")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
