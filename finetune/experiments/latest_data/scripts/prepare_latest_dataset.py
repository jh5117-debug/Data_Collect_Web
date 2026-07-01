#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from vigil_latest.qwen_format import build_qwen_and_kws_manifests
from vigil_latest.utils import ensure_dir, read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--config", default="finetune/configs/full.yaml")
    parser.add_argument("--output-root", default="finetune/data/processed")
    parser.add_argument("--report-dir", default="finetune/experiments/latest_data/reports")
    args = parser.parse_args()
    cmd = [
        sys.executable,
        "finetune/scripts/prepare_dataset.py",
        args.zip_path,
        "--config",
        args.config,
        "--output-root",
        args.output_root,
    ]
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    dataset_dir = Path(proc.stdout.strip().splitlines()[-1])
    latest_link = Path(args.output_root) / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        if latest_link.is_symlink() or latest_link.is_file():
            latest_link.unlink()
        else:
            shutil.rmtree(latest_link)
    latest_link.symlink_to(dataset_dir.resolve(), target_is_directory=True)
    manifest_summary = build_qwen_and_kws_manifests(dataset_dir / "manifest_all.jsonl", dataset_dir)
    report_dir = ensure_dir(args.report_dir)
    report = {
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_report": read_json(dataset_dir / "dataset_report.json"),
        "qwen_kws_manifest_summary": manifest_summary,
    }
    write_json(report_dir / "latest_dataset_preparation.json", report)
    print(dataset_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
