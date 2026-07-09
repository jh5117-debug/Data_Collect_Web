#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from report import score_audit_markdown, write_json, write_text
from score_audit import blocked_score_audit, score_manifest_with_current_detector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--model-run-dir",
        type=Path,
        default=Path("finetune/runs/20260624_075127_0fad4c7828149099_full"),
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("finetune/experiments/false_trigger_regression/reports"))
    args = parser.parse_args()
    if args.manifest is None or not args.manifest.exists():
        result = blocked_score_audit("No decoded rosbag_cases.jsonl manifest is available.", args.manifest)
    else:
        result = score_manifest_with_current_detector(args.manifest, args.model_run_dir)
    write_json(args.reports_dir / "false_trigger_score_audit.json", result)
    write_text(args.reports_dir / "FALSE_TRIGGER_SCORE_AUDIT.md", score_audit_markdown(result))
    print(result["status"])
    print(result["diagnosis"]["diagnosis"])


if __name__ == "__main__":
    main()
