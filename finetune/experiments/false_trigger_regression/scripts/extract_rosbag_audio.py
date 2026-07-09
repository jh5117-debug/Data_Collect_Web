#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from audio_extract import attempt_audio_extraction, write_json
from report import audio_status_markdown, write_text
from rosbag_index import DEFAULT_EXTRACT_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-root", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--runs-dir", type=Path, default=Path("finetune/experiments/false_trigger_regression/runs"))
    parser.add_argument("--reports-dir", type=Path, default=Path("finetune/experiments/false_trigger_regression/reports"))
    parser.add_argument("--ros2-decode", action="store_true")
    args = parser.parse_args()
    result = attempt_audio_extraction(args.extracted_root, args.runs_dir, force_ros2_decode=args.ros2_decode)
    write_json(args.reports_dir / "audio_extraction_status.json", result)
    write_text(args.reports_dir / "AUDIO_EXTRACTION_STATUS.md", audio_status_markdown(result))
    print(result["status"])
    print(result.get("reason"))


if __name__ == "__main__":
    main()

