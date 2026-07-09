#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from rosbag_index import DEFAULT_EXTRACT_DIR, DEFAULT_ZIP, inspect_zip, write_json
from report import inspection_markdown, write_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--reports-dir", type=Path, default=Path("finetune/experiments/false_trigger_regression/reports"))
    args = parser.parse_args()
    inspection = inspect_zip(args.zip, args.extract_dir)
    write_json(args.reports_dir / "rosbag_inspection.json", inspection)
    write_text(args.reports_dir / "ROSBAG_INSPECTION_REPORT.md", inspection_markdown(inspection))
    print(f"wrote {args.reports_dir / 'rosbag_inspection.json'}")


if __name__ == "__main__":
    main()

