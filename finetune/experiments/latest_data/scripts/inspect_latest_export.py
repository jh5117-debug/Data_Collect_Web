#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vigil_latest.dataset import assert_counts_close_to_admin, inspect_export_zip
from vigil_latest.utils import read_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--summary-json", default="finetune/experiments/latest_data/reports/admin_summary_before_export.json")
    parser.add_argument("--report-dir", default="finetune/experiments/latest_data/reports")
    args = parser.parse_args()
    inspection = inspect_export_zip(Path(args.zip_path), args.report_dir)
    summary_path = Path(args.summary_json)
    if summary_path.exists():
        assert_counts_close_to_admin(read_json(summary_path), inspection)
    print(json.dumps(inspection, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
