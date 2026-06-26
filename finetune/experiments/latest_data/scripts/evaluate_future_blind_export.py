#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from vigil_latest.dataset import inspect_export_zip
from vigil_latest.utils import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--future-export-zip", required=True)
    parser.add_argument("--known-aliases-json", required=True)
    parser.add_argument("--out", default="finetune/experiments/latest_data/reports/future_blind_eval_placeholder.json")
    args = parser.parse_args()
    inspection = inspect_export_zip(args.future_export_zip, Path(args.out).parent)
    known = set(read_json(args.known_aliases_json).get("participant_aliases", []))
    result = {
        "status": "metadata_checked_thresholds_not_tuned",
        "future_export": inspection["zip_path"],
        "known_aliases_rejected": bool(known),
        "note": "A real blind-test evaluator must compare future participants against the locked development alias set and never tune thresholds.",
    }
    write_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
