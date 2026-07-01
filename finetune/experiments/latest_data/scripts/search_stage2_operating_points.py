#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from vigil_latest.utils import write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="finetune/experiments/latest_data/runs/nested_zero_shot")
    parser.add_argument("--reports", default="finetune/experiments/latest_data/reports")
    args = parser.parse_args()
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    completed = sorted(Path(args.run_root).glob("fold_*/zero_shot_result.json"))
    status = "pending_nested_runs" if len(completed) < 5 else "requires_inner_oof_predictions"
    result = {
        "status": status,
        "completed_outer_folds": len(completed),
        "note": "Operating points must be selected from development/OOF predictions only, never outer-test predictions.",
    }
    write_json(reports / "latest_stage2_operating_points.json", result)
    (reports / "LATEST_STAGE2_OPERATING_POINT_REPORT.md").write_text(
        "# Latest Stage 2 Operating-Point Report\n\n"
        f"- Status: `{status}`\n"
        f"- Completed outer folds: `{len(completed)}`\n"
        "- No operating point is claimed until development-only OOF predictions exist.\n",
        encoding="utf-8",
    )
    return 0 if status != "pending_nested_runs" else 2


if __name__ == "__main__":
    raise SystemExit(main())
