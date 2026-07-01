#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from vigil_latest.utils import write_json


def main() -> int:
    reports = Path("finetune/experiments/latest_data/reports")
    reports.mkdir(parents=True, exist_ok=True)
    result = {"status": "blocked_until_final_model_locked", "false_accepts_per_hour": None}
    write_json(reports / "latest_long_speech_summary.json", result)
    (reports / "LATEST_LONG_SPEECH_FALSE_ACTIVATION_REPORT.md").write_text(
        "# Latest Long-Speech False-Activation Report\n\n"
        "Status: blocked until final model and thresholds are locked. No FAPH result is claimed.\n",
        encoding="utf-8",
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
